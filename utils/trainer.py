import json
import sys
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast, set_seed
from pathlib import Path
from timeit import default_timer
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, Iterable, Sequence, Set, Tuple, Type, Union

from .base_model import BaseModel
from .loss import Metric


TrainStepFn = Callable[[Dict[str, nn.Module], Any, Callable[..., None]], Dict[str, Any]]
EvalStepFn = Callable[[Dict[str, nn.Module], Any], Dict[str, Any]]


def to_device(
    obj: Any,
    device: Any,
) -> Any:
    """
    Move every tensor inside a nested structure to a device, leaving other values untouched.

    Parameters
    ----------
    obj : Any
        Tensor, or an arbitrarily nested dict, list, or tuple that may contain tensors.
        Non-tensor leaves such as file paths or sample ids are returned as they are.
    device : Any
        Target device accepted by ``torch.Tensor.to``.

    Returns
    -------
    moved : Any
        New structure of the same shape with every tensor placed on ``device``.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=obj.is_pinned())
    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        moved = [to_device(value, device) for value in obj]
        return type(obj)(*moved) if hasattr(obj, "_fields") else tuple(moved)
    return obj


def scalar(
    value: Any,
) -> Any:
    """
    Convert tensor metric values into plain Python numbers.

    Parameters
    ----------
    value : Any
        Value returned by ``Metric.compute``, possibly a tensor or a mapping of tensors.

    Returns
    -------
    converted : Any
        Python number for a zero-dimensional tensor, nested list otherwise, mapping with
        every entry converted for a dict, and the original object for anything else.
    """
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {name: scalar(item) for name, item in value.items()}
    return value


class Trainer:
    """Step-based trainer that builds named models from a config or checkpoint and drives training and evaluation."""

    def __init__(
        self,
        *,
        model_class: Dict[str, Type[BaseModel]],
        config_path: Union[str, Path],
        save_dir: Union[Path, str],
        optimizer: Callable[[Dict[str, nn.Module]], Dict[str, torch.optim.Optimizer]] | None = None,
        scheduler: Callable[[Dict[str, torch.optim.Optimizer]], Dict[str, object]] | None = None,
        accelerator: Accelerator | None = None,
        grad_clip_norm: Dict[str, float] | None = None,
        ema_decay: Dict[str, float] | None = None,
        seed: int = 0,
        deterministic: bool = False,
        compile_model: bool = False,
        verbose: bool = True,
        save_logs: bool = True,
    ) -> None:
        """
        Build the named models, optimizers, schedulers, and moving averages.

        The file extension of ``config_path`` decides the mode: a ``.json`` file starts a
        fresh run, a ``.pth`` checkpoint resumes every model, optimizer, scheduler, and the
        step counter.

        Parameters
        ----------
        model_class : Dict[str, Type[BaseModel]]
            Class of each named model, which must subclass ``BaseModel``.
        config_path : Union[str, Path]
            Path to a ``.json`` hyper-parameter file or a ``.pth`` checkpoint. The ``.json``
            file maps each model name to the keyword arguments of its class, or to a string
            ``"<checkpoint>.pth:<name>"`` that takes that model from another checkpoint.
        save_dir : Union[Path, str]
            Directory that receives checkpoints and ``logs.txt``.
        optimizer : Callable[[Dict[str, nn.Module]], Dict[str, torch.optim.Optimizer]] | None
            Factory that builds the named optimizers from the named models. A parameter in
            no optimizer takes no gradient, and a model with no such parameter at all is
            frozen, in evaluation mode.
            Training is unavailable when omitted, which is the evaluation-only mode.
            Default is None.
        scheduler : Callable[[Dict[str, torch.optim.Optimizer]], Dict[str, object]] | None
            Factory that builds learning-rate schedulers from the named optimizers, each
            under the name of the optimizer it follows and stepped once per update of that
            optimizer. Default is None.
        accelerator : Accelerator | None
            Accelerator carrying the device and mixed-precision settings. A default one
            is created when omitted. Default is None.
        grad_clip_norm : Dict[str, float] | None
            Maximum gradient norm per optimizer name, applied to the parameters of that
            optimizer before each of its updates. Optimizers left out are not clipped.
            Default is None.
        ema_decay : Dict[str, float] | None
            Decay per model name of the exponential moving average of its weights, updated
            after each step that updates the model, stored in checkpoints, and used
            for evaluation. Models left out keep no average. Default is None.
        seed : int
            Base seed for Python, NumPy, and torch. Each process is offset by its rank so
            that data augmentation differs across ranks. Default is 0.
        deterministic : bool
            Whether to request deterministic algorithms, which can slow training down.
            Default is False.
        compile_model : bool
            Whether to wrap every model with ``torch.compile``. Default is False.
        verbose : bool
            Whether the main process prints progress. Default is True.
        save_logs : bool
            Whether the main process appends log lines to ``logs.txt``. Default is True.
        """
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        self.device = self.accelerator.device
        self._is_main = self.accelerator.is_main_process
        self.grad_clip_norm = grad_clip_norm or {}
        self.ema_decay = ema_decay or {}
        # Every process builds from the same seed, so the models and their averages start identical.
        set_seed(seed, deterministic=deterministic)

        self.step = 0

        self.verbose = verbose and self._is_main
        self.save_logs = save_logs and self._is_main
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.log_path = self.save_dir / "logs.txt"
        if self._is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.save_logs:
            self.log_path.touch()

        self._modules, ckpt = self._build_models(config_path, model_class, optimizer is not None)
        self.optimizers = optimizer(self._modules) if optimizer is not None else {}
        self.schedulers = scheduler(self.optimizers) if scheduler is not None else {}
        self._check("scheduler", self.schedulers, "optimizer", self.optimizers)
        self._check("grad_clip_norm", self.grad_clip_norm, "optimizer", self.optimizers)
        self._params = {
            name: [
                param
                for group in built.param_groups
                for param in group["params"]
                if param.requires_grad
            ]
            for name, built in self.optimizers.items()
        }
        held = {name: {id(param) for param in params} for name, params in self._params.items()}
        self._owners = {
            name: {
                optimizer_name
                for optimizer_name, ids in held.items()
                if any(id(param) in ids for param in module.parameters())
            }
            for name, module in self._modules.items()
        }
        self._frozen = {name for name, owners in self._owners.items() if not owners}
        trained = set().union(*held.values())
        for module in self._modules.values():
            for param in module.parameters():
                if id(param) not in trained:
                    param.requires_grad_(False)
        if ckpt is not None:
            self._resume_state(ckpt)
        self._ema_modules = self._build_ema(ckpt, model_class)

        self.models = {
            name: self.accelerator.prepare_model(
                torch.compile(module) if compile_model else module,
                evaluation_mode=name in self._frozen,
            )
            for name, module in self._modules.items()
        }
        self.ema_models = {}
        for name, module in self._ema_modules.items():
            ema = self.accelerator.prepare_model(module, evaluation_mode=True)
            self.ema_models[name] = torch.compile(ema) if compile_model else ema

        self._loaders: Dict[DataLoader, DataLoader] = {}
        self._stepped: Set[str] = set()
        self._pending: Dict[str, Tuple[Any, int]] = {}
        self._window_loss: Dict[str, Any] = {}
        self._window_updates: Dict[str, int] = {}
        self._train_mode()

        # Reseed from the resume point once everything is built, so a run never replays past draws.
        set_seed(seed + self.step, device_specific=True, deterministic=deterministic)

    def train(
        self,
        *,
        max_steps: int,
        train_step: TrainStepFn,
        train_loader: Iterable[Any],
        train_metrics: Dict[str, Metric] | None = None,
        eval_step: Dict[str, EvalStepFn] | None = None,
        eval_loaders: Dict[str, Iterable[Any]] | None = None,
        eval_metrics: Dict[str, Dict[str, Metric]] | None = None,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
    ) -> None:
        """
        Train until ``max_steps`` steps, logging, evaluating, and saving on schedule.

        A step is one call of ``train_step``, or as many calls as it takes to apply every
        gradient they accumulate, one batch per call. Each of the three intervals also fires on the
        final step.

        Parameters
        ----------
        max_steps : int
            Global step at which training stops, counted from zero across resumes.
        train_step : TrainStepFn
            Callable receiving the named models, one batch, and ``Trainer.update``. It
            computes each loss and passes it to ``update`` in the order the step
            requires, then returns a dict forwarded to ``train_metrics``.
        train_loader : Iterable[Any]
            Any iterable of batches. A finite one is iterated again when exhausted, so it
            must not be a one-shot iterator.
        train_metrics : Dict[str, Metric] | None
            Metrics recorded per logging window and reset afterwards. Default is None.
        eval_step : Dict[str, EvalStepFn] | None
            Callable per evaluation loader, receiving the named models and one batch and
            returning a dict forwarded to the metrics of that loader. Default is None.
        eval_loaders : Dict[str, Iterable[Any]] | None
            Named evaluation sets, each of which must be finite. Default is None.
        eval_metrics : Dict[str, Dict[str, Metric]] | None
            Metrics per evaluation loader, keyed the same way as ``eval_loaders``.
            Default is None.
        log_interval : int
            Number of steps between log lines. Default is 100.
        eval_interval : int
            Number of steps between evaluations. Default is 1000.
        save_interval : int
            Number of steps between checkpoints. Default is 1000.
        """
        if not self.optimizers:
            raise ValueError(
                "Training requires optimizers. Pass optimizer=... at construction."
            )
        train_metrics = train_metrics or {}
        eval_step = eval_step or {}
        eval_loaders = eval_loaders or {}
        eval_metrics = eval_metrics or {}

        for name, value in (
            ("max_steps", max_steps),
            ("log_interval", log_interval),
            ("eval_interval", eval_interval),
            ("save_interval", save_interval),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        self._check("eval_loaders", eval_loaders, "eval_step", eval_step)
        self._check("eval_loaders", eval_loaders, "eval_metrics", eval_metrics)

        message = (
            f"Training from step {self.step} to {max_steps} on {self.device}"
            f" (models={sorted(self.models)}, optimizers={sorted(self.optimizers)},"
            f" processes={self.accelerator.num_processes})"
        )
        if eval_loaders:
            message += f", evaluating {sorted(eval_loaders)} every {eval_interval} steps"
        if self.verbose:
            print(message)
            sys.stdout.flush()

        train_loader = self._prepare_loader(train_loader)
        batches = iter(train_loader)

        self._train_mode()
        for metric in train_metrics.values():
            metric.reset()
        self._window_loss.clear()
        self._window_updates.clear()
        window_start = default_timer()

        while self.step < max_steps:
            batch = next(batches, None)
            if batch is None:
                batches = iter(train_loader)
                batch = next(batches, None)
            if batch is None:
                raise ValueError(
                    "train_loader yielded no batches after a restart, so it is empty or a "
                    "one-shot iterator."
                )
            with self.accelerator.autocast():
                result = train_step(self.models, to_device(batch, self.device), self.update)
            for metric in train_metrics.values():
                metric.update(**result)
            # Gradients still waiting to be applied mean the step continues on the next batch.
            if self._pending:
                continue
            self._update_ema()
            self.step += 1
            last = self.step >= max_steps

            if self.step % log_interval == 0 or last:
                self._log({
                    "time": default_timer() - window_start,
                    **{
                        label: float(total / self._window_updates[label])
                        for label, total in self._window_loss.items()
                    },
                    "train": {n: scalar(m.compute()) for n, m in train_metrics.items()},
                    "lr": {n: o.param_groups[-1]["lr"] for n, o in self.optimizers.items()},
                })
                for metric in train_metrics.values():
                    metric.reset()
                self._window_loss.clear()
                self._window_updates.clear()
                window_start = default_timer()

            if eval_loaders and (self.step % eval_interval == 0 or last):
                self.evaluate(
                    eval_step=eval_step,
                    eval_loaders=eval_loaders,
                    eval_metrics=eval_metrics,
                )
                self._train_mode()

            if self.step % save_interval == 0 or last:
                self._save_state(f"step_{self.step}")

    def update(
        self,
        loss: torch.Tensor,
        optimizers: Sequence[str] | None = None,
        accumulate: int = 1,
    ) -> None:
        """
        Apply one loss to a set of optimizers, the only way a training step changes weights.

        The backward pass reaches only the parameters of those optimizers, which then step
        once with their schedulers and clear their gradients. A non-finite loss or gradient
        norm raises before it reaches the weights. Under a gradient scaler, a non-finite
        gradient instead makes the scaler skip the step, and the schedulers with it.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss of this update.
        optimizers : Sequence[str] | None
            Names of the optimizers this loss updates, whose parameters must not overlap
            within one update. All optimizers when omitted. Default is None.
        accumulate : int
            Number of calls on the same optimizers that share one update. Each call adds the
            gradient of ``loss / accumulate``, and the last one applies the sum, which is the
            mean loss the update logs. The calls may lie in one call of the training step or
            in consecutive ones, which then form a single step. Default is 1.
        """
        names = sorted(self.optimizers if optimizers is None else optimizers)
        label = "loss" if len(names) == len(self.optimizers) else "loss/" + "+".join(names)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {label} after {self.step} steps. Stopping before the backward pass."
            )
        scaler = self.accelerator.scaler
        # The step runs under mixed-precision casting, which the backward pass and the updates leave.
        with torch.autocast(self.device.type, enabled=False):
            loss = loss / accumulate
            (loss if scaler is None else scaler.scale(loss)).backward(
                inputs=[param for name in names for param in self._params[name]]
            )
            total, n_calls = self._pending.get(label, (0.0, 0))
            self._pending[label] = (total + loss.detach(), n_calls + 1)
            if n_calls + 1 < accumulate:
                return
            for name in names:
                built = self.optimizers[name]
                max_norm = self.grad_clip_norm.get(name, float("inf"))
                if scaler is None:
                    grad_norm = nn.utils.clip_grad_norm_(self._params[name], max_norm)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm of optimizer {name!r} after {self.step} "
                            "steps. Stopping before the optimizer update."
                        )
                    built.step()
                else:
                    if name in self.grad_clip_norm:
                        scaler.unscale_(built)
                        nn.utils.clip_grad_norm_(self._params[name], max_norm)
                    scaler.step(built)
                built.zero_grad(set_to_none=True)
        self._window_loss[label] = self._window_loss.get(label, 0.0) + self._pending.pop(label)[0]
        self._window_updates[label] = self._window_updates.get(label, 0) + 1
        if scaler is not None:
            # The scale falls exactly when the scaler skipped the step, which leaves the weights unchanged.
            scale = scaler.get_scale()
            scaler.update()
            if scaler.get_scale() < scale:
                return
        for name in names:
            if name in self.schedulers:
                self.schedulers[name].step()
        self._stepped.update(names)

    def evaluate(
        self,
        *,
        eval_step: Dict[str, EvalStepFn],
        eval_loaders: Dict[str, Iterable[Any]],
        eval_metrics: Dict[str, Dict[str, Metric]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run every evaluation loader to exhaustion and reduce its metrics.

        Each model that keeps a moving average of its weights is replaced by that average.
        Everything runs in evaluation mode without gradients, gathering batch results across
        processes so every rank computes the same value over the full data.

        Parameters
        ----------
        eval_step : Dict[str, EvalStepFn]
            Callable per evaluation loader, receiving the named models and one batch and
            returning a dict forwarded to the metrics of that loader. With several processes
            and a ``DataLoader``, every value is a tensor whose first dimension is the batch.
        eval_loaders : Dict[str, Iterable[Any]]
            Named evaluation sets, each of which must be finite.
        eval_metrics : Dict[str, Dict[str, Metric]]
            Metrics per evaluation loader, keyed the same way as ``eval_loaders``.

        Returns
        -------
        eval_results : Dict[str, Dict[str, Any]]
            Mapping from loader name to its metric name and reduced value.
        """
        if self.accelerator.num_processes > 1:
            for module in self._ema_modules.values():
                for buffer in module.buffers():
                    broadcast(buffer, from_process=0)
        models = {**self.models, **self.ema_models}
        for model in models.values():
            model.eval()
        eval_results: Dict[str, Dict[str, Any]] = {}

        with torch.no_grad(), self.accelerator.autocast():
            for loader_name, eval_loader in eval_loaders.items():
                step_fn = eval_step[loader_name]
                loader_metrics = eval_metrics[loader_name]
                for metric in loader_metrics.values():
                    metric.reset()
                for batch in self._prepare_loader(eval_loader):
                    result = step_fn(models, to_device(batch, self.device))
                    # Only a DataLoader is split across processes. Any other iterable runs whole on each.
                    if self.accelerator.num_processes > 1 and isinstance(eval_loader, DataLoader):
                        result = self.accelerator.gather_for_metrics(result)
                    for metric in loader_metrics.values():
                        metric.update(**result)
                eval_results[loader_name] = {
                    name: scalar(metric.compute()) for name, metric in loader_metrics.items()
                }

        if eval_results:
            self._log(eval_results, prefix="Eval ")
        return eval_results

    def _build_models(
        self,
        config_path: Union[str, Path],
        model_class: Dict[str, Type[BaseModel]],
        training: bool,
    ) -> Tuple[Dict[str, BaseModel], dict | None]:
        """
        Construct every model on the device from a JSON hyper-parameter file or a checkpoint.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to a ``.json`` hyper-parameter file or a ``.pth`` checkpoint.
        model_class : Dict[str, Type[BaseModel]]
            Class of each named model, which must subclass ``BaseModel``.
        training : bool
            Whether the trainer will train. A resumed model then takes its raw weights,
            so that training continues from them. Otherwise it takes its moving average
            when the checkpoint stores one.

        Returns
        -------
        modules : Dict[str, BaseModel]
            Named models on the device.
        ckpt : dict | None
            Loaded checkpoint for a ``.pth`` source, None for a ``.json`` source.
        """
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            ckpt = None
            sources = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".pth":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            sources = ckpt["models"]
        else:
            raise ValueError(f"Unsupported config file type: {suffix}")
        modules = {}
        for name, cls in model_class.items():
            source = sources[name]
            if isinstance(source, str):
                other_path, _, other_name = source.rpartition(":")
                other = torch.load(
                    Path(other_path).expanduser(), map_location="cpu", weights_only=False
                )
                module = self._entry(cls, other["models"][other_name], evaluation=True)
            elif ckpt is not None:
                module = self._entry(cls, source, evaluation=not training)
            else:
                module = cls(**source)
            if not isinstance(module, BaseModel):
                raise TypeError(
                    f"{cls.__qualname__} must subclass BaseModel so its config can be "
                    "written to the checkpoint."
                )
            # Checkpoints store the config as JSON, so an unserialisable one must fail here.
            json.dumps(module.config)
            modules[name] = module.to(self.device)
        if ckpt is not None and self.verbose:
            print(f"Loaded models {sorted(modules)} from checkpoint.")
            sys.stdout.flush()
        return modules, ckpt

    @staticmethod
    def _entry(
        model_class: Type[BaseModel],
        entry: dict,
        evaluation: bool,
    ) -> BaseModel:
        """
        Rebuild one model from its checkpoint entry.

        Parameters
        ----------
        model_class : Type[BaseModel]
            Class used to build the model, checked against the class the entry stores.
        entry : dict
            Entry of one model in a checkpoint, holding its class path, config, weights,
            and moving average when one was kept.
        evaluation : bool
            Whether to load the weights evaluation uses, the moving average when the
            entry stores one, instead of the raw weights.

        Returns
        -------
        module : BaseModel
            Model on the CPU with the chosen weights loaded.
        """
        saved_name = str(entry["model_class"]).rsplit(".", 1)[-1]
        if saved_name != model_class.__name__:
            raise ValueError(
                f'Checkpoint holds model class "{saved_name}" but model_class is '
                f'"{model_class.__name__}".'
            )
        module = model_class(**json.loads(entry["config"]))
        module.load_state_dict(
            entry["ema"] if (evaluation and "ema" in entry) else entry["model"]
        )
        return module

    def _build_ema(
        self,
        ckpt: dict | None,
        model_class: Dict[str, Type[BaseModel]],
    ) -> Dict[str, BaseModel]:
        """
        Build the moving average of every model named in ``ema_decay``.

        An average starts from the one the resumed checkpoint stores, or from the current
        weights of its model in a fresh run.

        Parameters
        ----------
        ckpt : dict | None
            Checkpoint returned by ``_build_models``, None for a fresh run.
        model_class : Dict[str, Type[BaseModel]]
            Class of each named model.

        Returns
        -------
        ema_modules : Dict[str, BaseModel]
            Averages on the device, without gradients and in evaluation mode.
        """
        ema_modules = {}
        for name in self.ema_decay:
            module = self._modules[name]
            ema = model_class[name](**module.config).requires_grad_(False).eval()
            ema.load_state_dict(
                ckpt["models"][name]["ema"] if ckpt is not None else module.state_dict()
            )
            ema_modules[name] = ema.to(self.device)
        return ema_modules

    def _resume_state(
        self,
        ckpt: dict,
    ) -> None:
        """
        Restore training progress and the state of every optimizer and scheduler.

        Parameters
        ----------
        ckpt : dict
            Checkpoint returned by ``_build_models``.
        """
        self.step = int(ckpt["step"])
        for name, built in self.optimizers.items():
            built.load_state_dict(ckpt["optimizers"][name])
        for name, built in self.schedulers.items():
            built.load_state_dict(ckpt["schedulers"][name])
        if self.accelerator.scaler is not None and "scaler" in ckpt:
            self.accelerator.scaler.load_state_dict(ckpt["scaler"])
        if self.verbose:
            print(f"Loaded checkpoint at step {self.step}.")
            sys.stdout.flush()

    def _save_state(
        self,
        filename: str,
    ) -> None:
        """
        Write every model and the training state to ``<save_dir>/<filename>.pth``.

        All processes synchronise before and after, and only the main one writes the file.
        Each model is stored with its class path, config, weights, and moving average, under
        keys that match a plain module.

        Parameters
        ----------
        filename : str
            File stem without extension, such as ``"step_1000"``.
        """
        self.accelerator.wait_for_everyone()
        if self._is_main:
            ckpt = {
                "models": {
                    name: {
                        "model": module.state_dict(),
                        "config": json.dumps(module.config),
                        "model_class": f"{type(module).__module__}.{type(module).__qualname__}",
                    }
                    for name, module in self._modules.items()
                },
                "optimizers": {n: built.state_dict() for n, built in self.optimizers.items()},
                "schedulers": {n: built.state_dict() for n, built in self.schedulers.items()},
                "step": int(self.step),
            }
            for name, module in self._ema_modules.items():
                ckpt["models"][name]["ema"] = module.state_dict()
            if self.accelerator.scaler is not None:
                ckpt["scaler"] = self.accelerator.scaler.state_dict()
            save_path = self.save_dir / f"{filename}.pth"
            torch.save(ckpt, save_path)
            if self.verbose:
                print(f"Saved training state to {save_path}")
                sys.stdout.flush()
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _update_ema(
        self,
    ) -> None:
        """Move the moving average of every model this step updated toward its weights and copy its buffers."""
        for name, ema in self._ema_modules.items():
            if self._stepped.isdisjoint(self._owners[name]):
                continue
            module = self._modules[name]
            torch._foreach_lerp_(
                list(ema.parameters()),
                list(module.parameters()),
                1.0 - self.ema_decay[name],
            )
            for ema_buffer, buffer in zip(ema.buffers(), module.buffers()):
                ema_buffer.copy_(buffer)
        self._stepped.clear()

    def _train_mode(
        self,
    ) -> None:
        """Put every trained model in training mode and every frozen one in evaluation mode."""
        for name, model in self.models.items():
            model.train(name not in self._frozen)

    def _prepare_loader(
        self,
        loader: Iterable[Any],
    ) -> Iterable[Any]:
        """
        Prepare a ``DataLoader`` with the accelerator once, returning other iterables as they are.

        Parameters
        ----------
        loader : Iterable[Any]
            Training or evaluation iterable.

        Returns
        -------
        prepared : Iterable[Any]
            Prepared loader, the same object on every call with the same ``DataLoader``.
        """
        if not isinstance(loader, DataLoader):
            return loader
        if loader not in self._loaders:
            self._loaders[loader] = self.accelerator.prepare(loader)
        return self._loaders[loader]

    @staticmethod
    def _check(
        kind: str,
        names: Iterable[str],
        known_kind: str,
        known: Iterable[str],
    ) -> None:
        """
        Raise when a mapping names something that its counterpart does not hold.

        Parameters
        ----------
        kind : str
            What ``names`` are, for the error message.
        names : Iterable[str]
            Names to check.
        known_kind : str
            What ``known`` are, for the error message.
        known : Iterable[str]
            Names that exist.
        """
        missing = sorted(set(names) - set(known))
        if missing:
            raise ValueError(
                f"{kind} names {missing}, which {known_kind} does not hold. "
                f"It holds {sorted(known)}."
            )

    def _log(
        self,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """
        Format one line of metrics and send it to stdout and the log file.

        Nested mappings are flattened with ``/`` separators.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Possibly nested mapping from name to value.
        prefix : str
            Text inserted before the step number in the line tag. Default is an empty string.
        """
        if not (self.verbose or self.save_logs):
            return

        def walk(
            values: Dict[str, Any],
            parent: str = "",
        ) -> Iterable[str]:
            """
            Yield flattened ``name=value`` fragments from a possibly nested mapping.

            Parameters
            ----------
            values : Dict[str, Any]
                Possibly nested mapping from name to value.
            parent : str
                Path of the enclosing mapping, joined to child names with ``/``.
                Default is an empty string.

            Returns
            -------
            fragments : Iterable[str]
                Lazily generated ``name=value`` strings, with numbers formatted to six decimals.
            """
            for name, value in values.items():
                full = f"{parent}/{name}" if parent else name
                if isinstance(value, dict):
                    yield from walk(value, full)
                elif isinstance(value, (int, float)):
                    yield f"{full}={value:.6f}"
                else:
                    yield f"{full}={value}"

        message = f"[{prefix}step {self.step}] " + ", ".join(walk(metrics))
        if self.verbose:
            print(message)
            sys.stdout.flush()
        if self.save_logs:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
