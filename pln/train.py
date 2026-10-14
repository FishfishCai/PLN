import sys

from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pln.loader import build_train_loaders
from pln.loss import ROIMetric, XMSEMetric
from pln.model import PLNModel
from pln.step import eval_step, train_step
from utils.load_args import load_args
from utils.trainer import Trainer


def main(
    config_path: str,
) -> None:
    args = load_args(config_path)
    trainer = Trainer(
        model_class=PLNModel,
        config_path=args.model_path,
        save_dir=args.save_dir,
        optimizer=lambda model: AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        scheduler=lambda optimizer: CosineAnnealingLR(optimizer, T_max=args.max_steps),
        accelerator=Accelerator(cpu=args.train_cpu, mixed_precision=args.mixed_precision),
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        deterministic=args.deterministic,
        compile_model=args.compile_model,
        verbose=args.verbose,
        save_logs=args.save_logs,
    )
    train_loader, eval_loaders = build_train_loaders(args, trainer.device)
    trainer.train_with_eval(
        max_steps=args.max_steps,
        train_step=train_step,
        train_loader=train_loader,
        train_metrics={"xmse": XMSEMetric()},
        eval_step=eval_step,
        eval_loaders=eval_loaders,
        eval_metrics={"eval": {"xmse": XMSEMetric(), "roi": ROIMetric(args.n_wavelengths)}},
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main(sys.argv[1])
