import json
import sys
from pathlib import Path

from accelerate import Accelerator

from pln.loader import build_test_loaders
from pln.loss import ROIMetric, XMSEMetric
from pln.model import PLNModel
from pln.step import eval_step
from utils.load_args import load_args
from utils.trainer import Trainer


def main(
    config_path: str,
) -> None:
    args = load_args(config_path)
    trainer = Trainer(
        model_class=PLNModel,
        config_path=args.test_checkpoint,
        save_dir=args.save_dir,
        accelerator=Accelerator(cpu=args.test_cpu, mixed_precision=args.mixed_precision),
        seed=args.test_seed,
        deterministic=args.deterministic,
        compile_model=args.compile_model,
        verbose=args.verbose,
        save_logs=args.save_logs,
    )
    results = trainer.evaluate(
        build_test_loaders(args, trainer.device),
        {"test": {"xmse": XMSEMetric(), "roi": ROIMetric(args.n_wavelengths)}},
        eval_step,
    )
    Path(args.save_dir, "test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
