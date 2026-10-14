import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed

from nnls.loader import generate_nnls_data
from nnls.loss import roi_mse, roi_stats
from nnls.model import solve_polarized
from utils.load_args import load_args


def main(
    config_path: str,
) -> None:
    args = load_args(config_path)
    modeling = load_args(args.model_path)
    set_seed(args.test_seed)
    s0_gt, s1_gt, i_x, i_y, r_par, r_perp = generate_nnls_data(
        args,
        batch_size=args.test_batch_size,
        r_noise=args.test_r_noise,
        x_noise=args.test_x_noise,
        y_noise=args.test_y_noise,
    )

    times, s0_hats, s1_hats = [], [], []
    for i in range(s0_gt.shape[0]):
        x_s0, x_s1, _, _, dt = solve_polarized(
            y_x=i_x[i],
            y_y=i_y[i],
            r_par=r_par,
            r_perp=r_perp,
            x_gt=s0_gt[i],
            s1_gt=s1_gt[i],
            lam=modeling.lam,
            gamma=modeling.gamma,
        )
        times.append(dt)
        s0_hats.append(x_s0.astype(np.float32))
        s1_hats.append(x_s1.astype(np.float32))
    s0_hat = np.stack(s0_hats, axis=0)
    s1_hat = np.stack(s1_hats, axis=0)

    results = {
        "mean_time": float(np.mean(times)),
        "roi_S0": roi_stats(roi_mse(s0_hat, s0_gt, s0_gt)),
        "roi_S1": roi_stats(roi_mse(np.clip(s1_hat, -1.0, 1.0), s1_gt, s0_gt)),
    }
    print(f"mean_time  {results['mean_time']:.6f}")
    print(f"roi_S0 {results['roi_S0']}")
    print(f"roi_S1 {results['roi_S1']}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(
        {
            "inputs": {
                "test": {
                    "i_x": torch.from_numpy(i_x.astype(np.float32)),
                    "i_y": torch.from_numpy(i_y.astype(np.float32)),
                },
            },
            "gts": {
                "test": {
                    "S0": torch.from_numpy(s0_gt.astype(np.float32)),
                    "S1": torch.from_numpy(s1_gt.astype(np.float32)),
                },
            },
            "outputs": {
                "test": {
                    "S0": torch.from_numpy(s0_hat),
                    "S1": torch.from_numpy(s1_hat),
                },
            },
        },
        save_dir / "test.pt",
    )


if __name__ == "__main__":
    main(sys.argv[1])
