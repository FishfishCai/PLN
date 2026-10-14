import json
import sys
from pathlib import Path

import numpy as np
from accelerate.utils import set_seed

from nnls.loader import generate_nnls_data
from nnls.model import optimal_gamma, optimal_lambda
from utils.load_args import load_args


def main(
    config_path: str,
) -> None:
    args = load_args(config_path)
    set_seed(args.seed)
    s0_gt, s1_gt, i_x, i_y, r_par, r_perp = generate_nnls_data(
        args,
        batch_size=args.batch_size,
        r_noise=args.r_noise,
        x_noise=args.x_noise,
        y_noise=args.y_noise,
    )
    grid = np.logspace(
        start=args.grid_log10_min,
        stop=args.grid_log10_max,
        num=args.grid_size,
    )
    best_lam, s0_hats = optimal_lambda(
        ys_x=list(i_x),
        ys_y=list(i_y),
        r_par=r_par,
        r_perp=r_perp,
        xs_gt=list(s0_gt),
        lambdas=grid,
    )
    print(f"Best lambda for S0: {best_lam:.3e}")
    best_gam = optimal_gamma(
        ys_x=list(i_x),
        ys_y=list(i_y),
        r_par=r_par,
        r_perp=r_perp,
        s0_hats=s0_hats,
        s1_gts=list(s1_gt),
        gammas=grid,
    )
    print(f"Best gamma for S1: {best_gam:.3e}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "regularization.json").write_text(
        json.dumps({"lam": float(best_lam), "gamma": float(best_gam)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(sys.argv[1])
