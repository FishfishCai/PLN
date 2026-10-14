# PolarLiteNet (PLN)

Code for **_Single-Pass Neural Inversion of a Weak Difference Signal for Chip-Scale
Spectropolarimetry_**.

PolarLiteNet is a lightweight neural decoder that reconstructs the spectral power $S_0(\lambda)$ and the normalized Stokes parameter $S_1'(\lambda) = S_1/S_0$ of a partially polarized spectrum from the two signal arrays $I_x$ and $I_y$ produced by a polarization-sensitive nanostructural encoder of orthogonal elliptical nanopillar photodiodes. The two arrays are log min-max normalized jointly, so their ratio, which carries the polarization information, reaches the network intact. The network couples two fully connected blocks and a 1-D convolutional block through a residual connection and outputs $S_0$ and $S_1'$ jointly in a single forward pass. It is trained entirely on device-informed simulated spectra with randomized Lorentzian peak parameters and wavelength-dependent polarization angles, optionally augmented with response and measurement noise.

Compared with the two-stage least-squares solver with total-variation regularization (LS+TV), PolarLiteNet is less accurate only in the noise-free limit (median peak-region mean-squared error, MSE, of $S_1'$: $2.7\times10^{-4}$ versus $1.1\times10^{-4}$), overtakes it once the measurement noise reaches 0.01%, and at 1% measurement plus 1% response noise lowers the median peak-region MSE from $1.6\times10^{-2}$ to $3.6\times10^{-4}$ for $S_0$ (~46×) and from $6.7\times10^{-2}$ to $3.3\times10^{-4}$ for $S_1'$ (~200×); the peak region is where $S_0$ exceeds 10% of its maximum. It reduces per-spectrum runtime from 9.9 ms to 0.12 ms on a single CPU thread (~80×).

## Repository structure

```
data/      Measured spectral responsivity of the 36-device encoder
           on a 151-point grid over 400–700 nm
pln/       PolarLiteNet: model, loss, on-the-fly data simulation,
           training/evaluation entry points, per-run configs
           (capacity; five training seeds of the main model,
           the ablations and the noise-augmentation runs)
nnls/      LS+TV baseline: two-stage regularized solver (SciPy),
           regularization tuning and evaluation with timing per
           measurement-noise level
utils/     Shared components: base model, trainer, metric interface,
           config loading
```

## Requirements

Python ≥ 3.12 with `torch`, `numpy`, `scipy`, and `accelerate`.

## Usage

Install the environment with `uv sync`; every entry point reads one JSON run file and runs as a module from the repository root.

```bash
python -m pln.train pln/training/<run>.json     # train one PolarLiteNet run, from the repository root
python -m pln.eval pln/training/<run>.json      # evaluate its best checkpoint on the seeded test set
python -m nnls.train nnls/training/<run>.json   # tune the LS+TV weights at one noise level
python -m nnls.eval nnls/training/<run>.json    # run LS+TV on that noise level's test set
```

Checkpoints, logs, and test results are written to `pln/ckpts/<run>/` and `nnls/ckpts/<run>/`. Architecture variants (hidden widths, convolutional block, residual connection, separate heads) are defined in `pln/modeling/` and the LS+TV weights per noise level in `nnls/modeling/`, while training settings (noise levels `r_noise` and `y_noise`, steps, seeds, test set) are set per run in each `training/` folder.

## Citation

Manuscript in preparation. Citation information will be added upon publication.
