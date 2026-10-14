from functools import lru_cache
from types import SimpleNamespace
from typing import Dict, Iterator, Tuple

import numpy as np
import scipy.ndimage
import torch
from accelerate.utils import set_seed


# Per-device polarization sensitivity p: R_par = (1 + p) R, R_perp = (1 - p) R.
POL_SENSITIVITY = torch.tensor([
    0.556, 0.481, 0.403, 0.330, 0.296, 0.264, 0.0751, 0.0193,
    7.538e-05, 2.914e-07, 1.127e-09, 0,
    0.556, 0.481, 0.403, 0.330, 0.296, 0.264, 0.0751, 0.0193,
    7.538e-05, 2.914e-07, 1.127e-09, 0,
    0.556, 0.481, 0.403, 0.330, 0.296, 0.264, 0.0751, 0.0193,
    7.538e-05, 2.914e-07, 1.127e-09, 0,
])


def load_responsivity(
    path: str,
) -> torch.Tensor:
    """Read the measured responsivity matrix of shape ``(n_devices, n_wavelengths)``."""
    return torch.from_numpy(np.genfromtxt(path, dtype=np.float32))


def _lorentzian(
    x0: torch.Tensor,
    gamma_sq: torch.Tensor,
    amp: torch.Tensor,
    x_range: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``(batch, max_n_peaks)`` peaks of center ``x0``, squared HWHM ``gamma_sq``, and amplitude ``amp`` on the grid ``x_range``."""
    return (
        (gamma_sq * amp).unsqueeze(-1)
        / ((x_range - x0.unsqueeze(-1)) ** 2 + gamma_sq.unsqueeze(-1))
    )


def generate_x(
    batch: int,
    n_wavelengths: int,
    max_n_peaks: int = 1,
    width_range: Tuple[float, float] = (10.0, 25.0),
    magnitude_range: Tuple[float, float] = (0.6, 1.4),
    peak_range: Tuple[int, int] = (25, 125),
    min_dist: int = 15,
    noise: float = 0.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Simulate spectra ``S0`` as sums of Lorentzian peaks normalized to ``[0, 1]``.

    Parameters
    ----------
    batch : int
        Number of spectra.
    n_wavelengths : int
        Number of wavelength grid points.
    max_n_peaks : int
        Maximum number of peaks. Each spectrum draws its count uniformly from
        ``[1, max_n_peaks]``. Default is 1.
    width_range : Tuple[float, float]
        Range of the half width at half maximum, in grid-index units. Default is
        ``(10.0, 25.0)``.
    magnitude_range : Tuple[float, float]
        Range of the peak amplitude before normalization. Default is ``(0.6, 1.4)``.
    peak_range : Tuple[int, int]
        Range of the peak center, in grid-index units. Default is ``(25, 125)``.
    min_dist : int
        Minimum spacing between consecutive peak centers, in grid-index units.
        Default is 15.
    noise : float
        Standard deviation of multiplicative Gaussian noise on the spectrum, drawn only
        when positive. Default is 0.0.
    device : torch.device
        Device on which the spectra are simulated. Default is CPU.
    dtype : torch.dtype
        Floating-point type of the spectra. Default is ``torch.float32``.

    Returns
    -------
    x : torch.Tensor
        Spectra of shape ``(batch, n_wavelengths)``.
    """
    x_range = torch.arange(n_wavelengths, device=device, dtype=dtype)

    n_peaks = torch.randint(1, max_n_peaks + 1, (batch,), device=device)
    valid_mask = (
        torch.arange(max_n_peaks, device=device).unsqueeze(0)
        < n_peaks.unsqueeze(1)
    )

    widths = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(
        batch, max_n_peaks, dtype=dtype, device=device,
    )
    magnitudes = magnitude_range[0] + (magnitude_range[1] - magnitude_range[0]) * torch.rand(
        batch, max_n_peaks, dtype=dtype, device=device,
    )

    lo, hi = peak_range
    mu = torch.empty(batch, max_n_peaks, dtype=dtype, device=device)
    for k in range(1, max_n_peaks + 1):
        mask_k = (n_peaks == k)
        if not mask_k.any():
            continue
        n_k = mask_k.sum().item()
        eff_hi = hi - (k - 1) * min_dist
        raw = torch.rand(n_k, k, dtype=dtype, device=device) * (eff_hi - lo) + lo
        offsets = torch.arange(k, device=device, dtype=dtype) * min_dist
        mu[mask_k, :k] = raw.sort(dim=1).values + offsets.unsqueeze(0)

    x = _lorentzian(mu, widths ** 2, magnitudes, x_range)
    x = (x * valid_mask.unsqueeze(-1)).sum(dim=1)
    x = x - x.min(dim=1, keepdim=True).values
    x = x / (x.max(dim=1, keepdim=True).values + 1e-20)
    x = x + 1e-20

    if noise > 0:
        x = x + x * torch.randn_like(x) * noise
        x.clamp_min_(1e-8)

    return x


@lru_cache(maxsize=None)
def _smoothing_operator(
    n_wavelengths: int,
    sigma: float,
) -> np.ndarray:
    """Return ``G`` such that ``gaussian_filter1d(v, sigma)`` equals ``G @ v`` for every vector ``v`` of that length."""
    return scipy.ndimage.gaussian_filter1d(np.eye(n_wavelengths), sigma=sigma, axis=0)


def generate_polarization_angle(
    batch: int,
    n_wavelengths: int,
    polarization_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    sigma: float = 50.0,
) -> torch.Tensor:
    """
    Simulate slowly varying polarization angles by Gaussian-smoothing i.i.d. noise.

    Each row is smoothed along the wavelength axis, then rescaled so that its minimum and
    maximum span ``polarization_range``.

    Parameters
    ----------
    batch : int
        Number of angle profiles.
    n_wavelengths : int
        Number of wavelength grid points.
    polarization_range : Tuple[float, float]
        Target range of each profile, in radians. Default is ``(-pi / 2, pi / 2)``.
    sigma : float
        Standard deviation of the Gaussian kernel, in grid-index units. Default is 50.0.

    Returns
    -------
    angles : torch.Tensor
        Angle profiles of shape ``(batch, n_wavelengths)`` in ``float32``.
    """
    noise = np.random.randn(batch, n_wavelengths)
    # reflect-mode Gaussian smoothing is linear, so it is one matrix product over the batch
    smoothed = noise @ _smoothing_operator(n_wavelengths, sigma).T
    smoothed = smoothed - smoothed.min(axis=1, keepdims=True)
    smoothed = smoothed / (smoothed.max(axis=1, keepdims=True) + 1e-20)
    smoothed = smoothed * (polarization_range[1] - polarization_range[0]) + polarization_range[0]
    return torch.tensor(smoothed, dtype=torch.float32)


def generate_polarized_raw(
    batch: int,
    n_wavelengths: int,
    responsivity: torch.Tensor,
    pol_sensitivity: torch.Tensor,
    max_n_peaks: int = 1,
    width_range: Tuple[float, float] = (10.0, 25.0),
    magnitude_range: Tuple[float, float] = (0.6, 1.4),
    peak_range: Tuple[int, int] = (25, 125),
    min_dist: int = 15,
    polarization_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    r_noise: float = 0.02,
    x_noise: float = 0.0,
    y_noise: float = 0.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Simulate polarized spectra and their raw encoder readouts ``I_x`` and ``I_y``.

    Response noise multiplies every entry of ``R_par`` and ``R_perp`` by an independent
    ``1 + N(0, r_noise^2)`` factor, drawn once per call and shared by the whole batch.
    Measurement noise multiplies every readout entry by an independent
    ``1 + N(0, y_noise^2)`` factor. Each noise term is drawn only when its level is
    positive.

    Parameters
    ----------
    batch : int
        Number of samples.
    n_wavelengths : int
        Number of wavelength grid points.
    responsivity : torch.Tensor
        Measured responsivity of shape ``(n_devices, n_wavelengths)``.
    pol_sensitivity : torch.Tensor
        Per-device polarization sensitivity of shape ``(n_devices,)``.
    max_n_peaks : int
        Maximum number of spectral peaks. Default is 1.
    width_range : Tuple[float, float]
        Range of the peak half width, in grid-index units. Default is ``(10.0, 25.0)``.
    magnitude_range : Tuple[float, float]
        Range of the peak amplitude. Default is ``(0.6, 1.4)``.
    peak_range : Tuple[int, int]
        Range of the peak center, in grid-index units. Default is ``(25, 125)``.
    min_dist : int
        Minimum spacing between peak centers. Default is 15.
    polarization_range : Tuple[float, float]
        Range of the polarization angle, in radians. Default is ``(-pi / 2, pi / 2)``.
    r_noise : float
        Relative standard deviation of the response noise. Default is 0.02.
    x_noise : float
        Relative standard deviation of the spectrum noise. Default is 0.0.
    y_noise : float
        Relative standard deviation of the measurement noise. Default is 0.0.
    device : torch.device
        Device on which the data are simulated. Default is CPU.
    dtype : torch.dtype
        Floating-point type of the data. Default is ``torch.float32``.

    Returns
    -------
    s0 : torch.Tensor
        Spectra of shape ``(batch, n_wavelengths)``.
    s0_x : torch.Tensor
        x-polarized component ``cos^2(theta) S0``, same shape as ``s0``.
    s0_y : torch.Tensor
        y-polarized component ``sin^2(theta) S0``, same shape as ``s0``.
    i_x : torch.Tensor
        Readout of the x-oriented array, shape ``(batch, n_devices)``.
    i_y : torch.Tensor
        Readout of the y-oriented array, shape ``(batch, n_devices)``.
    r_par : torch.Tensor
        Parallel responsivity used for the readout, shape ``(n_devices, n_wavelengths)``.
    r_perp : torch.Tensor
        Perpendicular responsivity used for the readout, same shape as ``r_par``.
    """
    s0 = generate_x(
        batch,
        n_wavelengths,
        max_n_peaks=max_n_peaks,
        width_range=width_range,
        magnitude_range=magnitude_range,
        peak_range=peak_range,
        min_dist=min_dist,
        noise=x_noise,
        device=device,
        dtype=dtype,
    )

    pol_angle = generate_polarization_angle(
        batch,
        n_wavelengths,
        polarization_range,
    ).to(device=device, dtype=dtype)

    s0_x = torch.cos(pol_angle) ** 2 * s0
    s0_y = torch.sin(pol_angle) ** 2 * s0

    pol_sen = pol_sensitivity.to(device=device, dtype=dtype).unsqueeze(1).repeat(1, n_wavelengths)
    r_dev = responsivity.to(device=device, dtype=dtype)
    r_par = r_dev + pol_sen * r_dev
    r_perp = r_dev - pol_sen * r_dev

    if r_noise > 0:
        r_par = r_par * (1.0 + torch.randn_like(r_par) * r_noise)
        r_perp = r_perp * (1.0 + torch.randn_like(r_perp) * r_noise)

    i_x = s0_x @ r_par.T + s0_y @ r_perp.T
    i_y = s0_y @ r_par.T + s0_x @ r_perp.T

    if y_noise > 0:
        i_x = i_x + i_x * torch.randn_like(i_x) * y_noise
        i_y = i_y + i_y * torch.randn_like(i_y) * y_noise

    return s0, s0_x, s0_y, i_x, i_y, r_par, r_perp


class SimulatedBatches:
    """Iterable that simulates one fresh ``(inputs, gts)`` batch on every pass."""

    def __init__(
        self,
        *,
        batch_size: int,
        n_wavelengths: int,
        responsivity: torch.Tensor,
        pol_sensitivity: torch.Tensor,
        max_n_peaks: int,
        width_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        peak_range: Tuple[int, int],
        min_dist: int,
        polarization_range: Tuple[float, float],
        r_noise: float,
        x_noise: float,
        y_noise: float,
        device: torch.device,
    ) -> None:
        """
        Store the simulation settings used on every pass.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        n_wavelengths : int
            Number of wavelength grid points.
        responsivity : torch.Tensor
            Measured responsivity of shape ``(n_devices, n_wavelengths)``.
        pol_sensitivity : torch.Tensor
            Per-device polarization sensitivity of shape ``(n_devices,)``.
        max_n_peaks : int
            Maximum number of spectral peaks.
        width_range : Tuple[float, float]
            Range of the peak half width, in grid-index units.
        magnitude_range : Tuple[float, float]
            Range of the peak amplitude.
        peak_range : Tuple[int, int]
            Range of the peak center, in grid-index units.
        min_dist : int
            Minimum spacing between peak centers.
        polarization_range : Tuple[float, float]
            Range of the polarization angle, in radians.
        r_noise : float
            Relative standard deviation of the response noise.
        x_noise : float
            Relative standard deviation of the spectrum noise.
        y_noise : float
            Relative standard deviation of the measurement noise.
        device : torch.device
            Device on which the batches are simulated.
        """
        self.batch_size = batch_size
        self.simulation = dict(
            n_wavelengths=n_wavelengths,
            responsivity=responsivity,
            pol_sensitivity=pol_sensitivity,
            max_n_peaks=max_n_peaks,
            width_range=width_range,
            magnitude_range=magnitude_range,
            peak_range=peak_range,
            min_dist=min_dist,
            polarization_range=polarization_range,
            r_noise=r_noise,
            x_noise=x_noise,
            y_noise=y_noise,
            device=device,
            dtype=torch.float32,
        )

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Simulate one batch of raw readouts and joint ``[S0, S1']`` targets.

        Returns
        -------
        batches : Iterator[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
            A single ``({"i_x", "i_y"}, {"x"})`` pair, whose ``x`` has shape
            ``(batch_size, 2 * n_wavelengths)``.
        """
        s0, s0_x, s0_y, i_x, i_y, _, _ = generate_polarized_raw(
            self.batch_size,
            **self.simulation,
        )
        x = torch.cat((s0, (s0_x - s0_y) / s0), dim=1)
        yield {"i_x": i_x, "i_y": i_y}, {"x": x}


class FixedBatches(SimulatedBatches):
    """Simulated batches that reseed every random generator first, so each pass yields the same test set."""

    def __init__(
        self,
        *,
        seed: int,
        **simulation: object,
    ) -> None:
        """Store the seed applied to Python, NumPy, and torch before every pass and the ``SimulatedBatches`` settings."""
        super().__init__(**simulation)
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Reseed, then yield the single test batch, the same ``({"i_x", "i_y"}, {"x"})`` pair on every pass."""
        set_seed(self.seed)
        yield from super().__iter__()


def _simulation_settings(
    args: SimpleNamespace,
    device: torch.device,
) -> Dict[str, object]:
    """Collect the noise-independent simulation settings of a run file, the keyword arguments shared by every loader."""
    return dict(
        n_wavelengths=args.n_wavelengths,
        responsivity=load_responsivity(args.r_path),
        pol_sensitivity=POL_SENSITIVITY,
        max_n_peaks=args.max_n_peaks,
        width_range=tuple(args.width_range),
        magnitude_range=tuple(args.magnitude_range),
        peak_range=tuple(args.peak_range),
        min_dist=args.min_dist,
        polarization_range=tuple(args.polarization_range),
        device=device,
    )


def build_train_loaders(
    args: SimpleNamespace,
    device: torch.device,
) -> Tuple[SimulatedBatches, Dict[str, SimulatedBatches]]:
    """
    Build the on-the-fly training stream and the validation loader.

    The validation loader draws a fresh batch of ``eval_batch_size`` samples with the
    training noise at every evaluation.

    Parameters
    ----------
    args : SimpleNamespace
        Run file loaded by ``load_args``.
    device : torch.device
        Device on which the batches are simulated.

    Returns
    -------
    train_loader : SimulatedBatches
        Training iterable, one fresh batch per pass.
    eval_loaders : Dict[str, SimulatedBatches]
        ``{"eval": loader}`` used for best-model selection.
    """
    settings = _simulation_settings(args, device)
    noise = dict(r_noise=args.r_noise, x_noise=args.x_noise, y_noise=args.y_noise)
    train_loader = SimulatedBatches(batch_size=args.batch_size, **noise, **settings)
    eval_loaders = {"eval": SimulatedBatches(batch_size=args.eval_batch_size, **noise, **settings)}
    return train_loader, eval_loaders


def build_test_loaders(
    args: SimpleNamespace,
    device: torch.device,
) -> Dict[str, FixedBatches]:
    """Build ``{"test": loader}``, the seeded test loader with the test seed, size, and noise of the run file."""
    return {
        "test": FixedBatches(
            seed=args.test_seed,
            batch_size=args.test_batch_size,
            r_noise=args.test_r_noise,
            x_noise=args.test_x_noise,
            y_noise=args.test_y_noise,
            **_simulation_settings(args, device),
        ),
    }
