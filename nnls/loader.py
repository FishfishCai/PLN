from types import SimpleNamespace
from typing import Tuple

import numpy as np
import scipy.ndimage
import torch


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
    smoothed = scipy.ndimage.gaussian_filter1d(noise, sigma=sigma, axis=1)
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


def generate_nnls_data(
    args: SimpleNamespace,
    batch_size: int,
    r_noise: float,
    x_noise: float,
    y_noise: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one CPU batch for the LS+TV solver and convert it to ``float64`` arrays.

    The returned responsivities are the ones used to simulate the readouts, so the solver
    sees any response noise exactly.

    Parameters
    ----------
    args : SimpleNamespace
        Run file loaded by ``load_args``, which supplies the spectrum and polarization settings.
    batch_size : int
        Number of samples.
    r_noise : float
        Relative standard deviation of the response noise.
    x_noise : float
        Relative standard deviation of the spectrum noise.
    y_noise : float
        Relative standard deviation of the measurement noise.

    Returns
    -------
    s0_gt : np.ndarray
        Ground-truth ``S0`` of shape ``(batch_size, n_wavelengths)``.
    s1_gt : np.ndarray
        Ground-truth ``S1'`` of the same shape.
    i_x : np.ndarray
        Readouts of the x-oriented array, shape ``(batch_size, n_devices)``.
    i_y : np.ndarray
        Readouts of the y-oriented array, same shape as ``i_x``.
    r_par : np.ndarray
        Parallel responsivity of shape ``(n_devices, n_wavelengths)``.
    r_perp : np.ndarray
        Perpendicular responsivity of the same shape.
    """
    s0, s0_x, s0_y, i_x, i_y, r_par, r_perp = generate_polarized_raw(
        batch_size,
        args.n_wavelengths,
        load_responsivity(args.r_path),
        POL_SENSITIVITY,
        max_n_peaks=args.max_n_peaks,
        width_range=tuple(args.width_range),
        magnitude_range=tuple(args.magnitude_range),
        peak_range=tuple(args.peak_range),
        min_dist=args.min_dist,
        polarization_range=tuple(args.polarization_range),
        r_noise=r_noise,
        x_noise=x_noise,
        y_noise=y_noise,
    )
    s0, s0_x, s0_y, i_x, i_y, r_par, r_perp = (
        t.cpu().detach().numpy().astype(np.float64)
        for t in (s0, s0_x, s0_y, i_x, i_y, r_par, r_perp)
    )
    # S1' is formed in float64 after the cast
    return s0, (s0_x - s0_y) / s0, i_x, i_y, r_par, r_perp
