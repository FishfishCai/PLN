from typing import Dict, List, Tuple

import numpy as np
import scipy.signal


def find_peak_boundaries(
    spectrum: np.ndarray,
    threshold_ratio: float = 0.10,
) -> Tuple[int, int]:
    """
    Locate the peak region of interest of a 1D spectrum.

    Starting from the highest local maximum, walk outward in both directions until the
    spectrum drops to ``threshold_ratio`` times the peak value or the grid ends.

    Parameters
    ----------
    spectrum : np.ndarray
        1D spectrum, typically a ground-truth ``S0``.
    threshold_ratio : float
        Fraction of the peak height that bounds the ROI. Default is 0.10.

    Returns
    -------
    left_idx : int
        First index of the ROI.
    right_idx : int
        Index one past the last ROI index, so the ROI is ``spectrum[left_idx:right_idx]``.
    """
    spectrum = np.asarray(spectrum)
    peaks = scipy.signal.find_peaks(spectrum)[0]
    peak_idx = peaks[np.argmax(spectrum[peaks])]
    thr = spectrum[peak_idx] * threshold_ratio
    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > thr:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > thr:
        right_idx += 1
    return left_idx, right_idx


def roi_bounds(
    s0_gt: np.ndarray,
    threshold_ratio: float = 0.10,
) -> List[Tuple[int, int]]:
    """Return the ROI bounds of every ground-truth ``S0`` row of a ``(batch, n_wavelengths)`` array."""
    return [find_peak_boundaries(s0_gt[i], threshold_ratio) for i in range(s0_gt.shape[0])]


def roi_mse(
    pred: np.ndarray,
    gt: np.ndarray,
    s0_gt: np.ndarray,
    bounds: List[Tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Compute the per-sample MSE restricted to each sample's peak ROI.

    Parameters
    ----------
    pred : np.ndarray
        Predictions of shape ``(batch, n_wavelengths)`` for ``S0`` or ``S1'``.
    gt : np.ndarray
        Ground truth of the same quantity and shape as ``pred``.
    s0_gt : np.ndarray
        Ground-truth ``S0`` of shape ``(batch, n_wavelengths)``, used to locate each ROI.
    bounds : List[Tuple[int, int]] | None
        Precomputed ROI bounds, derived from ``s0_gt`` when None. Default is None.

    Returns
    -------
    errs : np.ndarray
        Per-sample ROI MSE of shape ``(batch,)``.
    """
    if bounds is None:
        bounds = roi_bounds(s0_gt)
    out = np.empty(pred.shape[0])
    for i, (left, right) in enumerate(bounds):
        out[i] = ((pred[i, left:right] - gt[i, left:right]) ** 2).mean()
    return out


def roi_stats(
    errs: np.ndarray,
) -> Dict[str, float]:
    """Summarize per-sample errors by their median, mean, standard deviation, and quartiles."""
    return {
        "median": float(np.median(errs)),
        "mean": float(np.mean(errs)),
        "std": float(np.std(errs)),
        "q1": float(np.percentile(errs, 25)),
        "q3": float(np.percentile(errs, 75)),
    }
