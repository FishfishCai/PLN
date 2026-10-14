import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import lsq_linear


def _build_d_matrix(
    n: int,
) -> np.ndarray:
    """Return the ``(n - 1, n)`` forward-difference operator used as the TV regularizer."""
    d_mat = np.eye(n) + np.diag(-np.ones(n - 1), k=1)
    return d_mat[:-1, :]


def solve_s0(
    y_x: np.ndarray,
    y_y: np.ndarray,
    r_par: np.ndarray,
    r_perp: np.ndarray,
    x_gt: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Recover ``S0`` from ``I_x + I_y`` by TV-regularized non-negative least squares.

    The stacked system ``vstack(R_par + R_perp, lam * D) S0 = vstack(I_x + I_y, 0)`` is
    divided by its largest absolute coefficient before ``lsq_linear`` solves it with
    ``S0 >= 0``.

    Parameters
    ----------
    y_x : np.ndarray
        Readout of the x-oriented array, shape ``(n_devices,)``.
    y_y : np.ndarray
        Readout of the y-oriented array, shape ``(n_devices,)``.
    r_par : np.ndarray
        Parallel responsivity of shape ``(n_devices, n_wavelengths)``.
    r_perp : np.ndarray
        Perpendicular responsivity of the same shape.
    x_gt : np.ndarray
        Ground-truth ``S0`` of shape ``(n_wavelengths,)``, used only for the error.
    lam : float
        TV regularization weight.

    Returns
    -------
    x_s0 : np.ndarray
        Estimated ``S0`` of shape ``(n_wavelengths,)``.
    err : float
        Full-band MSE of ``x_s0`` against ``x_gt``.
    elapsed : float
        Wall-clock solve time in seconds.
    """
    time_begin = time.time()
    n_wl = r_par.shape[1]
    d_mat = _build_d_matrix(n_wl)
    r_sum = np.vstack((r_par + r_perp, lam * d_mat))
    y_sum = np.concatenate((y_x + y_y, np.zeros(n_wl - 1)))
    scale = np.abs(r_sum).max()
    r_sum = r_sum / scale
    y_sum = y_sum / scale
    result = lsq_linear(r_sum, y_sum, bounds=(0, np.inf))
    x_s0 = result.x
    err = ((x_s0 - x_gt) ** 2).mean()
    return x_s0, err, time.time() - time_begin


def solve_s1(
    y_x: np.ndarray,
    y_y: np.ndarray,
    r_par: np.ndarray,
    r_perp: np.ndarray,
    x_s0_hat: np.ndarray,
    s1_gt: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Recover ``S1'`` from ``I_x - I_y`` given an estimated ``S0`` by TV-regularized least squares.

    The stacked system
    ``vstack((R_par - R_perp) diag(S0_hat), gamma * D) S1' = vstack(I_x - I_y, 0)`` is
    divided by its largest absolute coefficient before ``lstsq`` solves it.

    Parameters
    ----------
    y_x : np.ndarray
        Readout of the x-oriented array, shape ``(n_devices,)``.
    y_y : np.ndarray
        Readout of the y-oriented array, shape ``(n_devices,)``.
    r_par : np.ndarray
        Parallel responsivity of shape ``(n_devices, n_wavelengths)``.
    r_perp : np.ndarray
        Perpendicular responsivity of the same shape.
    x_s0_hat : np.ndarray
        First-stage ``S0`` estimate of shape ``(n_wavelengths,)``.
    s1_gt : np.ndarray
        Ground-truth ``S1'`` of shape ``(n_wavelengths,)``, used only for the error.
    gamma : float
        TV regularization weight.

    Returns
    -------
    x_s1 : np.ndarray
        Estimated ``S1'`` of shape ``(n_wavelengths,)``, which may fall outside ``[-1, 1]``.
    err : float
        Full-band MSE of ``x_s1`` against ``s1_gt``.
    elapsed : float
        Wall-clock solve time in seconds.
    """
    time_begin = time.time()
    n_wl = r_par.shape[1]
    d_mat = _build_d_matrix(n_wl)
    r_diff = np.vstack(((r_par - r_perp) @ np.diag(x_s0_hat), gamma * d_mat))
    y_diff = np.concatenate((y_x - y_y, np.zeros(n_wl - 1)))
    scale = np.abs(r_diff).max()
    r_diff = r_diff / scale
    y_diff = y_diff / scale
    x_s1, *_ = np.linalg.lstsq(r_diff, y_diff, rcond=None)
    err = ((x_s1 - s1_gt) ** 2).mean()
    return x_s1, err, time.time() - time_begin


def solve_polarized(
    y_x: np.ndarray,
    y_y: np.ndarray,
    r_par: np.ndarray,
    r_perp: np.ndarray,
    x_gt: np.ndarray,
    s1_gt: np.ndarray,
    lam: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Run the two-stage LS+TV reconstruction, ``S0`` first and then ``S1'`` from that estimate.

    Parameters
    ----------
    y_x : np.ndarray
        Readout of the x-oriented array, shape ``(n_devices,)``.
    y_y : np.ndarray
        Readout of the y-oriented array, shape ``(n_devices,)``.
    r_par : np.ndarray
        Parallel responsivity of shape ``(n_devices, n_wavelengths)``.
    r_perp : np.ndarray
        Perpendicular responsivity of the same shape.
    x_gt : np.ndarray
        Ground-truth ``S0``, used only for the error.
    s1_gt : np.ndarray
        Ground-truth ``S1'``, used only for the error.
    lam : float
        TV weight of the ``S0`` stage.
    gamma : float
        TV weight of the ``S1'`` stage.

    Returns
    -------
    x_s0 : np.ndarray
        Estimated ``S0``.
    x_s1 : np.ndarray
        Estimated ``S1'``, which may fall outside ``[-1, 1]``.
    s0_err : float
        Full-band MSE of ``x_s0``.
    s1_err : float
        Full-band MSE of ``x_s1``.
    elapsed : float
        Total wall-clock time of both stages in seconds.
    """
    x_s0, s0_err, t0 = solve_s0(
        y_x=y_x,
        y_y=y_y,
        r_par=r_par,
        r_perp=r_perp,
        x_gt=x_gt,
        lam=lam,
    )
    x_s1, s1_err, t1 = solve_s1(
        y_x=y_x,
        y_y=y_y,
        r_par=r_par,
        r_perp=r_perp,
        x_s0_hat=x_s0,
        s1_gt=s1_gt,
        gamma=gamma,
    )
    return x_s0, x_s1, s0_err, s1_err, t0 + t1


def optimal_lambda(
    ys_x: List[np.ndarray],
    ys_y: List[np.ndarray],
    r_par: np.ndarray,
    r_perp: np.ndarray,
    xs_gt: List[np.ndarray],
    lambdas: np.ndarray,
) -> Tuple[float, List[np.ndarray]]:
    """
    Pick the ``S0``-stage weight with the lowest summed full-band ``S0`` MSE over a sample set.

    Parameters
    ----------
    ys_x : List[np.ndarray]
        Per-sample readouts of the x-oriented array.
    ys_y : List[np.ndarray]
        Per-sample readouts of the y-oriented array.
    r_par : np.ndarray
        Parallel responsivity of shape ``(n_devices, n_wavelengths)``.
    r_perp : np.ndarray
        Perpendicular responsivity of the same shape.
    xs_gt : List[np.ndarray]
        Per-sample ground-truth ``S0``.
    lambdas : np.ndarray
        Candidate weights, scanned in order. Ties keep the earliest.

    Returns
    -------
    best_lam : float
        Selected weight.
    best_s0_list : List[np.ndarray]
        Per-sample ``S0`` estimates at ``best_lam``.
    """
    best_err = np.inf
    best_lam = lambdas[0]
    best_s0_list = None

    for lam in lambdas:
        err_total = 0.0
        s0_list = []
        for i in range(len(ys_x)):
            x_s0, err, _ = solve_s0(
                y_x=ys_x[i],
                y_y=ys_y[i],
                r_par=r_par,
                r_perp=r_perp,
                x_gt=xs_gt[i],
                lam=lam,
            )
            err_total += err
            s0_list.append(x_s0)
        if err_total < best_err:
            best_err = err_total
            best_lam = lam
            best_s0_list = s0_list

    return best_lam, best_s0_list


def optimal_gamma(
    ys_x: List[np.ndarray],
    ys_y: List[np.ndarray],
    r_par: np.ndarray,
    r_perp: np.ndarray,
    s0_hats: List[np.ndarray],
    s1_gts: List[np.ndarray],
    gammas: np.ndarray,
) -> float:
    """
    Pick the ``S1'``-stage weight with the lowest summed full-band ``S1'`` MSE given fixed ``S0`` estimates.

    Parameters
    ----------
    ys_x : List[np.ndarray]
        Per-sample readouts of the x-oriented array.
    ys_y : List[np.ndarray]
        Per-sample readouts of the y-oriented array.
    r_par : np.ndarray
        Parallel responsivity of shape ``(n_devices, n_wavelengths)``.
    r_perp : np.ndarray
        Perpendicular responsivity of the same shape.
    s0_hats : List[np.ndarray]
        Per-sample first-stage ``S0`` estimates.
    s1_gts : List[np.ndarray]
        Per-sample ground-truth ``S1'``.
    gammas : np.ndarray
        Candidate weights, scanned in order. Ties keep the earliest.

    Returns
    -------
    best_gam : float
        Selected weight.
    """
    best_err = np.inf
    best_gam = gammas[0]

    for gam in gammas:
        err_total = 0.0
        for i in range(len(ys_x)):
            _, err, _ = solve_s1(
                y_x=ys_x[i],
                y_y=ys_y[i],
                r_par=r_par,
                r_perp=r_perp,
                x_s0_hat=s0_hats[i],
                s1_gt=s1_gts[i],
                gamma=gam,
            )
            err_total += err
        if err_total < best_err:
            best_err = err_total
            best_gam = gam

    return best_gam
