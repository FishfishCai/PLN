from typing import Dict, Tuple

import torch
from torch import nn

from pln.loss import XMSELoss


_XMSE = XMSELoss()


def log_min_max_scaling(
    y: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    """
    Map each row to ``[0, 1]`` by min-max scaling its logarithm.

    Parameters
    ----------
    y : torch.Tensor
        Positive readouts of shape ``(batch, 2 * n_devices)``, the concatenated
        ``[I_x, I_y]``. Entries below ``eps`` are clamped to ``eps`` before the logarithm.
    eps : float
        Clamp floor and denominator offset. Default is 1e-15.

    Returns
    -------
    y_hat : torch.Tensor
        Normalized readouts, same shape as ``y``.
    """
    safe_log = torch.log(y.clamp_min(eps))
    min_vals = safe_log.min(dim=1, keepdim=True).values
    max_vals = safe_log.max(dim=1, keepdim=True).values
    return (safe_log - min_vals) / (max_vals - min_vals + eps)


def _network_input(
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Concatenate ``I_x`` and ``I_y`` and log min-max scale them jointly, with one minimum and maximum per sample."""
    return log_min_max_scaling(torch.cat((inputs["i_x"], inputs["i_y"]), dim=1))


def train_step(
    model: nn.Module,
    batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Run one training forward pass with the summed squared-error loss.

    Parameters
    ----------
    model : nn.Module
        PolarLiteNet in training mode.
    batch : Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ``({"i_x", "i_y"}, {"x"})`` pair produced by the loader.

    Returns
    -------
    result : Dict[str, torch.Tensor]
        ``"loss"`` for the backward pass plus the detached ``"x_pred"`` and the target
        ``"x"`` for the training metrics.
    """
    inputs, gts = batch
    x_pred = model(_network_input(inputs))["x_pred"]
    return {"loss": _XMSE(x_pred, gts["x"]), "x_pred": x_pred.detach(), "x": gts["x"]}


def eval_step(
    model: nn.Module,
    batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Run one evaluation forward pass on a ``({"i_x", "i_y"}, {"x"})`` batch and return ``"x_pred"`` with its target ``"x"``."""
    inputs, gts = batch
    return {"x_pred": model(_network_input(inputs))["x_pred"], "x": gts["x"]}
