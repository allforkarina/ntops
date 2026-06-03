import torch

import ntops
from ntops.torch.utils import _cached_make


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Mean squared error loss.

    .. math::
        \text{MSE} = \frac{1}{N} \sum (y_{\text{pred}} - y_{\text{true}})^2

    Args:
        pred:      predicted tensor.
        target:    ground-truth tensor (same shape as ``pred``).
        reduction: ``'none'``, ``'sum'``, or ``'mean'``.

    Returns:
        the loss tensor.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            f"{reduction} is not a valid value for reduction"
        )
    if pred.shape != target.shape:
        raise RuntimeError(
            "mse_loss expects input and target to have the same shape"
        )

    squared_error = torch.empty_like(pred)
    kernel = _cached_make(
        ntops.kernels.mse_loss.premake,
        pred.ndim,
        dtype=pred.dtype,
    )
    kernel(pred, target, squared_error)

    if reduction == "none":
        return squared_error
    if reduction == "sum":
        return torch.sum(squared_error)
    return torch.sum(squared_error) * (1.0 / squared_error.numel())
