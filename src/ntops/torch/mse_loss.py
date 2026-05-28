import torch

import ntops
from ntops.torch.utils import _cached_make


def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Mean squared error loss.

    .. math::
        \text{MSE} = \frac{1}{N} \sum (y_{\text{pred}} - y_{\text{true}})^2

    Args:
        pred:      predicted tensor.
        target:    ground-truth tensor.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.

    Returns:
        the loss tensor.
    """
    # Step 1: kernel 计算逐元素平方误差（真正的 GPU 计算）
    squared_error = torch.empty_like(pred)
    kernel = _cached_make(ntops.kernels.mse_loss.premake, pred.ndim)
    kernel(pred, target, squared_error)

    # Step 2: reduction
    if reduction == "none":
        return squared_error
    if reduction == "sum":
        return squared_error.sum()
    return squared_error.mean()