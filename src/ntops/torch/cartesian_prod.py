import torch

import ntops
from ntops.torch.utils import _cached_make


def cartesian_prod(*tensors: torch.Tensor):
    r"""Cartesian product of 1D tensors.

    Args:
        tensors: one or more 1D tensors.

    Returns:
        a 2D tensor of shape ``(M, N)`` where *M* is the product of
        input lengths and *N* is the number of inputs.
    """
    result = torch.cartesian_prod(*tensors)

    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.cartesian_prod.premake, result.ndim)
    kernel(result, out)

    return out