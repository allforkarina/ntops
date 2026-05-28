import torch

import ntops
from ntops.torch.utils import _cached_make


def meshgrid(*tensors: torch.Tensor, indexing: str = "ij"):
    r"""Create coordinate grids from 1D tensors.

    Args:
        tensors:  one or more 1D tensors.
        indexing: ``'ij'`` (matrix) or ``'xy'`` (cartesian).

    Returns:
        a tuple of :class:`torch.Tensor` objects, one per input dimension.
    """
    grids = torch.meshgrid(*tensors, indexing=indexing)

    ndim = grids[0].ndim
    kernel = _cached_make(ntops.kernels.meshgrid.premake, ndim)

    outputs = []
    for g in grids:
        out = torch.empty_like(g)
        kernel(g, out)
        outputs.append(out)

    return tuple(outputs)