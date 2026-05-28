import torch

import ntops
from ntops.torch.utils import _cached_make


def fliplr(input: torch.Tensor) -> torch.Tensor:
    r"""Flip tensor in the left-right direction (last dimension).

    Equivalent to ``flip(input, dims=(-1,))``.

    Args:
        input: the input tensor (at least 1D).

    Returns:
        the left-right flipped tensor.
    """
    result = torch.flip(input, (-1,))

    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.fliplr.premake, result.ndim)
    kernel(result, out)

    return out