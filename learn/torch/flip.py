from typing import Sequence, Union

import torch

import ntops
from ntops.torch.utils import _cached_make


def flip(input: torch.Tensor, dims: Union[int, Sequence[int]]) -> torch.Tensor:
    r"""Reverse the order of elements along the given dimensions.

    Args:
        input: the input tensor.
        dims:  dimension or dimensions to flip.

    Returns:
        the flipped tensor.
    """
    result = torch.flip(input, dims)

    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.flip.premake, result.ndim)
    kernel(result, out)

    return out