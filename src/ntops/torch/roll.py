"""Roll (cyclic shift) — torch interface layer."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def roll(input: torch.Tensor, shifts: int, dims: int = 0) -> torch.Tensor:
    r"""Roll the tensor along the given dimension.

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        input:  the input tensor.
        shifts: the number of places by which the tensor is shifted.
        dims:   dimension along which to roll.

    Returns:
        the rolled tensor.
    """
    dim = dims if dims >= 0 else input.ndim + dims
    N = input.shape[dim]
    shift = shifts % N

    out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.roll.premake, input.ndim, dim, shift)
    kernel(input, out, shift)

    return out
