from typing import Sequence

import torch

import ntops
from ntops.torch.utils import _cached_make


def flip(input: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    r"""Reverse the order of elements along the given dimensions.

    Args:
        input: the input tensor.
        dims:  sequence of dimensions to flip.

    Returns:
        the flipped tensor.
    """
    if isinstance(dims, int):
        raise TypeError(
            "flip expects dims to be a sequence of ints, not a single int"
        )

    dims = list(dims)
    if not dims:
        return input

    for i, d in enumerate(dims):
        if d < 0:
            dims[i] = input.ndim + d

    if len(set(dims)) != len(dims):
        raise RuntimeError("flip: duplicate dimensions are not allowed")

    for d in dims:
        if d < 0 or d >= input.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{input.ndim}, {input.ndim - 1}], but got {d})"
            )

    out = input
    for dim in dims:
        next_out = torch.empty_like(out)
        kernel = _cached_make(
            ntops.kernels.flip.premake,
            out.ndim,
            dim,
            out.shape[dim],
            dtype=out.dtype,
        )
        kernel(out, next_out)
        out = next_out

    return out
