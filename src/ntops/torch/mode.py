import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_dim(input: torch.Tensor, dim: int) -> int:
    if input.ndim == 0:
        raise IndexError("mode(): Expected input tensor to have at least one dimension.")

    normalized = dim if dim >= 0 else input.ndim + dim
    if normalized < 0 or normalized >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    return normalized


def _output_shape(input: torch.Tensor, dim: int) -> tuple[int, ...]:
    return tuple(size for axis, size in enumerate(input.shape) if axis != dim)


def mode(input: torch.Tensor, dim: int = -1, keepdim: bool = False):
    r"""Return the mode values and source indices along one dimension."""
    if keepdim:
        raise NotImplementedError("mode does not support keepdim yet")

    dim = _normalize_dim(input, dim)
    if input.shape[dim] == 0:
        raise IndexError(f"mode(): Expected reduction dim {dim} to have non-zero size.")

    shape = _output_shape(input, dim)
    values = torch.empty(shape, dtype=input.dtype, device=input.device)
    indices = torch.empty(shape, dtype=torch.int64, device=input.device)

    kernel = _cached_make(
        ntops.kernels.mode.premake,
        input.ndim,
        dim,
        dtype=input.dtype,
    )
    kernel(input, values, indices)

    return values, indices
