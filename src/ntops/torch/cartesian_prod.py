import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def cartesian_prod(*tensors: torch.Tensor):
    r"""Cartesian product of 1D tensors.

    Args:
        tensors: one or more 1D tensors of the same dtype.

    Returns:
        For a single input, returns the input unchanged.
        For multiple inputs, returns a 2D tensor of shape ``(M, N)``
        where *M* is the product of input lengths and *N* is the
        number of inputs.
    """
    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("cartesian_prod expects at least one tensor")

    device = tensors[0].device
    dtype = tensors[0].dtype
    for i, t in enumerate(tensors):
        if t.ndim != 1:
            raise RuntimeError(
                f"cartesian_prod expects 1D tensors, "
                f"but got {t.ndim}D at position {i}"
            )
        if t.device != device:
            raise RuntimeError(
                "cartesian_prod expects all tensors on the same device"
            )
        if t.dtype != dtype:
            raise RuntimeError(
                "cartesian_prod expects all tensors to have the same dtype"
            )

    sizes = tuple(t.shape[0] for t in tensors)
    n = len(tensors)

    if n == 1:
        return tensors[0]

    M = math.prod(sizes)

    output = torch.empty(M, n, dtype=dtype, device=device)

    stride = 1
    for i in range(n - 1, -1, -1):
        s_i = sizes[i]
        col = output.narrow(1, i, 1)

        kernel = _cached_make(
            ntops.kernels.cartesian_prod.premake,
            s_i,
            M,
            stride,
            dtype=dtype,
        )
        kernel(tensors[i], col, stride, s_i)

        stride *= s_i

    return output
