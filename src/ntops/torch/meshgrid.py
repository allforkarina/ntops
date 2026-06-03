import torch

import ntops
from ntops.torch.utils import _cached_make


def meshgrid(*tensors: torch.Tensor, indexing: str = "ij"):
    r"""Create coordinate grids from 1D tensors.

    Args:
        tensors:  one or more 1D tensors.
        indexing: ``'ij'`` (matrix) only; ``'xy'`` is not supported.

    Returns:
        a tuple of :class:`torch.Tensor` objects, one per input tensor.
    """
    if indexing != "ij":
        raise NotImplementedError("meshgrid only supports indexing='ij'")

    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("meshgrid expects at least one tensor")

    device = tensors[0].device
    for i, t in enumerate(tensors):
        if t.ndim != 1:
            raise RuntimeError(
                f"meshgrid expects 1D tensors, but got {t.ndim}D at position {i}"
            )
        if t.device != device:
            raise RuntimeError(
                "meshgrid expects all tensors to be on the same device"
            )

    sizes = tuple(t.shape[0] for t in tensors)
    n = len(tensors)

    outputs = []
    for i, t in enumerate(tensors):
        s_i = sizes[i]
        out_nd = torch.empty(sizes, dtype=t.dtype, device=device)

        kernel = _cached_make(
            ntops.kernels.meshgrid.premake,
            n,
            i,
            s_i,
            dtype=t.dtype,
        )
        kernel(t, out_nd)
        outputs.append(out_nd)

    return tuple(outputs)
