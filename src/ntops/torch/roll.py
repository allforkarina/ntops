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
    N = input.shape[dims]
    shift = shifts % N

    src = torch.cat(
        [
            input.narrow(dims, N - shift, shift),
            input.narrow(dims, 0, N - shift),
        ],
        dim=dims,
    )

    out = torch.empty_like(src)

    kernel = _cached_make(ntops.kernels.roll.premake, src.ndim)
    kernel(src, out)

    return out