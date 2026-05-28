import torch

import ntops
from ntops.torch.utils import _cached_make


def pixel_unshuffle(input: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    r"""Reverse of pixel shuffle — spatial-to-depth.

    Reshapes ``(N, C, H, W)`` → ``(N, C·r², H/r, W/r)``.

    Args:
        input:            4D tensor ``(N, C, H, W)``.
        downscale_factor: reduction factor *r*.

    Returns:
        the unshuffled tensor.
    """
    result = torch.nn.functional.pixel_unshuffle(input, downscale_factor)

    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.pixel_unshuffle.premake, result.ndim)
    kernel(result, out)

    return out