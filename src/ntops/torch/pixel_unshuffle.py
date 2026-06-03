import torch

import ntops
from ntops.torch.utils import _cached_make


def pixel_unshuffle(input: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    r"""Reverse of pixel shuffle — spatial-to-depth.

    Reshapes ``(N, C, H, W)`` → ``(N, C·r², H/r, W/r)`` where
    *r* = ``downscale_factor``.

    Args:
        input:            4D tensor ``(N, C, H, W)``.
        downscale_factor: reduction factor *r*.

    Returns:
        the unshuffled tensor.
    """
    r = downscale_factor
    if r <= 0:
        raise ValueError(
            f"pixel_unshuffle: downscale_factor must be positive, got {r}"
        )
    if input.ndim != 4:
        raise RuntimeError(
            f"pixel_unshuffle: expected 4D input, got {input.ndim}D"
        )

    N, C, H, W = input.shape
    if H % r != 0 or W % r != 0:
        raise RuntimeError(
            f"pixel_unshuffle: spatial dimensions ({H}, {W}) "
            f"must be divisible by downscale_factor ({r})"
        )

    C_out = C * r * r
    H_out = H // r
    W_out = W // r

    output = torch.empty(
        (N, C_out, H_out, W_out),
        dtype=input.dtype,
        device=input.device,
    )

    block_size = 1 if W_out <= 2 else min(W_out, 8)

    kernel = _cached_make(
        ntops.kernels.pixel_unshuffle.premake,
        N,
        C,
        H,
        W,
        r,
        dtype=input.dtype,
        block_size=block_size,
    )
    kernel(input, output, r)

    return output
