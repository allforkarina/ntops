import torch

import ntops


def fliplr(input: torch.Tensor) -> torch.Tensor:
    r"""Flip tensor in the left-right direction (last dimension).

    Equivalent to ``flip(input, dims=(-1,))``.

    Args:
        input: the input tensor (at least 1D).

    Returns:
        the left-right flipped tensor.
    """
    return ntops.torch.flip(input, dims=(-1,))