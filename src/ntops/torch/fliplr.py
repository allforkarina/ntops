import torch

import ntops


def fliplr(input: torch.Tensor) -> torch.Tensor:
    r"""Flip tensor in the left-right direction (dimension 1).

    Equivalent to ``flip(input, (1,))``.

    Args:
        input: the input tensor (at least 2D).

    Returns:
        the left-right flipped tensor.
    """
    if input.ndim < 2:
        raise RuntimeError(
            f"fliplr expects at least 2D input, got {input.ndim}D"
        )
    return ntops.torch.flip(input, (1,))
