import torch

import ntops
from ntops.torch.utils import _cached_make


def feature_alpha_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
) -> torch.Tensor:
    r"""Per-channel alpha dropout with affine transform.

    During training, each **channel** is independently masked with
    probability *p*.  Dropped positions are filled with :math:`\alpha'`
    and a linear transform :math:`a \cdot x + b` is applied to preserve
    mean and variance.

    Args:
        input:    input tensor of shape ``(N, C, ...)``.
        p:        dropout probability (default: ``0.5``).
        training: apply dropout only when ``True``.

    Returns:
        the transformed tensor.
    """
    if not training or p == 0:
        return input

    # Per-channel mask → broadcast to input shape
    N, C = input.shape[0], input.shape[1]
    mask = torch.bernoulli(
        torch.full((N, C, 1, 1), 1 - p, device=input.device)
    ).expand_as(input)

    # Affine coefficients that preserve mean / variance
    alpha = -1.7580993408473766
    a = (1 - p + p * alpha**2) ** -0.5
    b = -a * alpha * p

    out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.feature_alpha_dropout.premake, input.ndim)
    kernel(input, mask, alpha, a, b, out)

    return out