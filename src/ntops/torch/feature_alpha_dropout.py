import math
import random

import torch

import ntops
from ntops.torch.utils import _cached_make


_ALPHA = -1.7580993408473766


def _affine_coeffs(p: float) -> tuple[float, float]:
    a = (1 - p + p * _ALPHA**2) ** -0.5
    b = -a * _ALPHA * p
    return a, b


def feature_alpha_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = False,
) -> torch.Tensor:
    r"""Per-channel alpha dropout with affine transform.

    During training, each channel is independently masked with
    probability *p*.  Dropped positions are filled with :math:`\alpha'`
    and a linear transform :math:`a \cdot x + b` is applied to preserve
    mean and variance.

    Args:
        input:  tensor of shape ``(N, C, ...)``.
        p:      dropout probability (default: ``0.5``).
        training: apply dropout only when ``True`` (default: ``False``).

    Returns:
        the transformed tensor.
    """
    if not 0 <= p <= 1:
        raise ValueError(
            f"feature_alpha_dropout: p must be in [0, 1], got {p}"
        )
    if input.ndim < 2:
        raise RuntimeError(
            f"feature_alpha_dropout: expected at least 2D input, "
            f"got {input.ndim}D"
        )

    if not training or p == 0:
        return input

    was_2d = input.ndim == 2
    if was_2d:
        input = input.unsqueeze(2)

    N, C = input.shape[0], input.shape[1]
    a, b = _affine_coeffs(p)

    out = torch.empty_like(input)
    seed = random.getrandbits(31)

    kernel = _cached_make(
        ntops.kernels.feature_alpha_dropout.premake,
        input.ndim,
        N,
        C,
        dtype=input.dtype,
    )
    kernel(input, out, seed, p, _ALPHA, a, b)

    if was_2d:
        return out.squeeze(2)
    return out
