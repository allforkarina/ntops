import torch

import ntops
from ntops.torch.utils import _cached_make


def leaky_relu(input: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    r"""Leaky ReLU activation.

    .. math::
        \text{LeakyReLU}(x) = \begin{cases}
            x,            & \text{if } x > 0  \\
            \alpha \cdot x, & \text{otherwise}
        \end{cases}

    Args:
        input:           the input tensor.
        negative_slope:  multiplier :math:`\alpha` for negative values
                         (default: ``0.01``).

    Returns:
        the activated tensor.
    """
    out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.leaky_relu.premake, input.ndim)
    kernel(input, negative_slope, out)

    return out