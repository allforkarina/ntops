import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    *,
    out=None,
) -> Tensor:
    r"""Leaky ReLU activation.

    Args:
        input:           the input tensor.
        negative_slope:  multiplier for negative values (default: ``0.01``).
        out:             optional output tensor.
    """

    # ntops GPU 加速路径
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.leaky_relu(input, negative_slope=negative_slope)

    if out is None:
        return Tensor(
            _infinicore.leaky_relu(input._underlying, negative_slope)
        )

    _infinicore.leaky_relu_(out._underlying, input._underlying, negative_slope)
    return out