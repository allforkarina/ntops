import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mode(input: Tensor, dim: int = -1, *, out_values=None, out_indices=None):
    r"""Return the mode (most frequent value) along the given dimension.

    Args:
        input:       the input tensor.
        dim:         the dimension to reduce.
        out_values:  optional output tensor for mode values.
        out_indices: optional output tensor for mode indices.

    Returns:
        ``(values, indices)`` tuple of :class:`Tensor`.
    """

    # ntops GPU 加速路径
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        vals, idx = infinicore.ntops.torch.mode(input, dim=dim)
        return Tensor(vals), Tensor(idx)

    # InfiniCore 原生实现
    if out_values is None and out_indices is None:
        vals, idx = _infinicore.mode(input._underlying, dim)
        return Tensor(vals), Tensor(idx)

    _infinicore.mode_(
        out_values._underlying if out_values is not None else None,
        out_indices._underlying if out_indices is not None else None,
        input._underlying,
        dim,
    )
    return out_values, out_indices