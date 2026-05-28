import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def column_stack(tensors: Sequence[Tensor], *, out=None) -> Tensor:
    r"""Stack 1D tensors as columns into a 2D tensor.

    Args:
        tensors: a sequence of 1D tensors of the same length.
        out:     optional output tensor.

    Returns:
        a 2D tensor of shape ``(L, N)``.
    """

    # ntops GPU 加速路径
    if infinicore.use_ntops and all(
        t.device.type in ("cuda", "musa") for t in tensors
    ) and out is None:
        return infinicore.ntops.torch.column_stack(tensors)

    # InfiniCore 原生实现
    if out is None:
        return Tensor(_infinicore.column_stack(
            [t._underlying for t in tensors]
        ))

    _infinicore.column_stack_(out._underlying, [t._underlying for t in tensors])
    return out