import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def roll(input: Tensor, shifts: int, dims: int = 0, *, out=None) -> Tensor:
    r"""Roll the tensor along the given dimension.

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        input:  the input tensor.
        shifts: the number of places by which the tensor is shifted.
        dims:   dimension along which to roll.
        out:    optional output tensor.
    """

    if not input.is_contiguous():
        input = input.contiguous()

    # ntops GPU 加速路径
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.roll(input, shifts=shifts, dims=dims)

    # InfiniCore 原生实现
    if out is None:
        return Tensor(_infinicore.roll(input._underlying, shifts, dims))

    _infinicore.roll_(out._underlying, input._underlying, shifts, dims)
    return out