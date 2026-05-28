import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fliplr(input: Tensor, *, out=None) -> Tensor:
    r"""Left-right flip (last dimension)."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and out is None
    ):
        return infinicore.ntops.torch.fliplr(input)

    if out is None:
        return Tensor(_infinicore.fliplr(input._underlying))

    _infinicore.fliplr_(out._underlying, input._underlying)
    return out