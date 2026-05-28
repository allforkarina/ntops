import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def meshgrid(*tensors: Tensor, indexing: str = "ij"):
    r"""Create coordinate grids from 1D tensors."""

    if infinicore.use_ntops and all(
        t.device.type in ("cuda", "musa") for t in tensors
    ):
        return infinicore.ntops.torch.meshgrid(*tensors, indexing=indexing)

    return _infinicore.meshgrid(
        [t._underlying for t in tensors], indexing
    )