import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def cartesian_prod(*tensors: Tensor, out=None) -> Tensor:
    r"""Cartesian product of 1D tensors."""

    if infinicore.use_ntops and all(
        t.device.type in ("cuda", "musa") for t in tensors
    ) and out is None:
        return infinicore.ntops.torch.cartesian_prod(*tensors)

    if out is None:
        return Tensor(
            _infinicore.cartesian_prod([t._underlying for t in tensors])
        )

    _infinicore.cartesian_prod_(
        out._underlying, [t._underlying for t in tensors]
    )
    return out