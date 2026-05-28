from typing import Sequence, Union

import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def flip(input: Tensor, dims: Union[int, Sequence[int]], *, out=None) -> Tensor:
    r"""Reverse elements along given dimensions."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and out is None
    ):
        return infinicore.ntops.torch.flip(input, dims=dims)

    if out is None:
        return Tensor(_infinicore.flip(input._underlying, dims))

    _infinicore.flip_(out._underlying, input._underlying, dims)
    return out