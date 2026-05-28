import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def feature_alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    *,
    out=None,
) -> Tensor:
    r"""Per-channel alpha dropout with affine transform."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and out is None
    ):
        return infinicore.ntops.torch.feature_alpha_dropout(
            input, p=p, training=training
        )

    if out is None:
        return Tensor(
            _infinicore.feature_alpha_dropout(input._underlying, p, training)
        )

    _infinicore.feature_alpha_dropout_(
        out._underlying, input._underlying, p, training
    )
    return out