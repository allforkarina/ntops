import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mse_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    r"""Mean squared error loss."""

    if (
        infinicore.use_ntops
        and pred.device.type in ("cuda", "musa")
    ):
        return infinicore.ntops.torch.mse_loss(
            pred, target, reduction=reduction
        )

    return _infinicore.mse_loss(
        pred._underlying, target._underlying, reduction
    )