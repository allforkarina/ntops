import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_mse_loss_torch_layer_not_fake():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "mse_loss.py"
    ).read_text(encoding="utf-8")

    assert "torch.nn.functional.mse_loss" not in source
    assert ".mean()" not in source
    assert "torch.mean" not in source


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(16,), (4, 8), (2, 3, 5), (3, 2, 4, 3)])
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 5e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_mse_loss_matches_torch(shape, dtype, rtol, atol):
    pred = torch.randn(shape, device="cuda", dtype=dtype)
    target = torch.randn(shape, device="cuda", dtype=dtype)

    for reduction in ("none", "sum", "mean"):
        nout = ntops.torch.mse_loss(pred, target, reduction)
        rout = torch.nn.functional.mse_loss(pred, target, reduction)

        assert nout.shape == rout.shape
        assert nout.dtype == rout.dtype
        assert torch.allclose(nout, rout, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_mse_loss_rejects_invalid_reduction():
    pred = torch.randn(8, device="cuda")
    target = torch.randn(8, device="cuda")
    with pytest.raises(ValueError, match="valid value"):
        ntops.torch.mse_loss(pred, target, "median")


@skip_if_cuda_not_available
def test_mse_loss_rejects_shape_mismatch():
    pred = torch.randn(4, 3, device="cuda")
    target = torch.randn(4, 4, device="cuda")
    with pytest.raises(RuntimeError, match="same shape"):
        ntops.torch.mse_loss(pred, target)
