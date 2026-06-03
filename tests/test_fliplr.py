import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_fliplr_torch_layer_not_fake():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "fliplr.py"
    ).read_text(encoding="utf-8")

    assert "torch.fliplr" not in source
    assert "torch.flip" not in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape",
    [(4, 8), (2, 3, 5), (3, 4, 2, 3)],
)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 5e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_fliplr_matches_torch(shape, dtype, rtol, atol):
    input = torch.randn(shape, device="cuda", dtype=dtype)

    nout = ntops.torch.fliplr(input)
    rout = torch.fliplr(input)

    assert nout.shape == rout.shape
    assert nout.dtype == rout.dtype
    assert torch.allclose(nout, rout, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_fliplr_rejects_1d():
    input = torch.randn(5, device="cuda")
    with pytest.raises(RuntimeError, match="at least 2D"):
        ntops.torch.fliplr(input)
