import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_flip_torch_layer_not_fake():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "flip.py"
    ).read_text(encoding="utf-8")

    assert "torch.flip" not in source


def test_flip_kernel_has_reverse_index():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "kernels" / "flip.py"
    ).read_text(encoding="utf-8")

    assert "dim_size - 1 - i" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape, dims",
    [
        ((10,), (0,)),
        ((4, 8), (1,)),
        ((4, 8), (0, 1)),
        ((2, 3, 4), (2,)),
        ((2, 3, 4), (0, 2)),
        ((16, 32), (0, 1)),
    ],
)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 5e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_flip_matches_torch(shape, dims, dtype, rtol, atol):
    input = torch.randn(shape, device="cuda", dtype=dtype)

    nout = ntops.torch.flip(input, dims)
    rout = torch.flip(input, dims)

    assert nout.shape == rout.shape
    assert nout.dtype == rout.dtype
    assert torch.allclose(nout, rout, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_flip_empty_dims_returns_input():
    input = torch.randn(3, 4, device="cuda")
    out = ntops.torch.flip(input, ())
    assert out is input


@skip_if_cuda_not_available
def test_flip_rejects_int():
    input = torch.randn(3, 4, device="cuda")
    with pytest.raises(TypeError, match="sequence"):
        ntops.torch.flip(input, 0)


@skip_if_cuda_not_available
def test_flip_rejects_duplicate():
    input = torch.randn(3, 4, device="cuda")
    with pytest.raises(RuntimeError, match="duplicate"):
        ntops.torch.flip(input, (0, 0))


@skip_if_cuda_not_available
def test_flip_rejects_out_of_range():
    input = torch.randn(3, 4, device="cuda")
    with pytest.raises(IndexError, match="out of range"):
        ntops.torch.flip(input, (5,))
