import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pixel_unshuffle_torch_layer_not_fake():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "pixel_unshuffle.py"
    ).read_text(encoding="utf-8")

    assert "torch.pixel_unshuffle" not in source
    assert "torch.nn.functional.pixel_unshuffle" not in source


def test_pixel_unshuffle_kernel_has_index_mapping():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "kernels" / "pixel_unshuffle.py"
    ).read_text(encoding="utf-8")

    assert "h * r + i" in source
    assert "w * r + j" in source
    assert "c_out // r_sq" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "N, C, H, W, r",
    [
        (1, 8, 4, 6, 2),
        (2, 4, 6, 4, 2),
        (1, 3, 9, 6, 3),
        (2, 2, 4, 8, 2),
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
def test_pixel_unshuffle_matches_torch(N, C, H, W, r, dtype, rtol, atol):
    input = torch.randn(N, C, H, W, device="cuda", dtype=dtype)

    ninetoothed_output = ntops.torch.pixel_unshuffle(input, r)
    reference_output = torch.nn.functional.pixel_unshuffle(input, r)

    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype
    assert torch.allclose(
        ninetoothed_output, reference_output, rtol=rtol, atol=atol
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "N, C, H, W, r",
    [
        (2, 4, 8, 4, 2),
        (1, 3, 6, 9, 3),
    ],
)
def test_pixel_unshuffle_non_contiguous(N, C, H, W, r):
    full = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    input = full.permute(0, 1, 3, 2)  # non-contiguous spatial dims

    ninetoothed_output = ntops.torch.pixel_unshuffle(input, r)
    reference_output = torch.nn.functional.pixel_unshuffle(input, r)

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
def test_pixel_unshuffle_rejects_non_4d():
    input = torch.randn(2, 3, 4, 5, 6, device="cuda")
    with pytest.raises(RuntimeError, match="4D"):
        ntops.torch.pixel_unshuffle(input, 2)


@skip_if_cuda_not_available
def test_pixel_unshuffle_rejects_bad_factor():
    input = torch.randn(2, 3, 4, 6, device="cuda")
    with pytest.raises(ValueError, match="positive"):
        ntops.torch.pixel_unshuffle(input, 0)
    with pytest.raises(ValueError, match="positive"):
        ntops.torch.pixel_unshuffle(input, -2)


@skip_if_cuda_not_available
def test_pixel_unshuffle_rejects_indivisible():
    input = torch.randn(2, 3, 5, 6, device="cuda")
    with pytest.raises(RuntimeError, match="divisible"):
        ntops.torch.pixel_unshuffle(input, 2)
