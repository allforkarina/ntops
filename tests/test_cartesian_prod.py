import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cartesian_prod_torch_layer_not_fake():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "cartesian_prod.py"
    ).read_text(encoding="utf-8")

    assert "torch.cartesian_prod" not in source


def test_cartesian_prod_kernel_uses_computed_index():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "kernels" / "cartesian_prod.py"
    ).read_text(encoding="utf-8")

    assert "// stride" in source
    assert "% input_size" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "make_tensors",
    [
        lambda device, dtype: (
            torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            torch.tensor([10.0, 20.0, 30.0], device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            torch.tensor([10.0, 20.0], device=device, dtype=dtype),
            torch.tensor([100.0, 200.0, 300.0], device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            torch.tensor([10.0], device=device, dtype=dtype),
            torch.tensor([100.0, 200.0], device=device, dtype=dtype),
            torch.tensor([1000.0, 1000.0, 1000.0], device=device, dtype=dtype),
        ),
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
def test_cartesian_prod_matches_torch(make_tensors, dtype, rtol, atol):
    tensors = make_tensors("cuda", dtype)

    ninetoothed_output = ntops.torch.cartesian_prod(*tensors)
    reference_output = torch.cartesian_prod(*tensors)

    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cartesian_prod_single_input_returns_1d(dtype):
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype)

    ninetoothed_output = ntops.torch.cartesian_prod(t)
    reference_output = torch.cartesian_prod(t)

    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype
    assert torch.allclose(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
def test_cartesian_prod_rejects_non_1d():
    t = torch.randn(2, 3, device="cuda")
    with pytest.raises(RuntimeError, match="1D"):
        ntops.torch.cartesian_prod(t)


@skip_if_cuda_not_available
def test_cartesian_prod_rejects_mixed_dtype():
    t1 = torch.randn(3, device="cuda", dtype=torch.float16)
    t2 = torch.randn(3, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same dtype"):
        ntops.torch.cartesian_prod(t1, t2)


@skip_if_cuda_not_available
def test_cartesian_prod_rejects_mixed_device():
    t1 = torch.randn(3, device="cuda")
    t2 = torch.randn(3, device="cpu")
    with pytest.raises(RuntimeError, match="same device"):
        ntops.torch.cartesian_prod(t1, t2)


@skip_if_cuda_not_available
def test_cartesian_prod_rejects_empty():
    with pytest.raises(RuntimeError, match="at least one"):
        ntops.torch.cartesian_prod()
