import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_column_stack_torch_layer_does_not_wrap_with_torch_ops():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "column_stack.py"
    ).read_text(encoding="utf-8")

    assert "torch.column_stack" not in source
    assert "torch.stack" not in source
    assert "torch.cat" not in source
    assert "torch.hstack" not in source


def test_column_stack_kernel_remains_slice_copy_kernel():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "kernels" / "column_stack.py"
    ).read_text(encoding="utf-8")

    assert "src.flatten().tile" in source
    assert "dst.flatten().tile" in source
    assert "dst[i] = src[i]" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "make_tensors",
    [
        lambda device, dtype: (
            torch.tensor(1.5, device=device, dtype=dtype),
            torch.tensor(2.5, device=device, dtype=dtype),
        ),
        lambda device, dtype: (torch.randn(5, device=device, dtype=dtype),),
        lambda device, dtype: (
            torch.randn(7, device=device, dtype=dtype),
            torch.randn(7, device=device, dtype=dtype),
            torch.randn(7, device=device, dtype=dtype),
        ),
        lambda device, dtype: (torch.randn(4, 3, device=device, dtype=dtype),),
        lambda device, dtype: (
            torch.randn(4, 2, device=device, dtype=dtype),
            torch.randn(4, 5, device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.randn(4, 2, device=device, dtype=dtype),
            torch.randn(4, device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.randn(2, 3, 4, device=device, dtype=dtype),
            torch.randn(2, 2, 4, device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.randn(8, device=device, dtype=dtype)[::2],
            torch.randn(8, device=device, dtype=dtype)[::2],
        ),
        lambda device, dtype: (
            torch.randn(2, 4, device=device, dtype=dtype).t(),
            torch.randn(4, device=device, dtype=dtype),
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
def test_column_stack_matches_torch(make_tensors, dtype, rtol, atol):
    tensors = make_tensors("cuda", dtype)

    ninetoothed_output = ntops.torch.column_stack(tensors)
    reference_output = torch.column_stack(tensors)

    assert ninetoothed_output.dtype == reference_output.dtype
    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_column_stack_promotes_dtype_like_torch():
    tensors = (
        torch.randn(4, device="cuda", dtype=torch.float16),
        torch.randn(4, device="cuda", dtype=torch.float32),
    )

    ninetoothed_output = ntops.torch.column_stack(tensors)
    reference_output = torch.column_stack(tensors)

    assert ninetoothed_output.dtype == reference_output.dtype
    assert torch.allclose(ninetoothed_output, reference_output, rtol=1e-4, atol=1e-4)


def test_column_stack_rejects_empty_input():
    with pytest.raises(RuntimeError, match="non-empty"):
        ntops.torch.column_stack(())


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "tensors",
    [
        (
            torch.empty(2, 3, 4, device="cuda"),
            torch.empty(2, device="cuda"),
        ),
        (
            torch.empty(3, device="cuda"),
            torch.empty(4, device="cuda"),
        ),
        (
            torch.empty(2, 3, device="cuda"),
            torch.empty(3, device="cuda"),
        ),
    ],
)
def test_column_stack_rejects_incompatible_shapes(tensors):
    with pytest.raises(RuntimeError):
        ntops.torch.column_stack(tensors)


@skip_if_cuda_not_available
def test_column_stack_rejects_mixed_devices():
    tensors = (
        torch.empty(3, device="cuda"),
        torch.empty(3, device="cpu"),
    )

    with pytest.raises(RuntimeError, match="same device"):
        ntops.torch.column_stack(tensors)
