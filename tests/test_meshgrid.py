import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_meshgrid_torch_layer_does_not_use_torch_meshgrid():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "meshgrid.py"
    ).read_text(encoding="utf-8")

    assert "torch.meshgrid" not in source


def test_meshgrid_kernel_is_not_empty_copy():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "kernels" / "meshgrid.py"
    ).read_text(encoding="utf-8")

    assert "output_2d[j, k] = input_1d[j]" in source
    assert "permute" in source
    assert "flatten" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "make_tensors",
    [
        lambda device, dtype: (
            torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype),
            torch.tensor([4.0, 5.0], device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            torch.tensor([3.0, 4.0, 5.0], device=device, dtype=dtype),
            torch.tensor([6.0, 7.0], device=device, dtype=dtype),
        ),
        lambda device, dtype: (
            torch.tensor([1.0, 2.0], device=device, dtype=dtype),
            torch.tensor([3.0, 4.0, 5.0], device=device, dtype=dtype),
            torch.tensor([6.0], device=device, dtype=dtype),
            torch.tensor([7.0, 8.0], device=device, dtype=dtype),
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
def test_meshgrid_matches_torch(make_tensors, dtype, rtol, atol):
    tensors = make_tensors("cuda", dtype)
    indexing = "ij"

    ninetoothed_outputs = ntops.torch.meshgrid(*tensors, indexing=indexing)
    reference_outputs = torch.meshgrid(*tensors, indexing=indexing)

    assert len(ninetoothed_outputs) == len(reference_outputs)
    for nout, rout in zip(ninetoothed_outputs, reference_outputs):
        assert nout.shape == rout.shape
        assert nout.dtype == rout.dtype
        assert torch.allclose(nout, rout, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_meshgrid_rejects_non_1d():
    t = torch.randn(2, 3, device="cuda")
    with pytest.raises(RuntimeError, match="1D"):
        ntops.torch.meshgrid(t)


@skip_if_cuda_not_available
def test_meshgrid_rejects_xy_indexing():
    t = torch.randn(3, device="cuda")
    with pytest.raises(NotImplementedError, match="indexing"):
        ntops.torch.meshgrid(t, indexing="xy")


@skip_if_cuda_not_available
def test_meshgrid_rejects_empty():
    with pytest.raises(RuntimeError, match="at least one"):
        ntops.torch.meshgrid()
