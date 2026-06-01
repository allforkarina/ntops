import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_roll_torch_layer_does_not_wrap_with_torch_ops():
    source = (_PROJECT_ROOT / "src" / "ntops" / "torch" / "roll.py").read_text(
        encoding="utf-8"
    )

    assert "torch.cat" not in source
    assert ".narrow(" not in source
    assert "torch.roll" not in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape, shifts, dims",
    [
        ((8,), 2, 0),
        ((4, 8), 3, 1),
        ((4, 8), -2, 1),
        ((2, 3, 8), 5, 2),
        ((2, 8, 3), 3, 1),
    ],
)
@pytest.mark.parametrize("dtype, rtol, atol", [(torch.float32, 1e-4, 1e-4)])
def test_roll(shape, shifts, dims, dtype, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device="cuda")

    ninetoothed_output = ntops.torch.roll(input, shifts, dims=dims)
    reference_output = torch.roll(input, shifts, dims=dims)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
