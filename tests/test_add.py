import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import gauss, generate_arguments


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_add_kernel_contains_computed_index_experiment():
    source = (_PROJECT_ROOT / "src" / "ntops" / "kernels" / "add.py").read_text(
        encoding="utf-8"
    )

    assert "src = (i + 1) % output.shape[0]" in source
    assert "other[src]" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_add(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    alpha = gauss()

    ninetoothed_output = ntops.torch.add(input, other, alpha=alpha)
    reference_output = torch.add(input, other, alpha=alpha)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
