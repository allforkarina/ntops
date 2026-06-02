import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_mode_torch_layer_does_not_compute_with_torch_reductions():
    source = (_PROJECT_ROOT / "src" / "ntops" / "torch" / "mode.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "torch.mode",
        "torch.sort",
        "torch.argmax",
        "scatter_add",
        ".argmax(",
        ".cumsum(",
        ".movedim(",
    ]

    for pattern in forbidden:
        assert pattern not in source


def test_mode_kernel_contains_real_frequency_scan():
    source = (_PROJECT_ROOT / "src" / "ntops" / "kernels" / "mode.py").read_text(
        encoding="utf-8"
    )

    assert "candidate_count" in source
    assert "best_count" in source
    assert "elem == candidate" in source
    assert "ntl.cast(best_value, input.dtype)" in source
    assert "indices = best_index" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "data, dim",
    [
        (
            torch.tensor(
                [
                    [1, 2, 2, 1],
                    [3, 3, 2, 2],
                ],
                dtype=torch.float32,
            ),
            1,
        ),
        (
            torch.tensor(
                [
                    [2, 1, 1, 2],
                    [2, 2, 1, 1],
                ],
                dtype=torch.float32,
            ),
            -1,
        ),
        (
            torch.tensor(
                [
                    [[1, 1, 2], [4, 5, 5]],
                    [[1, 2, 2], [4, 4, 5]],
                ],
                dtype=torch.float32,
            ),
            2,
        ),
        (
            torch.tensor(
                [
                    [[1, 3], [1, 4], [2, 4]],
                    [[5, 5], [6, 5], [6, 7]],
                ],
                dtype=torch.float32,
            ),
            1,
        ),
        (
            torch.tensor(
                [
                    [[1, 8], [3, 4]],
                    [[1, 9], [3, 4]],
                    [[2, 9], [5, 4]],
                ],
                dtype=torch.float32,
            ),
            0,
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
def test_mode_matches_torch_on_repeated_values(data, dim, dtype, rtol, atol):
    input = data.to(device="cuda", dtype=dtype)

    values, indices = ntops.torch.mode(input, dim=dim)
    expected = torch.mode(input, dim=dim)

    assert values.shape == expected.values.shape
    assert indices.shape == expected.indices.shape
    assert values.dtype == expected.values.dtype
    assert indices.dtype == expected.indices.dtype
    assert torch.allclose(values, expected.values, rtol=rtol, atol=atol)
    assert torch.equal(indices, expected.indices)


@skip_if_cuda_not_available
def test_mode_rejects_empty_reduction_dim():
    input = torch.empty((2, 0), device="cuda")

    with pytest.raises(IndexError, match="non-zero size"):
        ntops.torch.mode(input, dim=1)


@skip_if_cuda_not_available
def test_mode_rejects_keepdim_for_now():
    input = torch.ones((2, 3), device="cuda")

    with pytest.raises(NotImplementedError, match="keepdim"):
        ntops.torch.mode(input, dim=1, keepdim=True)
