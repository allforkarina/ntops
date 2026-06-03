import pathlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_feature_alpha_dropout_torch_layer_not_fake():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "torch" / "feature_alpha_dropout.py"
    ).read_text(encoding="utf-8")

    assert "torch.feature_alpha_dropout" not in source
    assert "torch.nn.functional.feature_alpha_dropout" not in source


def test_feature_alpha_dropout_kernel_has_lcg():
    source = (
        _PROJECT_ROOT / "src" / "ntops" / "kernels" / "feature_alpha_dropout.py"
    ).read_text(encoding="utf-8")

    assert "1103515245" in source
    assert "2147483647" in source
    assert "rand_val" in source


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 5), (3, 4, 2, 3)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_training_false_returns_input(shape, dtype):
    input = torch.randn(shape, device="cuda", dtype=dtype)

    output = ntops.torch.feature_alpha_dropout(input, p=0.5, training=False)

    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert torch.equal(output, input)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 5), (3, 4, 2, 3)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_p_zero_returns_input(shape, dtype):
    input = torch.randn(shape, device="cuda", dtype=dtype)

    output = ntops.torch.feature_alpha_dropout(input, p=0.0, training=True)

    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert torch.equal(output, input)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape, p",
    [
        ((2, 3), 0.5),
        ((4, 3, 5), 0.3),
        ((2, 4, 3, 2), 0.7),
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
def test_training_true_has_valid_output(shape, p, dtype, rtol, atol):
    input = torch.ones(shape, device="cuda", dtype=dtype)
    a = (1 - p + p * (-1.7580993408473766) ** 2) ** -0.5
    b = -a * (-1.7580993408473766) * p

    output = ntops.torch.feature_alpha_dropout(input, p=p, training=True)

    assert output.shape == input.shape
    assert output.dtype == input.dtype

    expected_masked = -1.7580993408473766 * a + b
    expected_unmasked = 1.0 * a + b
    expected = {
        torch.tensor(expected_masked, dtype=torch.float64),
        torch.tensor(expected_unmasked, dtype=torch.float64),
    }

    flat = output.flatten().to(torch.float64)
    for val in flat[:50]:
        matches = [
            torch.isclose(val, exp, rtol=float(rtol), atol=float(atol))
            for exp in expected
        ]
        assert any(matches), f"value {val} not in expected set {expected}"

    for n in range(output.shape[0]):
        for c in range(output.shape[1]):
            feature_slice = output[n, c].flatten()
            first = feature_slice[0]
            assert torch.allclose(
                feature_slice, first.expand_as(feature_slice),
                rtol=float(rtol), atol=float(atol),
            )


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape",
    [(3, 4), (2, 3, 5), (4, 2, 3, 2)],
)
def test_training_true_non_contiguous(shape):
    """Non-contiguous inputs must preserve feature-level masking
    (same keep/drop decision for all elements in one channel)."""
    p = 0.3
    full = torch.ones(shape, device="cuda", dtype=torch.float32)
    a = (1 - p + p * (-1.7580993408473766) ** 2) ** -0.5
    b = -a * (-1.7580993408473766) * p

    if full.ndim >= 3:
        input = full.permute(0, 2, 1) if full.ndim == 3 else full.permute(0, 1, 3, 2)
    else:
        input = full.t()

    output = ntops.torch.feature_alpha_dropout(input, p=p, training=True)

    assert output.shape == input.shape
    assert output.dtype == input.dtype

    expected_masked = -1.7580993408473766 * a + b
    expected_unmasked = 1.0 * a + b
    expected = {
        torch.tensor(expected_masked, dtype=torch.float64),
        torch.tensor(expected_unmasked, dtype=torch.float64),
    }

    flat = output.flatten().to(torch.float64)
    for val in flat[:50]:
        matches = [
            torch.isclose(val, exp, rtol=1e-4, atol=1e-4)
            for exp in expected
        ]
        assert any(matches)

    # Feature-level semantics: all spatial elements in one (N,C) slice
    # share the same random decision, so must be identical.
    for n in range(output.shape[0]):
        for c in range(output.shape[1]):
            feature_slice = output[n, c].flatten()
            first = feature_slice[0]
            assert torch.allclose(feature_slice, first.expand_as(feature_slice))


@skip_if_cuda_not_available
def test_feature_alpha_dropout_rejects_bad_p():
    input = torch.randn(2, 3, device="cuda")
    with pytest.raises(ValueError, match="p must be"):
        ntops.torch.feature_alpha_dropout(input, p=1.5, training=True)
    with pytest.raises(ValueError, match="p must be"):
        ntops.torch.feature_alpha_dropout(input, p=-0.1, training=True)


@skip_if_cuda_not_available
def test_feature_alpha_dropout_rejects_1d():
    input = torch.randn(5, device="cuda")
    with pytest.raises(RuntimeError, match="at least 2D"):
        ntops.torch.feature_alpha_dropout(input, p=0.5, training=True)
