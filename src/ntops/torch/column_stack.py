from typing import Sequence

import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(1)
    return tensor


def _promoted_dtype(tensors: Sequence[torch.Tensor]) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _validate_and_normalize(
    tensors: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], torch.dtype]:
    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("column_stack expects a non-empty TensorList")

    device = tensors[0].device
    for tensor in tensors:
        if tensor.device != device:
            raise RuntimeError("column_stack expects all tensors to be on the same device")

    normalized = tuple(_normalize_tensor(tensor) for tensor in tensors)
    ndim = normalized[0].ndim
    common_shape = normalized[0].shape

    for index, tensor in enumerate(normalized[1:], start=1):
        if tensor.ndim != ndim:
            raise RuntimeError(
                "Tensors must have same number of dimensions: "
                f"got {ndim} and {tensor.ndim}"
            )
        for dim, (expected, actual) in enumerate(zip(common_shape, tensor.shape)):
            if dim != 1 and expected != actual:
                raise RuntimeError(
                    "Sizes of tensors must match except in dimension 1. "
                    f"Expected size {expected} but got size {actual} "
                    f"for tensor number {index} in the list."
                )

    return normalized, _promoted_dtype(tensors)


def _output_shape(tensors: Sequence[torch.Tensor]) -> tuple[int, ...]:
    shape = list(tensors[0].shape)
    shape[1] = sum(tensor.shape[1] for tensor in tensors)
    return tuple(shape)


def _column_slice(output: torch.Tensor, offset: int, width: int) -> torch.Tensor:
    index = (
        slice(None),
        slice(offset, offset + width),
        *((slice(None),) * (output.ndim - 2)),
    )
    return output[index]


def column_stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    r"""Stack tensors as columns by writing each input into its output column range."""
    normalized, dtype = _validate_and_normalize(tensors)
    output = torch.empty(
        _output_shape(normalized),
        dtype=dtype,
        device=normalized[0].device,
    )

    offset = 0
    for tensor in normalized:
        width = tensor.shape[1]
        destination = _column_slice(output, offset, width)
        kernel = _cached_make(
            ntops.kernels.column_stack.premake,
            tensor.ndim,
            input_dtype=tensor.dtype,
            output_dtype=output.dtype,
        )
        kernel(tensor, destination)
        offset += width

    return output
