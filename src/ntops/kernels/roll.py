"""Roll (cyclic shift) kernel."""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.roll_arrangement import arrangement


def application(input, output, shift, dim):
    dim_size = input.source.shape[dim]
    dim_stride = input.source.strides[dim]
    output_dim_offset = output.offsets(dim)
    input_dim_offset = (output_dim_offset + dim_size - shift) % dim_size

    input_offsets = (
        output.offsets() + (input_dim_offset - output_dim_offset) * dim_stride
    )

    mask = output.offsets(-1) < output.source.shape[-1]
    output = ntl.load(  # noqa: F841
        input.source.data_ptr + input_offsets,
        mask=mask,
        other=0.0,
    )


def premake(ndim, dim, shift, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=shift),
        Tensor(0, constexpr=True, value=dim),
    )

    return arrangement_, application, tensors
