"""Roll (cyclic shift) kernel."""

import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.roll_arrangement import arrangement


def application(input, output, shift):
    dim_size = input.shape[1]

    for row in range(input.shape[0]):
        index = (input[row].offsets(-1) + dim_size - shift) % dim_size
        output[row] = input[row][index]


def premake(ndim, dim, shift, dtype=None, block_size=None):
    if block_size is None:
        block_size = 1

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=shift),
    )

    return arrangement_, application, tensors
