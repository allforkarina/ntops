"""Roll (cyclic shift) kernel."""

import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.roll_arrangement import arrangement


def application(input, output, shift):
    dim_size = input.shape[1]

    for row in range(input.shape[0]):
        for i in range(dim_size):
            src = (i + dim_size - shift) % dim_size
            output[row][i] = input[row][src]


def premake(ndim, dim, shift, dim_size, dtype=None, block_size=None):
    if block_size is None:
        block_size = 1

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    input = Tensor(ndim, dtype=dtype)
    output = Tensor(ndim, dtype=dtype)

    for tensor in (input, output):
        tensor.shape = tensor.shape[:dim] + (dim_size,) + tensor.shape[dim + 1 :]

    tensors = (
        input,
        output,
        Tensor(0, constexpr=True, value=shift),
    )

    return arrangement_, application, tensors
