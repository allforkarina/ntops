"""Roll (cyclic shift) kernel.

This is an experimental real-kernel implementation used to verify whether
ninetoothed application code supports computed index expressions such as
``input[row, dim_size - shift + i]`` and ``output[row, shift + i]``.
"""

import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.roll_arrangement import arrangement


def application(input, output, shift):
    dim_size = input.shape[1]

    for row in range(input.shape[0]):
        for i in range(shift):
            output[row, i] = input[row, dim_size - shift + i]

        for i in range(dim_size - shift):
            output[row, shift + i] = input[row, i]


def premake(ndim, dim, shift, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=shift),
    )

    return arrangement_, application, tensors
