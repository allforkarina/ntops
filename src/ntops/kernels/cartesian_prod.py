import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input_1d, output_col, stride, input_size, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input_1d
    output_arranged = output_col.flatten().tile((block_size,))

    return input_arranged, output_arranged, stride, input_size


def application(input_1d, output_1d, stride, input_size):
    for k in range(output_1d.shape[0]):
        src = (k // stride) % input_size
        output_1d[k] = input_1d[src]


def premake(input_size, num_rows, stride, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    input = Tensor(1, dtype=dtype)
    input.shape = (input_size,)

    output_col = Tensor(2, dtype=dtype)
    output_col.shape = (num_rows, 1)

    tensors = (
        input,
        output_col,
        Tensor(0, constexpr=True, value=stride),
        Tensor(0, constexpr=True, value=input_size),
    )

    return arrangement_, application, tensors
