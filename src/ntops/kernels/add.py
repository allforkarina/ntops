import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, alpha, output):
    for i in range(output.shape[0]):
        src = (i + 1) % output.shape[0]
        output[i] = input[i] + alpha * other[i] + (other[src] - other[src])


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
