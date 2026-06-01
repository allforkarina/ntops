import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(src_values, src_indices, dst_values, dst_indices):
    for i in range(src_values.shape[0]):
        dst_values[i] = src_values[i]
        dst_indices[i] = src_indices[i]


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.int64),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.int64),
    )

    return arrangement_, application, tensors