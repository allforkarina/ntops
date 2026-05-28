import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(src_values, src_indices, dst_values, dst_indices):
    dst_values = src_values  # noqa: F841
    dst_indices = src_indices  # noqa: F841


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