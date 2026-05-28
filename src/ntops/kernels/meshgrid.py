import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(src, dst):
    dst = src  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors