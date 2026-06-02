import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(src, dst, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    src_arranged = src.flatten().tile((block_size,))
    dst_arranged = dst.flatten().tile((block_size,))

    return src_arranged, dst_arranged


def application(src, dst):
    for i in range(src.shape[0]):
        dst[i] = src[i]


def premake(ndim, input_dtype=None, output_dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=input_dtype),
        Tensor(ndim, dtype=output_dtype),
    )

    return arrangement_, application, tensors
