"""
Roll (cyclic shift) kernel.

The cyclic shift itself is performed in the torch layer (``torch/roll.py``)
via ``torch.cat`` + ``torch.narrow``, which produces a contiguous rolled tensor.
This kernel performs the final GPU-side copy into the output buffer.

Once ninetoothed supports computed index expressions, the roll can be moved
into the application::

    def application(input, output, dim_size, shift):
        for j in range(input.shape[0]):
            # copy tail → output head
            for i in range(shift):
                output[j, i] = input[j, dim_size - shift + i]
            # copy head → output tail
            for i in range(dim_size - shift):
                output[j, shift + i] = input[j, i]

This would require ``input[j, expr]`` and ``output[j, expr]`` where *expr*
is a loop-variable-plus-constant expression.  When that is available,
switch the premake to use ``roll_arrangement.arrangement`` instead of
``element_wise.arrangement``.
"""

import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(src, dst):
    for i in range(src.shape[0]):
        dst[i] = src[i]


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors