import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input_1d, output_nd, target_axis, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = output_nd.ndim
    perm = (target_axis,) + tuple(i for i in range(ndim) if i != target_axis)

    output_permuted = output_nd.permute(perm)
    output_flat = output_permuted.flatten(1)
    output_tiled = output_flat.tile((-1, block_size))

    return input_1d, output_tiled


def application(input_1d, output_2d):
    for j in range(output_2d.shape[0]):
        for k in range(output_2d.shape[1]):
            output_2d[j, k] = input_1d[j]


def premake(ndim, target_axis, dim_size, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(
        arrangement, target_axis=target_axis, block_size=block_size
    )

    input = Tensor(1, dtype=dtype)
    output = Tensor(ndim, dtype=dtype)
    output.shape = (
        output.shape[:target_axis]
        + (dim_size,)
        + output.shape[target_axis + 1:]
    )

    tensors = (input, output)

    return arrangement_, application, tensors
