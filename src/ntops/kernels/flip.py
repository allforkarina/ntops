import functools

from ninetoothed import Tensor


def arrangement(input, output, dim, block_size=None):
    if block_size is None:
        block_size = 1

    non_flip_dims = tuple(i for i in range(input.ndim) if i != dim)
    perm = non_flip_dims + (dim,)

    input = input.permute(perm)
    output = output.permute(perm)

    tile_shape = tuple(1 for _ in non_flip_dims) + (-1,)
    input = input.tile(tile_shape)
    output = output.tile(tile_shape)

    if non_flip_dims:
        squeeze_dims = tuple(range(len(non_flip_dims)))
        input.dtype = input.dtype.squeeze(squeeze_dims)
        output.dtype = output.dtype.squeeze(squeeze_dims)

    return input, output


def application(input, output):
    dim_size = input.shape[0]
    for i in range(dim_size):
        output[i] = input[dim_size - 1 - i]


def premake(ndim, dim, dim_size, dtype=None, block_size=None):
    if block_size is None:
        block_size = 1

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    input = Tensor(ndim, dtype=dtype)
    output = Tensor(ndim, dtype=dtype)
    input.shape = input.shape[:dim] + (dim_size,) + input.shape[dim + 1:]
    output.shape = output.shape[:dim] + (dim_size,) + output.shape[dim + 1:]

    tensors = (input, output)

    return arrangement_, application, tensors
