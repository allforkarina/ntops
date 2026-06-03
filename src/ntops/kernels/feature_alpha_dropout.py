import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


_MOD = 2147483647
_MAGIC = 1103515245
_OFFSET = 12345


def arrangement(input, output, seed, p, alpha, a, b, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    if input.ndim > 3:
        input = input.flatten(start_dim=2)
        output = output.flatten(start_dim=2)

    input = input.flatten(end_dim=1)
    output = output.flatten(end_dim=1)

    input = input.tile((-1, block_size))
    output = output.tile((-1, block_size))

    return input, output, seed, p, alpha, a, b


def application(input, output, seed, p, alpha, a, b):
    out_dtype = input.dtype

    p_f32 = ntl.cast(p, ntl.float32)

    for fid in range(output.shape[0]):
        mixed = (seed + fid * _MAGIC + _OFFSET) % _MOD
        rand_val = ntl.cast(mixed, ntl.float32) / ntl.cast(_MOD, ntl.float32)
        keep = rand_val > p_f32

        for k in range(output.shape[1]):
            val = input[fid, k]
            out = ntl.where(keep, val, ntl.cast(alpha, out_dtype))
            out = out * ntl.cast(a, out_dtype) + ntl.cast(b, out_dtype)
            output[fid, k] = out


def premake(ndim, N, C, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    input = Tensor(ndim, dtype=dtype)
    output = Tensor(ndim, dtype=dtype)
    input.shape = (N,) + (C,) + input.shape[2:]
    output.shape = (N,) + (C,) + output.shape[2:]

    tensors = (
        input,
        output,
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.float64),
    )

    return arrangement_, application, tensors
