import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, seed, p, alpha, a, b, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.flatten(start_dim=2)
    output = output.flatten(start_dim=2)

    input = input.tile((-1, -1, block_size))
    output = output.tile((-1, -1, block_size))

    return input, output, seed, p, alpha, a, b


def application(input, output, seed, p, alpha, a, b):
    _MOD = 2147483647
    _MAGIC = 1103515245
    _OFFSET = 12345

    out_dtype = input.dtype
    p_f32 = ntl.cast(p, ntl.float32)
    mod_f32 = ntl.cast(_MOD, ntl.float32)

    for n in range(output.shape[0]):
        for c in range(output.shape[1]):
            fid = n * output.shape[1] + c

            mixed = (seed + fid * _MAGIC + _OFFSET) % _MOD
            rand_val = ntl.cast(mixed, ntl.float32) / mod_f32
            keep = rand_val > p_f32

            for k in range(output.shape[2]):
                val = input[n, c, k]
                out = ntl.where(keep, val, ntl.cast(alpha, out_dtype))
                out = out * ntl.cast(a, out_dtype) + ntl.cast(b, out_dtype)
                output[n, c, k] = out


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
