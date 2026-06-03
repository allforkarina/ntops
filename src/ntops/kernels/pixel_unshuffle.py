import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input, output, r, downscale_factor, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.tile((-1, -1, -1, block_size * downscale_factor))
    output = output.tile((-1, -1, -1, block_size))

    return input, output, r


def application(input, output, r):
    rf = r
    r_sq = rf * rf

    for n in range(output.shape[0]):
        for c_out in range(output.shape[1]):
            c = c_out // r_sq
            ij = c_out % r_sq
            i = ij // rf
            j = ij % rf

            for h in range(output.shape[2]):
                for w in range(output.shape[3]):
                    output[n, c_out, h, w] = (
                        input[n, c, h * rf + i, w * rf + j]
                    )


def premake(N, C, H, W, downscale_factor, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    r = downscale_factor
    C_out = C * r * r
    H_out = H // r
    W_out = W // r

    arrangement_ = functools.partial(
        arrangement,
        downscale_factor=downscale_factor,
        block_size=block_size,
    )

    input = Tensor(4, dtype=dtype)
    output = Tensor(4, dtype=dtype)
    input.shape = (N, C, H, W)
    output.shape = (N, C_out, H_out, W_out)

    tensors = (
        input,
        output,
        Tensor(0, dtype=ninetoothed.int64, constexpr=True, value=r),
    )

    return arrangement_, application, tensors
