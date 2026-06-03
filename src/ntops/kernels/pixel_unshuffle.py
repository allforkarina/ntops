import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input, output, r, downscale_factor, block_h=None, block_w=None):
    if block_h is None:
        block_h = 4
    if block_w is None:
        block_w = 8

    input = input.tile(
        (-1, -1, block_h * downscale_factor, block_w * downscale_factor)
    )
    output = output.tile((-1, -1, block_h, block_w))

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


def premake(N, C, H, W, downscale_factor, dtype=None, block_h=None, block_w=None):
    r = downscale_factor
    C_out = C * r * r
    H_out = H // r
    W_out = W // r

    if block_h is None:
        block_h = min(H_out, 4)
    if block_w is None:
        block_w = min(W_out, 8)

    arrangement_ = functools.partial(
        arrangement,
        downscale_factor=downscale_factor,
        block_h=block_h,
        block_w=block_w,
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
