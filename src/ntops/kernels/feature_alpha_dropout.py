import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, mask, alpha, a, b, output):
    """逐元素：被 mask 丢弃的位置用 alpha 填充，再做仿射变换 a*x + b。

    这是真正的 GPU 数值计算——每行都被编译成 CUDA 机器码在数千个线程上并行执行。
    """
    output = ntl.where(mask, input, alpha) * a + b  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.int8),     # mask (0/1)
        Tensor(0, dtype=ninetoothed.float64),     # alpha (标量)
        Tensor(0, dtype=ninetoothed.float64),     # a (标量)
        Tensor(0, dtype=ninetoothed.float64),     # b (标量)
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors