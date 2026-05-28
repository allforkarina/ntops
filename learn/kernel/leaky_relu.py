import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, alpha, output):
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x

    这一行被 ninetoothed 编译成数千个 GPU 线程并行执行——这才是真正的 kernel！
    """
    output = ntl.where(input > 0, input, alpha * input)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),  # alpha 是标量
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors