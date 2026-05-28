from typing import Sequence

import torch

import ntops
from ntops.torch.utils import _cached_make


def column_stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    r"""Stack 1D tensors as columns into a 2D tensor.

    Creates a new tensor by horizontally stacking tensors:

        output[i][j] = tensors[j][i]

    Args:
        tensors: a sequence of 1D tensors of the same length.

    Returns:
        a 2D tensor of shape ``(L, N)``, where *L* is the length of each
        input and *N* is the number of inputs.
    """
    # Step 1: PyTorch 完成列堆叠重排
    src = torch.stack(tuple(tensors), dim=1)

    # Step 2: GPU kernel 拷贝输出
    out = torch.empty_like(src)
    kernel = _cached_make(ntops.kernels.column_stack.premake, src.ndim)
    kernel(src, out)

    return out