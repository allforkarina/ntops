import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.mode_arrangement import arrangement


def application(input, values, indices):
    best_value = input[0]
    best_index = ntl.cast(0, ntl.int64)
    best_count = ntl.cast(0, ntl.int64)

    for i in range(input.shape[0]):
        candidate = input[i]
        candidate_count = ntl.cast(0, ntl.int64)

        for j in range(input.shape[0]):
            matched = ntl.where(input[j] == candidate, 1, 0)
            candidate_count += ntl.cast(matched, ntl.int64)

        has_more_count = candidate_count > best_count
        has_same_count = candidate_count == best_count
        has_better_value = candidate <= best_value
        should_update = has_more_count | (has_same_count & has_better_value)

        best_value = ntl.where(should_update, candidate, best_value)
        best_index = ntl.where(should_update, ntl.cast(i, ntl.int64), best_index)
        best_count = ntl.where(should_update, candidate_count, best_count)

    values = best_value  # noqa: F841
    indices = best_index  # noqa: F841


def premake(ndim, dim, dtype=None, block_size=None):
    if block_size is None:
        block_size = -1

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim - 1, dtype=dtype),
        Tensor(ndim - 1, dtype=ninetoothed.int64),
    )

    return arrangement_, application, tensors
