import torch

import ntops
from ntops.torch.utils import _cached_make


def _mode_from_sorted(sorted_vals, sorted_idx, dim):
    """全程 GPU 向量化：从已排序张量中提取众数。

    Args:
        sorted_vals: 已沿 *dim* 排序的值张量。
        sorted_idx:  对应的原始索引张量。
        dim:         归约维度（已归一化为非负数）。

    Returns:
        ``(values, indices)`` — 众数值和对应的原始索引。
    """
    N = sorted_vals.shape[dim]

    # ── 1. 标记连续段起始位置（向量化差分） ────────────────────
    is_start = torch.ones_like(sorted_vals, dtype=torch.bool)
    if N > 1:
        slices_curr = [slice(None)] * sorted_vals.ndim
        slices_prev = [slice(None)] * sorted_vals.ndim
        slices_curr[dim] = slice(1, None)
        slices_prev[dim] = slice(None, -1)
        is_start[tuple(slices_curr)] = (
            sorted_vals[tuple(slices_curr)] != sorted_vals[tuple(slices_prev)]
        )

    # ── 2. 段 ID + 把 dim 移到最后 ────────────────────────────
    run_ids = is_start.long().cumsum(dim=dim)

    sorted_vals = sorted_vals.movedim(dim, -1)
    sorted_idx = sorted_idx.movedim(dim, -1)
    is_start = is_start.movedim(dim, -1)
    run_ids = run_ids.movedim(dim, -1)

    prefix_shape = sorted_vals.shape[:-1]
    num_slices = 1
    for s in prefix_shape:
        num_slices *= s

    run_ids_2d = run_ids.reshape(num_slices, N)
    is_start_2d = is_start.reshape(num_slices, N)
    sv_2d = sorted_vals.reshape(num_slices, N)
    si_2d = sorted_idx.reshape(num_slices, N)

    # ── 3. 给每个切面的段 ID 加偏移，使其全局唯一 ──────────────
    max_runs_per_slice = run_ids_2d.amax(dim=1)  # [num_slices]
    stride = int(max_runs_per_slice.amax().item()) + 1

    offsets = torch.arange(num_slices, device=sorted_vals.device) * stride
    unique_ids = run_ids_2d + offsets.unsqueeze(1)  # [num_slices, N]

    # ── 4. scatter_add 统计每段长度 ────────────────────────────
    total_bins = num_slices * stride
    counts = torch.zeros(total_bins, dtype=torch.long, device=sorted_vals.device)
    counts.scatter_add_(
        0, unique_ids.reshape(-1),
        torch.ones(unique_ids.numel(), dtype=torch.long, device=sorted_vals.device),
    )
    counts_2d = counts.reshape(num_slices, stride)  # [num_slices, stride]

    # ── 5. 找出每切面最长的段 ──────────────────────────────────
    best_run_local = counts_2d.argmax(dim=1)  # [num_slices]
    best_run_global = best_run_local + offsets  # [num_slices]

    # 每切面中 best_run_global 首次出现的位置
    mask = (unique_ids == best_run_global.unsqueeze(1))  # [num_slices, N]
    pos = mask.long().argmax(dim=1)  # [num_slices]  ← 首个 True 的位置

    # ── 6. 取出对应值 ─────────────────────────────────────────
    result_vals = sv_2d[torch.arange(num_slices), pos]
    result_idx = si_2d[torch.arange(num_slices), pos]

    return result_vals.reshape(prefix_shape), result_idx.reshape(prefix_shape)


def mode(input: torch.Tensor, dim: int = -1):
    r"""Return the mode (most frequent value) along the given dimension.

    Computed by sorting along *dim* and finding the longest run of
    consecutive equal values — entirely on GPU without host-side loops.

    Args:
        input: the input tensor.
        dim:   the dimension to reduce.

    Returns:
        ``(values, indices)`` — *values* is the most frequent element in
        each slice along *dim*, and *indices* is the index of that element
        in the original (unsorted) tensor.  Ties are broken in favour of
        the **smallest** value.
    """
    dim = dim if dim >= 0 else input.ndim + dim

    # ── Step 1: 排序 ──────────────────────────────────────────
    sorted_vals, sorted_idx = torch.sort(input, dim=dim)

    # ── Step 2: 向量化找众数（全程 GPU） ───────────────────────
    result_vals, result_idx = _mode_from_sorted(sorted_vals, sorted_idx, dim)

    # ── Step 3: kernel 拷贝 ───────────────────────────────────
    out_vals = torch.empty_like(result_vals)
    out_idx = torch.empty_like(result_idx)

    kernel = _cached_make(ntops.kernels.mode.premake, result_vals.ndim)
    kernel(result_vals, result_idx, out_vals, out_idx)

    return out_vals, out_idx