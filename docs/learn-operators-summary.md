# learn/ 算子实现总览

> 基于 `learn/kernel/` 和 `learn/torch/` 中的 11 个算子，逐一分析 kernel 层和 torch 接口层的实现细节、设计意图和分工边界。

---

## 一、分类总览

### Real Kernel（kernel 中有实际 GPU 数值计算）

| 算子 | kernel 中的计算 | torch 层核心工作 |
|------|----------------|-----------------|
| `leaky_relu` | `ntl.where(x > 0, x, alpha * x)` | 分配输出、调用 kernel |
| `feature_alpha_dropout` | `ntl.where(mask, input, alpha) * a + b` | Bernoulli mask 生成、affine 系数计算 |
| `mse_loss` | `(pred - target)^2` | reduction 分支（mean/sum/none） |

### Copy Kernel（kernel 只做逐元素拷贝 `dst = src`）

| 算子 | torch 层核心工作 | torch 层使用的 PyTorch API |
|------|-----------------|--------------------------|
| `roll` | narrow + cat 重排 | `torch.cat`, `torch.narrow` |
| `column_stack` | 列堆叠 | `torch.stack` |
| `mode` | sort + 向量化扫描（全程 GPU） | `torch.sort`, `scatter_add_`, `argmax` |
| `flip` | 翻转 | `torch.flip` |
| `fliplr` | **委托给 flip** | 一行 `ntops.torch.flip(input, dims=(-1,))` |
| `meshgrid` | 生成坐标网格 | `torch.meshgrid` |
| `cartesian_prod` | 笛卡尔积 | `torch.cartesian_prod` |
| `pixel_unshuffle` | 空间到深度转换 | `F.pixel_unshuffle` |

---

## 二、leaky_relu

### Kernel 层 (`learn/kernel/leaky_relu.py`)

```python
def application(input, alpha, output):
    output = ntl.where(input > 0, input, alpha * input)
```

这是一个 **Real Kernel**——`ntl.where` 和乘法被 nintoothed 编译为 GPU 机器码，在数千个线程上并行执行。每个线程处理输入张量中的一个元素，判断其正负，然后计算输出值。

**arrangement**: 使用 `element_wise.arrangement`，因为 Leaky ReLU 是逐元素操作，每个元素的输出只依赖于同一个位置的输入，不需要跨元素通信。

**premake 张量声明**: 3 个张量——`input`（ndim 维浮点）、`alpha`（0 维标量，float64）、`output`（ndim 维浮点）。alpha 用 `Tensor(0, dtype=ninetoothed.float64)` 声明，因为它是单个标量值，不是数组。

### Torch 接口层 (`learn/torch/leaky_relu.py`)

```python
def leaky_relu(input, negative_slope=0.01):
    out = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.leaky_relu.premake, input.ndim)
    kernel(input, negative_slope, out)
    return out
```

五步法中最简短的实现。没有参数规范化或数据预处理——`negative_slope` 直接作为标量传入 kernel。`_cached_make` 的第二个参数 `input.ndim` 传给 premake，确保 kernel 编译期知道张量的维数。kernel 调用时传 3 个参数，与 premake 中声明的 3 个 Tensor 一一对应。

**设计意图**: 这是一个纯计算算子，torch 层只负责内存分配和 kernel 调度，所有数学逻辑都在 kernel 中。典型的"重 kernel、轻 torch"模式。

---

## 三、feature_alpha_dropout

### Kernel 层 (`learn/kernel/feature_alpha_dropout.py`)

```python
def application(input, mask, alpha, a, b, output):
    output = ntl.where(mask, input, alpha) * a + b
```

**Real Kernel**。`ntl.where(mask, input, alpha)` 根据 mask 选择保留原值或替换为 alpha，然后做仿射变换 `a * x + b`。mask 是 int8 类型（0 或 1），alpha、a、b 是标量。

**premake 张量声明**: 6 个张量——`input`（ndim）、`mask`（ndim, int8）、`alpha`（标量, float64）、`a`（标量, float64）、`b`（标量, float64）、`output`（ndim）。这是当前 learn/ 中参数最多的 kernel。

### Torch 接口层 (`learn/torch/feature_alpha_dropout.py`)

```python
def feature_alpha_dropout(input, p=0.5, training=True):
    if not training or p == 0:
        return input
    # 生成 per-channel Bernoulli mask
    N, C = input.shape[0], input.shape[1]
    mask = torch.bernoulli(
        torch.full((N, C, 1, 1), 1 - p, device=input.device)
    ).expand_as(input)
    # 计算保持均值/方差的 affine 系数
    alpha = -1.7580993408473766
    a = (1 - p + p * alpha**2) ** -0.5
    b = -a * alpha * p
    out = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.feature_alpha_dropout.premake, input.ndim)
    kernel(input, mask, alpha, a, b, out)
    return out
```

torch 层在这里做了大量数学预处理：生成 `(N, C, 1, 1)` 形状的 Bernoulli mask 并 broadcast 到 input 形状；计算 affine 系数 a 和 b 以保持输出的均值和方差不变。kernel 只负责"应用"这些已计算好的参数。

**设计意图**: 参数计算（mask 生成、affine 推导）放在 torch 层，逐元素的条件选择放在 kernel 层。分工清晰——torch 做"准备"，kernel 做"执行"。

---

## 四、mse_loss

### Kernel 层 (`learn/kernel/mse_loss.py`)

```python
def application(pred, target, output):
    diff = pred - target
    output = diff * diff
```

**Real Kernel**。逐元素计算 `(pred - target)^2`，不做 reduction（求和/平均）。减法和乘法被编译为 GPU 机器码。

**premake 张量声明**: 3 个张量——`pred`（ndim）、`target`（ndim）、`output`（ndim），全部同维数同类型。

### Torch 接口层 (`learn/torch/mse_loss.py`)

```python
def mse_loss(pred, target, reduction="mean"):
    squared_error = torch.empty_like(pred)
    kernel = _cached_make(ntops.kernels.mse_loss.premake, pred.ndim)
    kernel(pred, target, squared_error)
    if reduction == "none":
        return squared_error
    if reduction == "sum":
        return squared_error.sum()
    return squared_error.mean()
```

kernel 只输出逐元素平方误差，reduction（`mean` / `sum` / `none`）在 torch 层用 `torch.Tensor.sum()` 和 `torch.Tensor.mean()` 完成。

**设计意图**: 这是一个混合模式——kernel 做元素级计算（平方误差），torch 做跨维度规约。因为规约操作（sum/mean）难以用 ninetoothed 的 tile 模型表达，所以留在 torch 层。与其他 copy kernel 不同，mse_loss 的 kernel 有真正的数值运算。

---

## 五、roll

### Kernel 层 (`learn/kernel/roll.py`)

```python
def application(src, dst):
    dst = src
```

**Copy Kernel**。2 张量逐元素拷贝，不包含任何计算逻辑。arrangement 使用 `element_wise.arrangement`。

### Torch 接口层 (`learn/torch/roll.py`)

```python
def roll(input, shifts, dims=0):
    N = input.shape[dims]
    shift = shifts % N
    src = torch.cat([
        input.narrow(dims, N - shift, shift),
        input.narrow(dims, 0, N - shift),
    ], dim=dims)
    out = torch.empty_like(src)
    kernel = _cached_make(ntops.kernels.roll.premake, src.ndim)
    kernel(src, out)
    return out
```

torch 层完成了 roll 的全部复杂逻辑：用 `narrow` 把张量沿指定维度切成两段（尾部 [N-shift:] 和 头部 [:N-shift]），再用 `cat` 拼接成"滚动后"的结果。kernel 只是把这个结果搬到 GPU 输出缓冲区。

**设计意图**: roll 的本质是**内存重排**（切片 + 拼接），ninetoothed 的 DSL 没有 slice/concat 原语，因此重排逻辑只能在 torch 层完成。kernel 做 copy 是务实的工程选择。

**五步法对照**:
1. 参数规范化 — `shift = shifts % N`
2. 预处理 — narrow + cat
3. `_cached_make` — 传 `src.ndim`
4. kernel 调用 — `kernel(src, out)`
5. return — `out`

---

## 六、column_stack

### Kernel 层 (`learn/kernel/column_stack.py`)

和 roll 完全相同的 copy kernel 模式：`dst = src`，2 张量。

### Torch 接口层 (`learn/torch/column_stack.py`)

```python
def column_stack(tensors):
    src = torch.stack(tuple(tensors), dim=1)
    out = torch.empty_like(src)
    kernel = _cached_make(ntops.kernels.column_stack.premake, src.ndim)
    kernel(src, out)
    return out
```

一行 `torch.stack(tensors, dim=1)` 完成列堆叠（把 N 个 1D 张量沿 dim=1 堆成 2D 张量），kernel 做拷贝。类型签名接受 `Sequence[torch.Tensor]`。

**设计意图**: 同样是内存重排算子。torch 层做重排，kernel 做搬运。column_stack 比 roll 更简单——只有一次 stack 调用。

---

## 七、mode

### Kernel 层 (`learn/kernel/mode.py`)

```python
def application(src_values, src_indices, dst_values, dst_indices):
    dst_values = src_values
    dst_indices = src_indices
```

**Copy Kernel，但处理 4 张量**。与其他 copy kernel 不同，mode 需要同时拷贝 values 和 indices 两个结果张量，因此 premake 声明了 4 个 Tensor——两个 `ndim` 浮点（values）和两个 `ninetoothed.int64`（indices）。

### Torch 接口层 (`learn/torch/mode.py`)

```python
def mode(input, dim=-1):
    dim = dim if dim >= 0 else input.ndim + dim
    sorted_vals, sorted_idx = torch.sort(input, dim=dim)
    result_vals, result_idx = _mode_from_sorted(sorted_vals, sorted_idx, dim)
    out_vals = torch.empty_like(result_vals)
    out_idx = torch.empty_like(result_idx)
    kernel = _cached_make(ntops.kernels.mode.premake, result_vals.ndim)
    kernel(result_vals, result_idx, out_vals, out_idx)
    return out_vals, out_idx
```

torch 层是 learn/ 中逻辑最复杂的接口。核心在 `_mode_from_sorted()` 辅助函数中：

1. **向量化差分** — 用切片语法标记连续段边界，全程 GPU
2. **全局唯一段 ID** — 每个切面的 `run_ids` 加 `slice_index * stride` 偏移，使不同切面的段互不冲突
3. **一次 `scatter_add_`** — 所有切面所有段一次性统计
4. **`argmax` 找位置** — 用 `mask.long().argmax(dim=1)` 找到每切面最长段的首次出现位置

这是经过 code review 优化后的版本，完全消除了 CPU for 循环。原始版本对每个切面做 Python 串行迭代，每次索引都会触发 GPU → CPU 数据传输，优化前性能极差。

**设计意图**: mode 是"规约操作"的代表——统计频次 + 找最大值。这类操作难以用 ninetoothed 的 tile 模型表达（tile 之间需要通信），因此复杂的扫描/规约逻辑放在 torch 层，kernel 只做结果拷贝。

---

## 八、flip

### Kernel 层 (`learn/kernel/flip.py`)

标准的 copy kernel：`dst = src`，2 张量。

### Torch 接口层 (`learn/torch/flip.py`)

```python
def flip(input, dims):
    result = torch.flip(input, dims)
    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.flip.premake, result.ndim)
    kernel(result, out)
    return out
```

torch 层一行 `torch.flip(input, dims)` 完成翻转，kernel 拷贝。接受 `Union[int, Sequence[int]]` 类型的 `dims` 参数。

**设计意图**: 内存重排行算子。`torch.flip` 沿指定维度反转元素顺序（本质是索引重映射），ninetoothed 无法在 tile 内表达跨步幅的索引操作。

---

## 九、fliplr

### Kernel 层 (`learn/kernel/fliplr.py`)

标准的 copy kernel：`dst = src`，2 张量。

### Torch 接口层 (`learn/torch/fliplr.py`)

```python
def fliplr(input):
    return ntops.torch.flip(input, dims=(-1,))
```

一行委托。`fliplr` 是 `flip` 的特例——固定沿最后一维翻转。经过重构优化后，不再重复调用 `torch.flip` + kernel 拷贝，而是直接委托给 `ntops.torch.flip`，由 flip 统一完成翻转 + 拷贝。

**设计意图**: fliplr 是"语义别名"——它没有独特的计算逻辑，只是给 `flip(input, dims=(-1,))` 起了一个更直观的名字。委托给 flip 避免了代码重复，也体现了算子之间的层级关系。

---

## 十、meshgrid

### Kernel 层 (`learn/kernel/meshgrid.py`)

标准的 copy kernel：`dst = src`，2 张量。

### Torch 接口层 (`learn/torch/meshgrid.py`)

```python
def meshgrid(*tensors, indexing="ij"):
    grids = torch.meshgrid(*tensors, indexing=indexing)
    ndim = grids[0].ndim
    kernel = _cached_make(ntops.kernels.meshgrid.premake, ndim)
    outputs = []
    for g in grids:
        out = torch.empty_like(g)
        kernel(g, out)
        outputs.append(out)
    return tuple(outputs)
```

与其他 copy kernel 不同的是，meshgrid 产生**多个输出张量**（每个输入维度一个 grid）。torch 层先用 `torch.meshgrid` 生成所有 grid，然后在 for 循环中对每个 grid 调用一次 copy kernel（同一个编译好的 kernel，不同数据）。

**设计意图**: 同样是内存扩展算子（`torch.meshgrid` 本质是 broadcast），kernel 只做搬运。for 循环在这里是合理的——每个 grid 形状相同，kernel 只编译一次，循环中只是换数据。

---

## 十一、cartesian_prod

### Kernel 层 (`learn/kernel/cartesian_prod.py`)

标准的 copy kernel：`dst = src`，2 张量。

### Torch 接口层 (`learn/torch/cartesian_prod.py`)

```python
def cartesian_prod(*tensors):
    result = torch.cartesian_prod(*tensors)
    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.cartesian_prod.premake, result.ndim)
    kernel(result, out)
    return out
```

一行 `torch.cartesian_prod` 完成笛卡尔积计算，kernel 拷贝。接受多个 1D 张量，输出 2D 张量 `(M, N)`，其中 M 是各输入长度的乘积。

**设计意图**: 笛卡尔积是组合生成操作，逻辑复杂（多层嵌套循环），torch 层完成全部计算。

---

## 十二、pixel_unshuffle

### Kernel 层 (`learn/kernel/pixel_unshuffle.py`)

标准的 copy kernel：`dst = src`，2 张量。

### Torch 接口层 (`learn/torch/pixel_unshuffle.py`)

```python
def pixel_unshuffle(input, downscale_factor):
    result = torch.nn.functional.pixel_unshuffle(input, downscale_factor)
    out = torch.empty_like(result)
    kernel = _cached_make(ntops.kernels.pixel_unshuffle.premake, result.ndim)
    kernel(result, out)
    return out
```

一行 `F.pixel_unshuffle(input, downscale_factor)` 完成空间到深度的重排：`(N, C, H, W)` -> `(N, C*r^2, H/r, W/r)`。kernel 只做拷贝。

**设计意图**: pixel_unshuffle 是复杂的空间重排——把空间维度上的像素块重新排列到通道维度。这种跨维度重排 ninetoothed 无法在 tile 模型内表达。

---

## 十三、模式总结

### 算子分类决策树

```
这个算子的核心操作是？
  ├── 逐元素数值计算（加减乘除、条件选择、激活函数）
  │     → Real Kernel：计算逻辑放在 application() 中
  │     示例：leaky_relu, feature_alpha_dropout, mse_loss（平方部分）
  │
  └── 内存重排 / 规约 / 组合生成
        → Copy Kernel：逻辑放在 torch 层，kernel 只做 dst = src
        示例：roll, flip, meshgrid, mode, pixel_unshuffle
```

### Kernel 层规律

| 要素 | Real Kernel | Copy Kernel |
|------|------------|-------------|
| arrangement | `element_wise.arrangement` | `element_wise.arrangement` |
| application | 有 ntl 运算（where, abs, +, -, *） | `dst = src` |
| 张量数量 | 3~6 个（含标量和中间结果） | 2 个（src, dst），mode 是 4 个 |
| premake | 标量用 `Tensor(0)`，中间结果用 `Tensor(ndim)` | 输入输出同型 |

### Torch 层规律

| 算子类型 | torch 层职责 | 典型 API |
|---------|-------------|---------|
| 纯计算 | 分配输出 + 传参 + 调用 kernel | — |
| 重排 | torch API 完成逻辑 + kernel 拷贝 | torch.cat, torch.flip, torch.stack |
| 规约 | torch 完成复杂逻辑 + reduction | scatter_add_, argmax, sum, mean |
| 别名 | 一行委托给底层算子 | `return ntops.torch.flip(...)` |

### 分工原则（重述）

```
能在 GPU tile 内完成的计算  →  放进 kernel（application）
需要跨 tile 或跨张量的逻辑 →  放进 torch 接口（预处理/后处理）
```

- **tile 内**：逐元素运算、tile 内规约 → kernel
- **跨 tile**：排序、扫描、scatter、reduction → torch
- **跨张量**：拼接、切片、broadcast、reshape → torch