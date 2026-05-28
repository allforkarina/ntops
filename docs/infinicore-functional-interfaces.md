# InfiniCore nn.functional 接口知识手册

> 基于 [InfiniCore](https://github.com/allforkarina/InfiniCore/tree/main/python/infinicore/nn/functional) 源码分析，梳理全部 38 个功能接口的设计模式、调用链路与实现方式。

---

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 统一设计模式](#2-统一设计模式)
- [3. 接口分类详解](#3-接口分类详解)
  - [3.1 激活函数（8 个）](#31-激活函数)
  - [3.2 归一化层（2 个）](#32-归一化层)
  - [3.3 注意力机制（2 个）](#33-注意力机制)
  - [3.4 线性与嵌入（3 个）](#34-线性与嵌入)
  - [3.5 位置编码（1 个）](#35-位置编码)
  - [3.6 池化（3 个）](#36-池化)
  - [3.7 损失函数（8 个）](#37-损失函数)
  - [3.8 张量操作（5 个）](#38-张量操作)
  - [3.9 其他（6 个）](#39-其他)
- [4. 与 PyTorch 对照表](#4-与-pytorch-对照表)
- [5. 与 ntops 对照表](#5-与-ntops-对照表)
- [6. 开发一个新接口的模板](#6-开发一个新接口的模板)

---

## 1. 整体架构

### 调用链路

```
用户代码: F.silu(x)
    │
    ▼
┌─────────────────────────────────────┐
│  Python 包装层 (functional/*.py)     │
│  - 参数校验                          │
│  - 类型转换                          │
│  - 可选 ntops 加速路径               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  C++ 绑定层 (_infinicore)            │
│  - 接收 Python 参数                  │
│  - 调用底层 kernel                   │
│  - 返回 Tensor handle               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  GPU Kernel (CUDA/MUSA)             │
│  - 手写 CUDA 或 ninetoothed 编译     │
│  - 在 GPU 上并行执行                 │
└─────────────────────────────────────┘
```

**关键对象：**

| 对象 | 作用 | 示例 |
|-----|------|------|
| `infinicore.Tensor` | InfiniCore 的 Tensor 类型 | 包装底层 handle |
| `_infinicore` | C++ 扩展模块 | C 扩展导出的函数 |
| `tensor._underlying` | 底层 C++ Tensor handle | 传给 C++ 层 |

---

## 2. 统一设计模式

InfiniCore 所有 functional 接口遵循**完全一致的代码模式**。理解一个即理解全部：

### 模式 A：基本型（创建新输出）

```python
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def some_op(input: Tensor, param1, param2, *, out=None) -> Tensor:
    # 1. 参数校验（可选）
    if not input.is_contiguous():
        input = input.contiguous()

    # 2. 无 out → 创建新 Tensor
    if out is None:
        return Tensor(
            _infinicore.some_op(
                input._underlying,   # 解包底层 handle
                param1,              # Python 值直接传
                param2,
            )
        )

    # 3. 有 out → in-place 写入
    _infinicore.some_op_(
        out._underlying,   # 输出 handle 也解包
        input._underlying,
        param1,
        param2,
    )
    return out
```

**核心要点：**
- `tensor._underlying` 提取底层 C++ handle，传给 `_infinicore` 函数
- `Tensor(...)` 把底层 handle 包装成 InfiniCore Tensor
- `_infinicore.xxx()` 返回**新 Tensor** 的 handle
- `_infinicore.xxx_()` 写入已有 Tensor（下划线后缀 = in-place）

---

### 模式 B：带 ntops 加速的激活函数

```python
import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def silu(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    # ntops 加速路径（仅在 CUDA/MUSA 设备上生效）
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.silu(input, inplace=inplace)

    # 回退到 InfiniCore 原生 kernel
    if inplace:
        _infinicore.silu_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.silu(input._underlying))

    _infinicore.silu_(out._underlying, input._underlying)
    return out
```

**`infinicore.use_ntops`** 是一个功能开关，当启用时，优先走 ntops 编译的 kernel；否则回退到 InfiniCore 原生 C++/CUDA kernel。

---

### 模式 C：带 inplace 参数

```python
def hardswish(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    # ...
    if inplace:
        _infinicore.hardswish_(input._underlying, input._underlying)
        return input  # 输入和输出是同一个对象
```

**`inplace=True`** 时，`input._underlying` 作为输入也作为输出被修改。

---

### 命名约定

| 后缀 | 含义 | 示例 |
|-----|------|------|
| 无后缀 | 创建新 Tensor 返回 | `_infinicore.silu()` |
| `_` 后缀 | in-place 写入已有 Tensor | `_infinicore.silu_()` |

---

## 3. 接口分类详解

### 3.1 激活函数（8 个）

#### `silu(x)` — Sigmoid Linear Unit

```python
silu(input: Tensor, inplace: bool = False, *, out=None) -> Tensor
```

公式：`silu(x) = x * sigmoid(x)`

特点：支持 `infinicore.use_ntops` 加速路径。

---

#### `hardswish(x)` — Hard Swish

```python
hardswish(input: Tensor, inplace: bool = False, *, out=None) -> Tensor
```

公式：`hardswish(x) = x * clamp(x+3, 0, 6) / 6`

特点：比 silu 更省计算，移动端常用。支持 ntops 加速。

---

#### `hardtanh(x)` — Hard Tanh

```python
hardtanh(input: Tensor, inplace: bool = False, *, out=None) -> Tensor
```

公式：`hardtanh(x) = clamp(x, -1, 1)`

---

#### `relu6(x)` — ReLU6

```python
relu6(input: Tensor, *, out=None) -> Tensor
```

公式：`relu6(x) = clamp(x, 0, 6)`

---

#### `prelu(x, weight)` — Parametric ReLU

```python
prelu(input: Tensor, weight: Tensor) -> Tensor
```

公式：`prelu(x) = x if x > 0 else weight * x`

**注意：** 不提供 `out` 参数（无 in-place 版本）。

---

#### `selu(x)` — Scaled ELU

```python
selu(input: Tensor, *, out=None) -> Tensor
```

公式：`selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`

---

#### `softplus(x)` — Softplus

```python
softplus(input: Tensor, *, out=None) -> Tensor
```

公式：`softplus(x) = log(1 + exp(x))`

---

#### `softsign(x)` — Softsign

```python
softsign(input: Tensor, *, out=None) -> Tensor
```

公式：`softsign(x) = x / (1 + |x|)`

---

#### `tanhshrink(x)` — Tanh Shrink

```python
tanhshrink(input: Tensor, *, out=None) -> Tensor
```

公式：`tanhshrink(x) = x - tanh(x)`

---

### 3.2 归一化层（2 个）

#### `layer_norm(x, normalized_shape, weight, bias, eps)` — 层归一化

```python
layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
    *, out=None,
) -> Tensor
```

公式：`y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`

- `normalized_shape` 必须与 `weight.shape` 一致
- 支持 `out` 参数的 in-place 写入

---

#### `rms_norm(x, normalized_shape, weight, eps)` — RMS 归一化

```python
rms_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Tensor,
    eps: float = 1e-5,
    *, out=None,
) -> Tensor
```

公式：`y = x / sqrt(mean(x^2) + eps) * weight`

- 相比 LayerNorm 去掉了均值减法，仅用均方根做缩放
- Llama 等现代 LLM 的标准归一化方式

---

### 3.3 注意力机制（2 个）

#### `flash_attention(query, key, value, total_kv_len, ...)` — Flash Attention

```python
flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    total_kv_len: Tensor,
    attn_mask=None,
    dropout_p=0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> Tensor
```

- 使用 Flash Attention 算法，避免中间矩阵 O(n^2) 显存
- 当前版本不支持 `attn_mask`、`dropout_p`、`enable_gqa`
- `scale` 默认为 `1 / sqrt(emb_dim)`
- `total_kv_len` 是 KV Cache 的总长度（含历史）

---

#### `causal_softmax(x)` — Causal Softmax

```python
causal_softmax(input: Tensor, *, out=None) -> Tensor
```

- 在注意力计算中，沿 causal 方向做 softmax
- 下三角部分正常计算，上三角严格为 0

---

### 3.4 线性与嵌入（3 个）

#### `linear(input, weight, bias, alpha)` — 线性变换

```python
linear(
    input: Tensor,
    weight: Tensor,
    bias=None,
    *, alpha: float = 1.0,
    out=None,
) -> Tensor
```

公式：`y = alpha * input @ weight^T + bias`

- `alpha` 是输出缩放因子（标量）
- 支持无 bias 模式

---

#### `linear_w8a8i8(input, weight, scales, bias)` — INT8 量化线性层

```python
linear_w8a8i8(
    input: Tensor,
    weight: Tensor,
    scales: Tensor,
    bias=None,
    *, out=None,
) -> Tensor
```

- 输入和权重都是 INT8 量化
- `scales` 是反量化缩放因子
- 用于推理加速

---

#### `embedding(input, weight, ...)` — 词嵌入查表

```python
embedding(
    input: Tensor,        # 索引张量 [device-side 或 CPU]
    weight: Tensor,       # 嵌入矩阵 [vocab_size, emb_dim]
    padding_idx=None,     # 暂不支持
    max_norm=None,        # 暂不支持
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    *, out=None,
) -> Tensor
```

- `input` 可以是 device-side Tensor（支持 CUDA Graph）
- 使用向量化访存 (`float4/float2`) 和 `__ldg` 指令优化
- 暂不支持 padding、max_norm 等高级参数

---

### 3.5 位置编码（1 个）

#### `rope(x, pos_ids, sin_table, cos_table, algo)` — 旋转位置编码

```python
class RopeAlgo:
    GPT_J    = _infinicore.RoPEAlgo.GPT_J     # GPT-J 风格的 RoPE
    GPT_NEOX = _infinicore.RoPEAlgo.GPT_NEOX  # GPT-NeoX 风格的 RoPE

rope(
    x: Tensor,
    pos_ids: Tensor,
    sin_table: Tensor,
    cos_table: Tensor,
    algo: RopeAlgo = RopeAlgo.GPT_NEOX,
    *, out=None,
) -> Tensor
```

公式：

```
GPT_J 风格:  将相邻维度配对旋转
GPT_NEOX 风格: 将前半和后半维度配对旋转

（Llama 使用 GPT_NEOX）
```

---

### 3.6 池化（3 个）

#### `avg_pool1d(x, kernel_size, stride, padding)`

```python
avg_pool1d(
    input: Tensor,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
    *, out=None,
) -> Tensor
```

---

#### `adaptive_avg_pool1d(x, output_size)`

```python
adaptive_avg_pool1d(input: Tensor, output_size: int, *, out=None) -> Tensor
```

- 自动计算 kernel_size 和 stride，使输出为指定大小

---

#### `adaptive_max_pool1d(x, output_size)`

```python
adaptive_max_pool1d(input: Tensor, output_size: int, *, out=None) -> Tensor
```

---

### 3.7 损失函数（8 个）

#### 损失函数通用模式

所有损失函数都支持 `reduction` 参数（字符串 → 整数映射）：

```python
_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum":  2,
}
```

部分还要求输入内存连续：

```python
if not input.is_contiguous():
    input = input.contiguous()
```

---

| 函数 | 签名 | 说明 |
|-----|------|------|
| `huber_loss` | `(input, target, delta=1.0, reduction="mean", *, out)` | Huber 损失：delta 内 L2，外 L1 |
| `smooth_l1_loss` | `(input, target, beta=1.0, reduction="mean", *, out)` | Smooth L1：beta 内 L2，外 L1 |
| `binary_cross_entropy_with_logits` | `(input, target, weight=None, pos_weight=None, reduction="mean", *, out)` | 二分类交叉熵（带 logits） |
| `hinge_embedding_loss` | `(input, target, margin=1.0, reduction="mean", *, out)` | Hinge 嵌入损失 |
| `multi_margin_loss` | `(input, target, p=1, margin=1.0, weight=None, reduction="mean", *, out)` | 多分类间隔损失 |
| `triplet_margin_loss` | `(anchor, positive, negative, margin=1.0, p=2.0, swap=False, reduction="mean", *, out)` | 三元组间隔损失 |
| `triplet_margin_with_distance_loss` | `(anchor, positive, negative, distance_function=None, margin=1.0, swap=False, reduction="mean", *, out)` | 自定义距离的三元组损失 |
| `gaussian_nll_loss` | `(input, target, var, full=False, eps=1e-6, reduction="mean", *, out)` | 高斯负对数似然损失 |

---

### 3.8 张量操作（5 个）

| 函数 | 签名 | 说明 |
|-----|------|------|
| `pad` | `(input, pad: Sequence[int], mode="constant", value=0.0, *, out)` | 张量填充 |
| `interpolate` | `(input, size=None, scale_factor=None, mode="nearest", align_corners=None)` | 插值（nearest/bilinear） |
| `upsample_nearest` | `(input, size, scale_factor, *, out)` | 最近邻上采样 |
| `upsample_bilinear` | `(input, size, scale_factor, align_corners, *, out)` | 双线性上采样 |
| `unfold` | `(input, kernel_size, dilation=1, padding=0, stride=1, *, out)` | 滑动窗口展开 (im2col) |

---

### 3.9 其他（6 个）

| 函数 | 说明 |
|-----|------|
| `affine_grid` | 生成仿射变换网格（用于 grid_sample） |
| `log_softmax` | Log-Softmax（数值稳定版本） |
| `silu_and_mul` | SiLU + 乘法：将输入最后维分成两半，对前半做 SiLU 后与后半逐元素相乘 |
| `swiglu` | SwiGLU：silu(input) * other |
| `random_sample` | 随机采样 |
| `adaptive_avg_pool3d` | 自适应 3D 平均池化 |

---

## 4. 与 PyTorch 对照表

| PyTorch `torch.nn.functional.xxx` | InfiniCore `infinicore.nn.functional.xxx` | 差异 |
|----------------------------------|------------------------------------------|------|
| `F.silu(x)` | `F.silu(x, inplace=False)` | InfiniCore 多 `inplace` 参数 |
| `F.relu6(x)` | `F.relu6(x)` | 相同 |
| `F.prelu(x, weight)` | `F.prelu(x, weight)` | 相同 |
| `F.layer_norm(x, ...)` | `F.layer_norm(x, ...)` | 相同 |
| `F.linear(x, w, b)` | `F.linear(x, w, b, alpha=1.0)` | InfiniCore 多 `alpha` 缩放因子 |
| `F.scaled_dot_product_attention(...)` | `F.flash_attention(q, k, v, total_kv_len, ...)` | InfiniCore 多 `total_kv_len`，少 `dropout_p` |
| `F.embedding(x, w)` | `F.embedding(x, w)` | 相同（高级参数暂不支持） |
| `F.huber_loss(x, t)` | `F.huber_loss(x, t, delta, reduction)` | 相同 |
| `F.pad(x, pad, mode, value)` | `F.pad(x, pad, mode, value)` | 相同 |
| `F.interpolate(x)` | `F.interpolate(x)` | InfiniCore 功能简化 |
| `F.unfold(x)` | `F.unfold(x)` | 相同 |

---

## 5. 与 ntops 对照表

InfiniCore 部分算子有 **ntops 加速路径**（通过 `infinicore.use_ntops` 标志控制）：

| InfiniCore 算子 | ntops 加速 | ntops 对应文件 |
|----------------|-----------|---------------|
| `silu` | ✅ | `ntops.torch.silu` |
| `hardswish` | ✅ | `ntops.torch.hardswish` (需 `hasattr` 检查) |
| `hardtanh` | ❌ | 无 ntops 加速路径 |
| `relu6` | ❌ | 无 ntops 加速路径 |
| `layer_norm` | ❌ | InfiniCore 原生实现 |
| `rms_norm` | ❌ | InfiniCore 原生实现 |
| `flash_attention` | ❌ | InfiniCore 原生实现 |
| `linear` | ❌ | InfiniCore 原生实现 |

**关键差异：**
- InfiniCore 的底层是 **手写 CUDA C++**，直接编译为 C++ 扩展
- ntops 使用 **ninetoothed DSL** 自动编译为 CUDA kernel
- ntops 适合快速开发新算子，InfiniCore 适合对性能有极致要求的场景

---

## 6. 开发一个新接口的模板

如果你想在 InfiniCore 中添加一个新的 functional 接口（例如 `my_op`），标准流程如下：

### Step 1: 创建 Python 包装文件

`python/infinicore/nn/functional/my_op.py`：

```python
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def my_op(
    input: Tensor,
    param: float = 0.5,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""Apply the MyOp function element-wise.

    Formula: y = my_formula(x, param)
    """
    # 可选：处理非连续内存
    if not input.is_contiguous():
        input = input.contiguous()

    if out is None:
        return Tensor(
            _infinicore.my_op(input._underlying, param)
        )

    _infinicore.my_op_(out._underlying, input._underlying, param)
    return out
```

### Step 2: 在 `__init__.py` 中注册

```python
from .my_op import my_op
```

### Step 3: 实现 C++/CUDA kernel

在 C++ 层实现 `my_op` 和 `my_op_` 两个函数（不在本手册范围）。

### Step 4: 编写测试

参照现有测试文件的模式：

```python
def test_my_op():
    input = torch.randn(16, 32, device="cuda")
    inf_output = F.my_op(infinicore.Tensor(input))
    ref_output = ...  # PyTorch 参考实现
    assert torch.allclose(inf_output.torch(), ref_output)
```

---

## 附录：完整接口清单（38 个）

```
激活函数 (8):
  silu, hardswish, hardtanh, relu6, prelu, selu, softplus, softsign, tanhshrink

融合算子 (2):
  silu_and_mul, swiglu

归一化 (2):
  layer_norm, rms_norm

注意力 (2):
  flash_attention, causal_softmax

线性与嵌入 (3):
  linear, linear_w8a8i8, embedding

位置编码 (1):
  rope

池化 (3):
  avg_pool1d, adaptive_avg_pool1d, adaptive_avg_pool3d, adaptive_max_pool1d

损失函数 (8):
  huber_loss, smooth_l1_loss, binary_cross_entropy_with_logits,
  hinge_embedding_loss, multi_margin_loss, triplet_margin_loss,
  triplet_margin_with_distance_loss, gaussian_nll_loss

张量操作 (5):
  pad, interpolate, upsample_nearest, upsample_bilinear, unfold

其他 (4):
  affine_grid, log_softmax, random_sample
```