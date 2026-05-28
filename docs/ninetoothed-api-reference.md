# 🔧 ninetoothed 常用 API 参考手册

> 面向算子开发者：涵盖 ntops 项目中所有实际使用的 ninetoothed 库函数、语言原语和辅助工具。

---

## 📋 目录

- [1. 核心框架函数](#1-核心框架函数)
- [2. Tensor 类](#2-tensor-类)
- [3. 数据类型](#3-数据类型)
- [4. 语言原语 (ntl.\*)](#4-语言原语-ntl)
  - [数学函数](#数学函数)
  - [张量操作](#张量操作)
  - [条件与控制流](#条件与控制流)
  - [随机与辅助](#随机与辅助)
- [5. libdevice 扩展](#5-libdevice-扩展)
- [6. Dtype 属性与方法](#6-dtype-属性与方法)
- [7. arrangement 布局策略](#7-arrangement-布局策略)
- [8. 快速对照表](#8-快速对照表)

---

## 1. 核心框架函数

### `ninetoothed.make()`

**作用：** 将 Python DSL 编译为 GPU 可执行的 CUDA kernel。

```python
# torch/utils.py 中的实际用法
return ninetoothed.make(
    *premake(*args, **keywords),
    num_warps=num_warps,
    num_stages=num_stages,
    max_num_configs=max_num_configs,
)
```

| 参数 | 类型 | 说明 |
|-----|------|------|
| 第1个位置参数 | callable | `arrangement_` — 内存布局函数 |
| 第2个位置参数 | callable | `application` — 计算逻辑函数 |
| 第3个位置参数 | tuple | `tensors` — Tensor 声明元组 |
| `num_warps` | int / None | GPU warp 数量（None = 自动） |
| `num_stages` | int / None | 流水线阶段数（None = 自动） |
| `max_num_configs` | int / None | 最大配置数（None = 自动） |

**返回值：** 一个可调用的 kernel 对象。调用方式与 `application()` 参数一致。

**类比：**
```
ninetoothed.make() = 编译器
  输入：Python 写的"计算逻辑"（application）+ "内存布局"（arrangement）+ "变量声明"（tensors）
  输出：已编译的 GPU 二进制程序（kernel 对象）
```

---

### `ninetoothed.block_size()`

**作用：** 返回 ninetoothed 自动推断的 GPU 线程块大小。

```python
# element_wise.py 中的用法
def arrangement(*tensors, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()  # 自动推断
```

在实际使用中，你**不需要手动设置** `block_size`。如果 `premake()` 中传了 `None`，`arrangement` 函数内部会自动调用 `ninetoothed.block_size()`。

---

## 2. Tensor 类

### `Tensor(ndim, dtype=None, ...)`

**作用：** 声明一个张量的元信息（不是创建真实数据，只是类型声明）。

```python
from ninetoothed import Tensor
```

#### 基本用法

| 构造参数 | 说明 | 示例 |
|---------|------|------|
| `ndim` | 维度数（0 = 标量，1 = 向量，2 = 矩阵…） | `Tensor(2)` |
| `dtype` | 数据类型 | `Tensor(2, dtype=torch.float32)` |
| `other` | 默认值（通常用于特殊数值标记） | `Tensor(ndim, other=float("-inf"))` |
| `shape_options` | 形状选项（如标记为编译时常量） | `shape_options={"constexpr": True}` |

#### 在项目中的实际模式

```python
# 模式 1: 最简单的一元运算（1个输入 + 1个输出 = 2个Tensor）
tensors = (
    Tensor(ndim, dtype=dtype),   # 输入
    Tensor(ndim, dtype=dtype),   # 输出
)

# 模式 2: 带标量参数的一元运算（3个Tensor）
tensors = (
    Tensor(ndim, dtype=dtype),               # 输入
    Tensor(0, dtype=ninetoothed.float64),    # 标量参数
    Tensor(ndim, dtype=dtype),               # 输出
)

# 模式 3: 二元运算（3个Tensor，同维度）
tensors = (
    Tensor(ndim, dtype=dtype),   # 输入A
    Tensor(ndim, dtype=dtype),   # 输入B
    Tensor(ndim, dtype=dtype),   # 输出
)

# 模式 4: 带标量参数的二元运算（4个Tensor）
tensors = (
    Tensor(ndim, dtype=dtype),               # 输入A
    Tensor(ndim, dtype=dtype),               # 输入B
    Tensor(0, dtype=ninetoothed.float64),    # 标量参数
    Tensor(ndim, dtype=dtype),               # 输出
)

# 模式 5: 多个标量参数
tensors = (
    Tensor(ndim, dtype=dtype),               # 输入
    Tensor(0, dtype=ninetoothed.float64),    # 标量1（float）
    Tensor(0, dtype=ninetoothed.int64),      # 标量2（int）
    Tensor(ndim, dtype=dtype),               # 输出
)
```

**关键规则：** `tensors` 元组中 Tensor 的顺序，对应 `application()` 函数参数从左到右的顺序。

**类比：** `Tensor()` 就像 C 语言中的变量声明 `int x; float y;`——只声明类型，不赋值。

---

#### 高级用法：`other` 和 `shape_options`

```python
# softmax.py 中的用法 —— 带默认值和编译时常量
Tensor(
    ndim,
    dtype=dtype,
    other=float("-inf"),                    # 数据中特殊值标记
    shape_options={"constexpr": True}       # 形状是编译时常量
)
```

| 参数 | 含义 | 何时使用 |
|-----|------|---------|
| `other` | 张量中表示"特殊值"的数值 | softmax 用 `-inf` 标记 padding 位置 |
| `shape_options={"constexpr": True}` | 声明形状在编译时已知 | 归约类算子、注意力机制 |

---

## 3. 数据类型

ninetoothed 支持的数据类型别名，在 `Tensor()` 的 `dtype` 参数中使用：

| ninetoothed 类型 | 对应 Python/NumPy | 对应 PyTorch | 精度 |
|-----------------|-------------------|-------------|------|
| `ninetoothed.float16` | `numpy.float16` | `torch.float16` | 半精度（16位） |
| `ninetoothed.float32` | `numpy.float32` | `torch.float32` | 单精度（32位） |
| `ninetoothed.float64` | `numpy.float64` | `torch.float64` | 双精度（64位） |
| `ninetoothed.int32` | `numpy.int32` | `torch.int32` | 32位整数 |
| `ninetoothed.int64` | `numpy.int64` | `torch.int64` | 64位整数 |

同样可通过 `ntl.float16`、`ntl.float32`、`ntl.int64` 等方式访问：

```python
import ninetoothed.language as ntl

# 在 application() 中使用的类型
exp_dtype = dtype if dtype != ntl.float16 else ntl.float32
```

**常用约定：** 标量参数（如 `alpha`、`p`）通常声明为 `Tensor(0, dtype=ninetoothed.float64)`，以获得最高精度。

---

## 4. 语言原语 (ntl.\*)

用法：`import ninetoothed.language as ntl`

> **重要：** 在 `application()` 函数中不能使用 Python 内置函数（如 `math.exp()`、`abs()`），必须使用 `ntl.*` 系列函数，因为它们会被编译为 GPU 指令。

---

### 数学函数

#### `ntl.exp(x)`
**作用：** 计算 e^x（自然指数）。

```python
# exp.py
output = ntl.exp(input)

# tanh.py
exp_input = ntl.exp(input)
exp_neg_input = ntl.exp(-input)
```

**对标：** Python `math.exp()` / PyTorch `torch.exp()`

---

#### `ntl.sqrt(x)`
**作用：** 计算平方根 √x。

```python
# gelu.py
output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))

# layer_norm.py
std = ntl.sqrt(var + eps)
```

**对标：** Python `math.sqrt()` / PyTorch `torch.sqrt()`

---

#### `ntl.rsqrt(x)`
**作用：** 计算 1/√x（平方根的倒数）。

```python
# rsqrt.py
output = ntl.rsqrt(ntl.cast(input, ntl.float32))
```

**对标：** PyTorch `torch.rsqrt()`

---

#### `ntl.abs(x)`
**作用：** 计算绝对值 |x|。

```python
# abs.py
output = ntl.abs(input)
```

**对标：** Python `abs()` / PyTorch `torch.abs()`

---

#### `ntl.sin(x)` / `ntl.cos(x)`
**作用：** 正弦 / 余弦函数。

```python
# sin.py
output = ntl.sin(input)

# cos.py
output = ntl.cos(input)
```

---

#### `ntl.tanh(x)`
**作用：** 双曲正切函数 tanh(x)。

```python
# 方式一：直接调用
# sigmoid.py 等可直接使用高级函数
output = ntl.sigmoid(input)

# 方式二：手动实现
output = (ntl.exp(input) - ntl.exp(-input)) / (ntl.exp(input) + ntl.exp(-input))
```

---

#### `ntl.sigmoid(x)`
**作用：** Sigmoid 函数 1/(1+e^(-x))。

```python
# sigmoid.py
output = ntl.sigmoid(input)
```

---

#### `ntl.erf(x)`
**作用：** 误差函数（Gaussian error function）。

```python
# gelu.py
output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))
```

---

#### `ntl.exp2(x)`
**作用：** 计算 2^x（以 2 为底的指数）。

```python
# scaled_dot_product_attention.py
stable_qk = ntl.exp2(qk - next_max[:, None])
alpha = ntl.exp2(max - next_max)
```

**性能提示：** `ntl.exp2()` 比 `ntl.exp()` 快，适合注意力机制等高性能场景。

---

### 张量操作

#### `ntl.cast(x, dtype)`
**作用：** 类型转换，将 `x` 转换为目标 `dtype`。

```python
# softmax.py —— 三重转换模式
return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)
#               ↑↑↑↑↑↑↑           ↑↑↑↑↑↑↑
#               从 exp_dtype       从原始类型
#               转回 dtype         转为 exp_dtype

# 标量初始化
prev_max = ntl.cast(float("-inf"), dtype)   # Python float → GPU dtype
denominator = ntl.cast(0, dtype)             # Python int   → GPU dtype
```

**使用场景：**
- float16 计算时转 float32 提升精度（最常用）
- Python 常量转为 GPU 类型
- 整数除法前做类型转换

**对标：** PyTorch `tensor.to(dtype)`

---

#### `ntl.max(x, axis=...)`
**作用：** 沿指定轴取最大值（归约操作）。

```python
# softmax.py
curr_max = ntl.maximum(prev_max, ntl.max(input_i))
#                                     ↑↑↑↑↑↑↑↑↑
#                                     沿所有维度取最大值，返回标量

# max_pool2d.py
output = ntl.max(input, axis=-1)
#                       ↑↑↑↑↑↑
#                       沿最后一个轴取最大
```

**对标：** PyTorch `torch.max(x, dim=...)`

---

#### `ntl.maximum(a, b)`
**作用：** 逐元素取最大值（非归约，输入和输出形状相同）。

```python
# relu.py —— 用 max(0, x) 实现 ReLU
output = max(0.0, input)  # Python 内置 max 也可用，会被 ninetoothed 识别

# softmax.py
curr_max = ntl.maximum(prev_max, ntl.max(input_i))
#          ↑↑↑↑↑↑↑↑↑↑↑  ← 逐元素比较
```

**对标：** PyTorch `torch.maximum(a, b)`

---

#### `ntl.sum(x, axis=...)`
**作用：** 沿指定轴求和（归约操作）。

```python
# softmax.py
denominator = denominator * prev_score + ntl.sum(input_max_diff_exp)

# layer_norm.py
mean = ntl.sum(_mean, 0) / num_normalized_elements
var = ntl.sum(_var, 0) / num_normalized_elements
```

**对标：** PyTorch `torch.sum(x, dim=...)`

---

#### `ntl.zeros(shape, dtype=...)`
**作用：** 创建全零张量（用于累加器初始化）。

```python
# layer_norm.py
_mean = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
_var = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

# addmm.py
mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
```

**对标：** PyTorch `torch.zeros(shape, dtype=...)`

---

#### `ntl.full(shape, value, dtype=...)`
**作用：** 创建填充为常数值的张量。

```python
# scaled_dot_product_attention.py
lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
max = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)
```

**对标：** PyTorch `torch.full(shape, value, dtype=...)`

---

#### `ntl.dot(a, b, input_precision=...)`
**作用：** 两个矩阵的乘法（内积）。

```python
# mm.py
accumulator += ntl.dot(input[k], other[k], input_precision=input_precision_)

# scaled_dot_product_attention.py
qk = ntl.dot(query_i, ntl.trans(key[j]))
```

**对标：** PyTorch `torch.matmul(a, b)`

---

#### `ntl.trans(x)`
**作用：** 矩阵转置。

```python
# scaled_dot_product_attention.py
qk = ntl.dot(query_i, ntl.trans(key[j]))
```

---

### 条件与控制流

#### `ntl.where(condition, x, y)`
**作用：** 三元条件选择。`condition` 为真时取 `x`，否则取 `y`。

```python
# bitwise_not.py — 逻辑取反
output = ntl.where(input, False, True)

# dropout.py — 随机丢弃
output = ntl.where(ntl.rand(seed, input.offsets()) > p, input / (1 - p), 0)

# scaled_dot_product_attention.py — mask 处理
qk = ntl.where(mask, qk, float("-inf"))

# layer_norm.py — padding 跳过
diff = ntl.where(input[i].offsets(-1) < input.source.shape[-1], diff, 0)
```

**对标：** PyTorch `torch.where(condition, x, y)` / Python `x if cond else y`

---

#### `ntl.clamp(x, min, max)`
**作用：** 将值限制在 `[min, max]` 范围内。

```python
# clamp.py
output = ntl.clamp(input, min_val, max_val)
```

**对标：** PyTorch `torch.clamp(x, min, max)`

---

#### `ntl.rand(seed, offsets)`
**作用：** 生成随机数（用于 dropout 等）。

```python
# dropout.py
ntl.rand(seed, input.offsets())
```

---

### 随机与辅助

#### `ntl.zeros(shape, dtype=...)`
（见上文张量操作部分）

---

## 5. libdevice 扩展

#### `libdevice.pow(base, exponent)`

```python
from ninetoothed.language import libdevice

# pow.py
output = libdevice.pow(input, exponent)
```

**说明：** `libdevice` 提供了对 NVIDIA libdevice 库的访问，包含更底层的数学函数。`libdevice.pow()` 是幂运算的标准实现。

**何时使用 vs ntl.\*：** 当 `ntl.*` 中没有对应函数时使用 `libdevice`。大多数常用操作 `ntl` 已覆盖。

---

## 6. Dtype 属性与方法

在 `application()` 中可通过张量的 `.dtype` 属性访问类型信息：

### `.dtype.dtype`
**作用：** 获取张量的数据类型。

```python
# softmax.py
dtype = output.dtype.dtype  # Tensor -> dtype of element
```

### `.dtype.shape`
**作用：** 获取张量的形状信息。

```python
# layer_norm.py
_mean = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
```

### `.offsets(dim=None)`
**作用：** 获取张量在指定维度的偏移量。

```python
# dropout.py
ntl.rand(seed, input.offsets())

# layer_norm.py
input[i].offsets(-1)  # 最后一个维度的偏移
```

### `.source.shape[dim]`
**作用：** 访问原始张量的形状。

```python
# layer_norm.py
input.source.shape[-1]  # 原始张量最后一维的大小
```

### `.flatten()` / `.tile()` / `.permute()` / `.squeeze()`
这些是 arrangement 策略中使用的张量操作方法：

```python
# element_wise.py
tensor.flatten().tile((block_size,))

# reduction.py
arranged = tensor.permute(non_target_dims + dims)
arranged = arranged.flatten(start_dim=-len(dims))
arranged = arranged.tile(inner_block_shape)
arranged.dtype = arranged.dtype.squeeze(non_target_dim_indices)
```

> **注意：** 这些方法主要用于 `arrangement` 函数中（预处理阶段），在 `application()` 中较少直接使用。

---

## 7. arrangement 布局策略

### element_wise.arrangement

```python
from ntops.kernels.element_wise import arrangement

def arrangement(*tensors, block_size=None):
    # 将所有张量展平为一维，按 block_size 分块
    return tuple(
        tensor.flatten().tile((block_size,)) if tensor.ndim != 0 else tensor
        for tensor in tensors
    )
```

**适用场景：** 逐元素独立运算（add, relu, mul, abs, neg, exp, tanh…）

**特点：**
- 每个元素计算互不依赖
- 标量（`ndim=0`）不做处理
- 自动处理 block_size

---

### reduction.arrangement

```python
from ntops.kernels.reduction import arrangement

def arrangement(*tensors, dim, block_size=None):
    # 沿指定维度归约
    # 重排维度 → 展平目标维度 → 分块
```

**适用场景：** 需要沿某维度聚合的操作（softmax, sum, max pooling…）

**特点：**
- 支持多维度归约
- 支持负索引（`dim=-1` 表示最后一维）
- 更复杂的内存重排逻辑

---

### 选择策略

| 操作类型 | 示例 | 使用策略 |
|---------|------|---------|
| 逐元素一元 | relu, neg, abs, exp, sin, cos | `element_wise.arrangement` |
| 逐元素二元 | add, sub, mul, div | `element_wise.arrangement` |
| 逐元素三元 | clamp, dropout | `element_wise.arrangement` |
| 归约操作 | softmax, layer_norm, max_pool2d | `reduction.arrangement` |
| 矩阵乘法 | mm, addmm | 自定义 arrangement |

---

## 8. 快速对照表

### 我要实现的功能 → 对应的 API

| 我需要… | 用这个 |
|--------|--------|
| 创建算子模板 | `from ninetoothed import Tensor` + `def application() + def premake()` |
| 声明一个张量变量 | `Tensor(ndim, dtype=...)` |
| 声明一个标量参数 | `Tensor(0, dtype=ninetoothed.float64)` |
| 编译成 GPU kernel | `ninetoothed.make(arrangement_, application, tensors)` |
| 自动推断 block size | `ninetoothed.block_size()` |
| 绑定固定参数 | `functools.partial(arrangement, block_size=block_size)` |

| 我需要数学计算… | 用 ntl 的… | 代替 Python 的… |
|---------------|-----------|----------------|
| 指数 e^x | `ntl.exp(x)` | `math.exp(x)` |
| 平方根 | `ntl.sqrt(x)` | `math.sqrt(x)` |
| 绝对值 | `ntl.abs(x)` | `abs(x)` |
| 正弦 | `ntl.sin(x)` | `math.sin(x)` |
| 余弦 | `ntl.cos(x)` | `math.cos(x)` |
| 双曲正切 | `ntl.tanh(x)` 或手动实现 | `math.tanh(x)` |
| 误差函数 | `ntl.erf(x)` | `math.erf(x)` |
| sigmoid | `ntl.sigmoid(x)` | `torch.sigmoid(x)` |
| 幂运算 | `libdevice.pow(x, e)` | `math.pow(x, e)` |

| 我需要张量操作… | 用 ntl 的… | 对标 PyTorch 的… |
|---------------|-----------|-----------------|
| 类型转换 | `ntl.cast(x, dtype)` | `x.to(dtype)` |
| 逐元素最大值 | `ntl.maximum(a, b)` | `torch.maximum(a, b)` |
| 沿轴取最大值 | `ntl.max(x, axis=n)` | `torch.max(x, dim=n)` |
| 求和 | `ntl.sum(x, axis=n)` | `torch.sum(x, dim=n)` |
| 矩阵乘法 | `ntl.dot(a, b)` | `torch.matmul(a, b)` |
| 矩阵转置 | `ntl.trans(x)` | `x.T` |
| 创建全零张量 | `ntl.zeros(shape, dtype)` | `torch.zeros(shape, dtype)` |
| 创建常量张量 | `ntl.full(shape, val, dtype)` | `torch.full(shape, val, dtype)` |

| 我需要条件逻辑… | 用 ntl 的… | 对标 Python 的… |
|---------------|-----------|----------------|
| 三元选择 | `ntl.where(cond, a, b)` | `a if cond else b` |
| 值裁剪 | `ntl.clamp(x, min, max)` | `max(min, min(x, max))` |
| 随机数 | `ntl.rand(seed, offsets)` | `random.random()` |

---

## 💡 总结：写一个算子的最小 API 集合

对于像 leaky_relu 这样的**简单元素级算子**，你只需要掌握：

```python
# 1. 导入（固定写法）
import functools
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement

# 2. 声明变量（Tensor）
Tensor(ndim, dtype=dtype)                 # 张量
Tensor(0, dtype=ninetoothed.float64)      # 标量

# 3. 写计算逻辑（application）
output = input if input > 0 else alpha * input   # Python 语法即可
output = ntl.exp(input)                           # 数学函数用 ntl
output = ntl.clamp(input, min, max)                # 复杂操作用 ntl

# 4. 配置（premake）
arrangement_ = functools.partial(arrangement, block_size=block_size)
return arrangement_, application, tensors
```

**记住：** 90% 的简单算子只需要 `Tensor(ndim)` + Python 基础运算符 + `ntl.*` 数学函数。