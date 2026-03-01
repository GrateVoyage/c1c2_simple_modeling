# C1V1C2V2 流水线建模

昇腾NPU Flash Attention算子完整流水线性能建模工具。

Flash Attention包含四个阶段：

| 阶段 | 操作 | 说明 |
|------|------|------|
| C1 | Q @ K^T → P | 第一次矩阵乘（Attention Score） |
| V1 | Softmax(P) | 第一次向量操作（1600 cycles） |
| C2 | P @ V → O | 第二次矩阵乘（Output） |
| V2 | 后处理(O) | 第二次向量操作（400 cycles） |

---

## 项目结构

```
c1c2_modeling/
├── core/                        # 核心定义
│   ├── enums.py                # 枚举类型
│   ├── dataclasses.py          # 数据结构
│   └── hardware_config.py      # 硬件参数配置
│
├── modelers/                    # 建模器
│   ├── c1_modeler.py           # C1Modeler 主类
│   └── templates/              # 配置模板
│       ├── standard.py         # 标准配置
│       ├── dn_mode.py          # DN模式配置
│       └── full_load.py        # Full Load配置
│
├── utils/                       # 工具
│   ├── visualizer.py           # 时间线可视化
│   └── logger.py               # 日志
│
├── examples/                    # 示例
│   ├── run_c1_examples.py      # 各配置示例
│   ├── test_full_pipeline.py   # 完整流水线演示
│   └── test_preload.py         # 核间流水模式对比
│
├── tests/                       # 测试（50个）
├── outputs/                     # 生成的时间线图片
└── reference.md                 # 硬件规格参考文档
```

---

## 快速开始

### 安装依赖

```bash
pip install matplotlib numpy
```

### 运行示例

```bash
# 各种配置示例（生成 outputs/*.png）
python examples/run_c1_examples.py

# 完整流水线演示
python examples/test_full_pipeline.py

# 核间流水模式对比（DEFAULT / PRELOAD / TWOBUFFER）
python examples/test_preload.py
```

---

## C1Modeler 完整参数说明

```python
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

modeler = C1Modeler(
    # ── 矩阵总尺寸 ──────────────────────────────────────
    s1_total=256,           # Q 序列长度（总）
    s2_total=512,           # K/V 序列长度（总）
    d_total=128,            # 特征维度（总）

    # ── 基本块大小（每次迭代处理的块） ──────────────────
    s1_base_size=128,       # Q 块的 s1 维度
    s2_base_size=128,       # K/V 块的 s2 维度
    d_base_size=128,        # 块的 d 维度

    # ── 矩阵乘内部切分（C1: Q @ K^T）───────────────────
    baseM_C1=128,           # C1 MAC 基本块 M 上限
    baseN_C1=128,           # C1 MAC 基本块 N 上限（超出则 matmulN 切分）
    baseK_C1=128,           # C1 MAC 基本块 K 上限（超出则 matmulK 切分）

    # ── 矩阵乘内部切分（C2: P @ V）──────────────────────
    baseM_C2=128,           # C2 MAC 基本块 M 上限
    baseN_C2=128,           # C2 MAC 基本块 N 上限（超出则 matmulN 切分）
    baseK_C2=128,           # C2 MAC 基本块 K 上限（超出则 matmulK 切分）

    # ── 数据类型 ─────────────────────────────────────────
    q_data_type=DataType.FP16,   # Q 矩阵精度（FP16 / FP8）
    kv_data_type=DataType.FP16,  # K/V 矩阵精度（FP16 / FP8）

    # ── 存储和带宽策略 ───────────────────────────────────
    is_l2cache=True,        # True: 相同块第二次加载用 L2 带宽；False: 全用 DRAM
    use_dn=False,           # True: DN模式（Q/K 路径互换，见下文）
    L1_db=True,             # True: L1 使用双缓冲（需块大小 ≤ L1 slot 上限）
    L0_db=True,             # True: L0A/L0B 使用双缓冲（需块 ≤ 32 KB）
    full_load=False,        # True: Q 在仿真开始前全量加载到 L1，后续不触发 MTE2

    # ── 流水线模式 ───────────────────────────────────────
    inter_core_pipeline=InterCorePipeline.DEFAULT,   # 核间流水（见下文）
    inner_core_pipeline=InnerCorePipeline.DEFAULT,   # 核内流水（当前仅 DEFAULT）

    # ── PRELOAD 专用 ─────────────────────────────────────
    two_buffer=False,       # 仅 PRELOAD 模式有效：True 时 V 用独立 L1 slot
)
```

---

## 数据类型

```python
from core import DataType

DataType.FP16   # 半精度（元素 2 字节）
DataType.FP8    # 8位浮点（元素 1 字节）
```

**MAC 吞吐量规则**：
- C1 (Q@K^T)：`q_data_type` **和** `kv_data_type` 同时为 FP8 → FP8 吞吐；否则 FP16 吞吐
- C2 (P@V)：由 `kv_data_type` 单独决定

**FIXPIPE 输出大小恒为 FP32**（L0C 累加结果），与数据类型无关：
- FIXPIPE P：`s1_base × s2_base × 4 bytes`
- FIXPIPE O：`s1_base × d_base × 4 bytes`

---

## 核间流水模式（InterCorePipeline）

### DEFAULT — 顺序流水

```
执行顺序: C1V1C2V2 → C1V1C2V2 → ... → C1V1C2V2
```

每个基本块完整走完 C1→V1→C2→V2 后再处理下一块。

```python
inter_core_pipeline=InterCorePipeline.DEFAULT
```

### PRELOAD — 渐进式流水

```
执行顺序: C1 → C1V1C2 → C1V1C2V2 → C1V1C2V2 → ... → V1C2V2 → C2V2 → V2
```

V 矩阵在 K 加载完成后立即预加载（不等 V1 结束），与 C1/V1 计算并行。C2 阶段跳过 MTE2-V，直接 MTE1 从 L1→L0。

```python
inter_core_pipeline=InterCorePipeline.PRELOAD
inter_core_pipeline=InterCorePipeline.PRELOAD, two_buffer=True   # V 使用独立 L1 slot，无 slot 竞争
```

**性能**：典型场景下比 DEFAULT 快约 20%。

### N_BUFFER — N=2 批次流水

```
执行顺序: C1C1 → V1V1 → C2C2 → V2V2 → C1C1 → V1V1 → ...
```

每组 2 个 k-block，按阶段批次执行。V1[k0] 与 MAC-C1[k1] 可在不同硬件上并行。

```python
inter_core_pipeline=InterCorePipeline.N_BUFFER
```

---

## 矩阵乘切分（baseM/N/K 参数）

当基本块的某维度超过 base 上限时，自动切分：

| 切分类型 | 触发条件（C1 为例） | 行为 |
|----------|---------------------|------|
| **matmulFull** | `s2_base ≤ baseN_C1` 且 `d_base ≤ baseK_C1` | 不切分，单次 MAC |
| **matmulK** | `d_base > baseK_C1` | 沿 K 轴切分，子 MAC 在 L0C 累加，最后一次 FIXPIPE |
| **matmulN** | `s2_base > baseN_C1` | 沿 N 轴切分，每个子块独立 FIXPIPE |

C2 同理，使用 `baseM/N/K_C2`，矩阵维度为 `(s1_base, d_base, s2_base)`。

**示例（matmulK）**：`d_base_size=256, baseK_C1=128` → 切成 2 个子 MAC，共享 L0C 累加，只有 1 次 FIXPIPE。

---

## DN 模式

标准模式与 DN 模式的数据路径区别：

| | 标准模式 | DN 模式 |
|-|----------|---------|
| Q 路径 | L1A → L0A | L1B → L0B |
| K 路径 | L1B → L0B | L1A → L0A |
| V 路径 | L1B → L0B | L1A → L0A |

```python
use_dn=True   # 开启 DN 模式
```

---

## Full Load 模式

Q 矩阵在仿真开始前一次性全量加载到 L1，后续所有 k 块迭代不再触发 Q 的 MTE2。

```python
full_load=True
```

---

## 配置模板

提供开箱即用的配置组合：

```python
from modelers.templates import StandardConfig, DNModeConfig, FullLoadConfig
from core import DataType, InterCorePipeline, InnerCorePipeline

config = StandardConfig(
    s1_total=256, s2_total=512, d_total=128,
    s1_base_size=128, s2_base_size=128, d_base_size=128,
    q_data_type=DataType.FP16,
    kv_data_type=DataType.FP16,
    # 其余参数有合理默认值
)
modeler = C1Modeler(**config.to_dict())
```

| 模板 | 说明 |
|------|------|
| `StandardConfig` | 通用基础配置 |
| `DNModeConfig` | 自动开启 `use_dn=True` |
| `FullLoadConfig` | 自动开启 `full_load=True` |

---

## 运行仿真

```python
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()

# 打印性能报告
modeler.print_performance(unit_times, total_cycles)

# 生成时间线图（保存到 outputs/ 目录）
modeler.plot_timeline(timeline, unit_times, total_cycles, "outputs/my_timeline.png")
```

返回值：
- `timeline`：事件列表（`List[TimelineEvent]`）
- `bound_type`：瓶颈单元（`BoundType` 枚举）
- `unit_times`：各单元总耗时字典
- `total_cycles`：总周期数

---

## 自定义硬件配置

```python
from core import HardwareConfig

hw = HardwareConfig(
    CHIP_FREQ_GHZ=1.65,
    MTE2_DRAM_BW_GBPS=1600,
    MTE2_L2_BW_GBPS=5400,
    MTE1_FIXPIPE_BYTES_PER_CYCLE=256.0,
    MTE3_BYTES_PER_CYCLE=256.0,
    MAC_THROUGHPUT_FP16=16*16*16*2,
    MAC_THROUGHPUT_FP8=16*32*16*2,
)

modeler = C1Modeler(..., hw_config=hw)
```

---

## 运行测试

```bash
python -m pytest tests/ -q
# 50 passed
```

---

## 硬件单元参考

| 单元 | 功能 | 带宽/吞吐 |
|------|------|-----------|
| MTE2 | DRAM/L2 → L1 | DRAM: 1600/32 GB/s；L2: 5400/32 GB/s |
| MTE1 | L1 → L0 | 256 bytes/cycle |
| MTE3 | UB → CUBE | 256 bytes/cycle |
| MAC | 矩阵乘法 | FP16: 16×16×16×2；FP8: 16×32×16×2 ops/cycle |
| FIXPIPE | L0C → UB | 128 bytes/cycle |
| VECTOR_V1 | Softmax 等 | 1600 cycles（固定） |
| VECTOR_V2 | 后处理 | 400 cycles（固定） |
