# C1V1C2V2 完整流水线建模

昇腾NPU Flash Attention完整流水线性能建模工具。

Flash Attention算子包含完整的四个阶段：
- **C1**: Q @ K^T → S (Attention Score) - 矩阵乘法
- **V1**: Softmax等向量操作 S → P (Attention Probability)
- **C2**: P @ V → O (Output) - 矩阵乘法
- **V2**: 后处理向量操作

当前实现包含C1V1C2V2四个阶段的完整建模。

## 项目结构

```
c1c2_modeling/
├── core/                   # 核心定义
│   ├── enums.py           # 枚举类型 (BoundType, DataType, LoadOrder)
│   ├── dataclasses.py     # 数据结构 (TimelineEvent)
│   └── hardware_config.py # 硬件配置参数
│
├── modelers/              # 建模器实现
│   ├── c1_modeler.py      # C1建模器
│   └── templates/         # 配置模板
│       ├── standard.py    # 标准配置
│       ├── dn_mode.py     # DN模式配置
│       └── full_load.py   # Full Load配置
│
├── utils/                 # 工具模块
│   ├── visualizer.py      # 时间线可视化
│   └── logger.py          # 日志配置
│
└── examples/              # 示例代码
    └── run_c1_examples.py # C1运行示例
```

## 快速开始

### 1. 使用配置模板

```python
from modelers import C1Modeler
from modelers.templates import StandardConfig, DNModeConfig, FullLoadConfig
from core import DataType

# 标准配置
config = StandardConfig(
    s1_total=256,
    s2_total=1024,
    d_total=128,
    data_type=DataType.FP16,
)
modeler = C1Modeler(**config.to_dict())
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "output.png")
```

### 2. 自定义配置

```python
from modelers import C1Modeler
from core import DataType

modeler = C1Modeler(
    s1_total=512,
    s2_total=2048,
    d_total=256,
    s1_base_size=128,
    s2_base_size=256,
    d_base_size=256,
    data_type=DataType.FP16,
    use_dn=True,
    L1_db=True,
    L0_db=True,
)
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
```

### 3. 运行示例

```bash
python examples/run_c1_examples.py
```

## 配置模板

### StandardConfig
标准配置，适用于基础场景。

### DNModeConfig
DN模式配置，L1A/L0A存储K矩阵，L1B/L0B存储Q矩阵。

### FullLoadConfig
Full Load配置，预加载所有Q块以提高性能。

## 参数说明

### 矩阵维度
- `s1_total`: Q矩阵的sequence长度
- `s2_total`: K矩阵的sequence长度
- `d_total`: 特征维度

### 块大小
- `s1_base_size`: Q块大小
- `s2_base_size`: K块大小
- `d_base_size`: D块大小

### 流水线配置
- `use_dn`: 是否使用DN模式
- `L1_db`: L1是否使用双缓冲
- `L0_db`: L0是否使用双缓冲
- `is_l2cache`: 是否使用L2缓存
- `full_load`: 是否全量加载Q矩阵
- `preload`: 预加载模式 (0=正常，1=先所有C1再所有C2)

## 性能模式

### Preload模式 (推荐)

启用 `preload=1` 可以显著提升性能：

```python
modeler = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    preload=1  # 先执行所有C1，再执行所有C2
)
```

**性能提升**:
- 正常模式 (preload=0): 32,991 cycles
- Preload模式 (preload=1): 21,471 cycles
- **提升**: 35%

**执行顺序**:
- preload=0: C1→V1→C2→V2→C1→V1→C2→V2...
- preload=1: C1→C1→C1→C1 → V1→V1→V1→V1 → C2→C2→C2→C2 → V2→V2→V2→V2

### 数据类型
- `DataType.FP16`: 半精度浮点
- `DataType.FP8`: 8位浮点

## 硬件配置

可以自定义硬件参数:

```python
from core import HardwareConfig

hw_config = HardwareConfig(
    CHIP_FREQ_GHZ=1.65,
    MTE2_DRAM_BW_GBPS=1600,
    MTE2_L2_BW_GBPS=5400,
    MTE1_FIXPIPE_BYTES_PER_CYCLE=256.0,
)

modeler = C1Modeler(..., hw_config=hw_config)
```

## 流水线阶段

### C1阶段 (第一次矩阵乘法)
- **MTE2**: 从DRAM/L2加载Q和K到L1
- **MTE1**: 从L1搬运Q和K到L0
- **MAC**: 执行矩阵乘法 Q @ K^T → P (标注: P11, P12...)
- **FIXPIPE**: 搬运P矩阵到UB

### V1阶段 (第一次向量操作)
- **VECTOR_V1**: Softmax等向量操作 (1600 cycles，标注: P11, P12...)
- 在FIXPIPE完成后执行

### 转换阶段
- **MTE3**: 从UB搬P回到CUBE (256B/cycle)
- **MTE2**: 并行加载V矩阵到L1 (DN模式路径改变)
- **MTE1**: 搬V到L0

### C2阶段 (第二次矩阵乘法)
- **MAC**: 执行矩阵乘法 P @ V → O (标注: O11, O12...)
- **FIXPIPE**: 搬运O矩阵到UB

### V2阶段 (第二次向量操作)
- **VECTOR_V2**: 后处理操作 (400 cycles，标注: O11, O12...)
- 在FIXPIPE完成后执行

## 输出

### 性能分析
- 总周期数
- 各单元利用率 (MTE2, MTE1, MTE3, MAC, FIXPIPE, VECTOR_V1, VECTOR_V2)
- 瓶颈分析

### 时间线图表
- 可视化完整流水线执行情况
- 显示L0/L1缓冲区使用
- 标注L2缓存命中
- P矩阵标注 (P11, P12, ...) - C1阶段和V1阶段
- O矩阵标注 (O11, O12, ...) - C2阶段和V2阶段
- VECTOR_V1和VECTOR_V2分别在不同高度显示

## 向后兼容

原始的 `single_mm_modeling.py` 文件保留在根目录，仍可直接运行:

```bash
python single_mm_modeling.py
```
