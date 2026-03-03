# AGENTS.md

本助手是 C1V1C2V2 流水线建模工具的智能体，用于昇腾NPU Flash Attention算子性能建模。

## 项目概述

本项目是昇腾NPU Flash Attention算子完整流水线性能建模工具，用于建模和分析四个阶段（C1→V1→C2→V2）的执行性能。

### 核心功能

- 使用 Python 建模昇腾NPU Flash Attention 算子流水线性能
- 支持多种流水线（核间、核内）和配置模式
- 提供时间线可视化和性能瓶颈分析

---

## 开发技能系统

本项目使用模块化技能系统（Skills），按需加载相关知识，提高效率。

### 技能列表

| 技能 | 触发时机 | 说明 |
|-----|---------|------|
| **ascendc-docs-search** | 需要查找开发资源时 | 本地资源索引 + 在线文档搜索（本地优先） |
| **ascendc-kernel-develop-workflow** | 接到算子开发任务 | 工作流程、需求检查、Plan 模式 |

---

## 核心原则

> 严格遵循以下原则，可避免 95% 的开发问题

1. **遇问题先探索，不简化代码**
    - 搜索代码库中的相关实现和文档
    - 综合审视代码，分析根本原因
    - 定位问题点后修复该部分
    - 禁止：下意识简化代码、凭直觉实现、遇到错误就推翻重写

2. **充分了解后再决策**
   - 查阅资料、搜索代码、理解原理
   - 不要轻易下结论或直接开始实现

3. **持续探索更优方案**
    - 方案走通后，继续搜索是否有更优实现

4. **环境兼容性验证**
   - 确认 API/方法适用于当前环境

5. **禁止降级简化**
   - 不能因"能跑"就降低优化标准
   - 不能简化必要的优化

---

## 项目目录结构

```
c1c2_simple_modeling/
├── core/                        # 核心定义
│   ├── enums.py                # 枚举类型（BoundType, DataType, LoadOrder等）
│   ├── dataclasses.py          # 数据结构（TimelineEvent）
│   └── hardware_config.py      # 硬件参数配置（HardwareConfig）
│
├── modelers/                    # 建模器
│   ├── c1_modeler.py           # C1Modeler 主类
│   └── templates/              # 配置模板
│       ├── standard.py         # 标准配置
│       ├── dn_mode.py          # DN模式配置
│       └── full_load.py        # Full Load配置
│
├── utils/                       # 工具
│   ├── visualizer.py           # 时间线可视化（matplotlib）
│   └── logger.py               # 日志
│
├── examples/                    # 示例
│   ├── playground.py           # 推荐：单一可配参数调试入口
│   ├── run_c1_examples.py      # 各配置示例
│   ├── test_full_pipeline.py   # 完整流水线演示
│   └── test_preload.py         # 核间流水模式对比
│
├── tests/                       # 测试（52个）
├── outputs/                     # 生成的时间线图片
├── reference.md                 # 硬件规格参考文档
└── CLAUDE_MEMORY.md             # AI 助手开发记录
```

---

## 构建和测试命令

### 安装依赖

```bash
pip install matplotlib matplotlib numpy
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -q

# 运行单个测试文件
python -m pytest tests/test_c1_modeler_basic.py -v

# 运行单个测试函数
python -m pytest tests/test_c1_modeler_basic.py::test_instantiation_new_api -v

# 运行特定模式的测试
python -m pytest tests/ -k "fp8" -v

# 详细输出
python -m pytest tests/ -v
```

### 运行示例

```bash
# 参数调试（推荐）
python examples/playground.py

# 完整流水线演示
python examples/test_full_pipeline.py

# 核间流水模式对比
python examples/test_preload.py
```

---

## 代码风格指南

### 命名约定

- **类名**: PascalCase（如 `C1Modeler`, `HardwareConfig`, `TimelineEvent`）
- **函数/方法**: snake_case（如 `run_simulation`, `get_mac_throughput`）
- **变量**: snake_case（如 `s1_total`, `baseM_C1`）
- **常量**: UPPER_CASE（如 `MTE2_DRAM_BW_GBPS`, `MAC_THROUGHPUT_FP16`）
- **枚举成员**: UPPER_CASE（如 `DataType.FP16`, `InterCorePipeline.DEFAULT`）

### 导入顺序

```python
# 1. 标准库
import sys
import math
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

# 2. 第三方库
import matplotlib
import matplotlib.pyplot as plt

# 3. 本地模块（使用相对导入）
from core import BoundType, DataType, LoadOrder
from core.dataclasses import TimelineEvent
from utils.visualizer import TimelineVisualizer
```

### 类型注解

- 所有函数参数和返回值都应使用类型注解
- 使用 `typing` 模块中的类型（`List`, `Dict`, `Tuple`, `Optional`）

```python
def run_simulation(self) -> Tuple[List[TimelineEvent], BoundType, Dict, float]:
    """运行仿真，返回时间线、瓶颈类型、单元耗时和总周期"""
    pass

def _get_q_element_size(self) -> int:
    """获取Q矩阵元素大小（字节）"""
    return 2 if self.q_data_type == DataType.FP16 else 1
```

### 数据类使用

- 使用 `@dataclass` 装饰器定义配置和数据结构
- 提供清晰的文档字符串

```python
@dataclass
class HardwareConfig:
    """昇腾NPU硬件配置"""

    CHIP_FREQ_GHZ: float = 1.65
    MTE2_DRAM_BW_GBPS: float = 1600
    L1_CAPACITY: int = 512 * 1024

    def get_mac_throughput(self, data_type: DataType) -> int:
        """获取MAC吞吐量"""
        return self.MAC_THROUGHPUT_FP16 if data_type == DataType.FP16 else self.MAC_THROUGHPUT_FP8
```

### 枚举使用

- 使用 `Enum` 定义固定类型
- 提供清晰的文档字符串

```python
class DataType(Enum):
    """数据类型"""
    FP16 = "fp16"
    FP8 = "fp8"

class InterCorePipeline(Enum):
    """核间流水线模式"""
    DEFAULT  = "default"    # 顺序流水
    PRELOAD  = "preload"    # 渐进式
    N_BUFFER = "n_buffer"   # N=2批次
```

### 注释和文档

- 使用中文注释说明算法逻辑和硬件细节
- 类和重要方法提供文档字符串
- 复杂计算添加行内注释

```python
def _flat_store(self, block_id: str, size: int, ready_time: float):
    """存储块到flat LRU缓存"""
    if block_id in self.flat:
        self.lru_counter += 1
        self.flat[block_id] = (self.flat[block_id][0], ready_time, self.lru_counter)
        return
    # 驱逐LRU块，为新块腾出空间
    while self._flat_used() + size > self.capacity and self.flat:
        lru_id = min(self.flat, key=lambda k: self.flat[k][2])
        del self.flat[lru_id]
```

### 错误处理

- 使用 Python 标准异常
- 提供清晰的错误信息

```python
if not (s1_total % s1_base_size == 0):
    raise ValueError(f"s1_total ({s1_total}) must be divisible by s1_base_size ({s1_base_size})")
```

---

## 测试指南

### 测试文件命名

- 测试文件命名为 `test_*.py`
- 测试函数命名为 `test_*`

### 测试结构

```python
import pytest
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(**kwargs):
    """创建模型器的辅助函数"""
    defaults = dict(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        # ... 其他默认参数
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_instantiation_new_api():
    """测试新API实例化"""
    m = make_modeler()
    assert m.q_data_type == DataType.FP16
    assert m.inter_core_pipeline == InterCorePipeline.DEFAULT

def test_q_element_size_fp16():
    """测试FP16类型Q元素大小"""
    m = make_modeler(q_data_type=DataType.FP16)
    assert m._get_q_element_size() == 2
```

### 使用 conftest.py

```python
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
```

---

## 硬件参数参考

| 单元 | 功能 | 带宽/吞吐 |
|------|------|-----------|
| MTE2 | DRAM/L2 → L1 | DRAM: 1600 GB/s；L2: 5400 GB/s |
| MTE1 | L1 → L0 | 256 bytes/cycle |
| MTE3 | UB → L1 | 256 bytes/cycle |
| MAC | 矩阵乘法 | FP16: 16×16×16×2；FP8: 16×32×16×2 ops/cycle |
| FIXPIPE | L0C → UB | 128 bytes/cycle |
| VECTOR_V1 | Softmax 等 | 1600 cycles（固定） |
| VECTOR_V2 | 后处理 | 400 cycles（固定） |

> VECTOR_V1 与 VECTOR_V2 共用同一向量单元，在同一基本块内串行执行。

---

## 开发资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| 核心定义 | `core/` | 枚举、数据类、硬件配置 |
| 主建模器 | `modelers/c1_modeler.py` | C1Modeler 主类 |
| 配置模板 | `modelers/templates/` | 标准配置、DN模式、Full Load |
| 可视化工具 | `utils/visualizer.py` | 时间线可视化 |
| 测试 | `tests/` | 52个测试用例 |
| 硬件规格 | `reference.md` | 硬件规格参考文档 |
