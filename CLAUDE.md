# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

昇腾NPU Flash Attention 算子完整流水线性能**建模仿真工具**（非真实执行），对 C1→V1→C2→V2 四个阶段进行周期级时间线建模：

| 阶段 | 操作 | 说明 |
|------|------|------|
| C1 | Q @ K^T → P | 第一次矩阵乘（Attention Score） |
| V1 | Softmax(P) | 向量操作（VECTOR_V1） |
| C2 | P @ V → O | 第二次矩阵乘（Output） |
| V2 | 后处理(O) | 向量操作（VECTOR_V2） |

> VECTOR_V1 和 VECTOR_V2 共用同一向量单元（resource key `"VECTOR"`），在同一基本块内**串行**执行。

---

## Commands

```bash
# 安装依赖
pip install matplotlib numpy

# 运行所有测试
python -m pytest tests/ -q

# 运行单个测试文件
python -m pytest tests/test_c1_modeler_basic.py -v

# 运行单个测试函数
python -m pytest tests/test_c1_modeler_basic.py::test_instantiation_new_api -v

# 按关键字过滤测试
python -m pytest tests/ -k "fp8" -v

# 参数调试（推荐入口，编辑后直接运行，输出图片到 outputs/）
python examples/playground.py

# 核间流水模式对比
python examples/test_preload.py
```

---

## Architecture

### 核心层（`core/`）

- `enums.py` — 所有枚举：`BoundType`, `DataType`, `LoadOrder`, `InterCorePipeline`（`DEFAULT`/`PRELOAD_1`/`PRELOAD_2`）, `InnerCorePipeline`
- `dataclasses.py` — `TimelineEvent`（时间线事件数据类）
- `hardware_config.py` — `HardwareConfig`（芯片频率、各单元带宽/吞吐，可覆盖默认值）

### 建模器（`modelers/`）

- `c1_modeler.py` — **主类 `C1Modeler`**，包含：
  - `L0CSlotTracker`：L0C doublebuffer 槽位追踪（2槽），MAC 前 `allocate()`，FIXPIPE 后 `release()`；matmulN/K 多 sub-MAC 共用一槽
  - `UBWorkspaceTracker`：UB Workspace 追踪，PRELOAD_1=2WS，PRELOAD_2=3WS；`allocate()` 返回值作为 MTE2 开始时间下界
  - `L1CacheTracker`：L1 缓存 LRU 追踪，支持 `DEFAULT`（平坦 LRU）和 `Q_RESIDENT`（固定分区）两种策略
  - 仿真调度：`resource_free_time` dict 追踪各硬件单元空闲时间
  - L1槽位状态：`{'mte2_free': float, 'mte1_free': float}`，MTE2 和 MTE1 **可并行**（各自独立等前序完成）
  - `run_simulation()` → `(timeline, bound_type, unit_times, total_cycles)`
- `templates/` — 开箱即用配置：`StandardConfig`, `DNModeConfig`, `FullLoadConfig`

### 关键设计决策

**矩阵乘切分**（优先级：N → K → Full）：
- `matmulFull`：单次 MTE2→MTE1→MAC→FIXPIPE
- `matmulN`：沿 N 轴切分，多 sub-tile MAC 写 L0C 不同列，最后**一次** FIXPIPE
- `matmulK`：沿 K 轴切分，多 sub-tile MAC 在 L0C 累加，最后**一次** FIXPIPE
- 两种切分均用 `sub_slot = i % len(l0_slots)` 实现 `L0_db` 流水

**数据大小固定规则**：
- FIXPIPE-P 输出：`s1_base × s2_base × 4 bytes`（恒为 FP32）
- FIXPIPE-O 输出：`s1_base × d_base × 4 bytes`（恒为 FP32）
- MTE3-P：`s1_base × s2_base × kv_elem_size`
- MAC C1 吞吐：q 和 kv **均为 FP8** 才用 FP8 吞吐；MAC C2：由 kv 单独决定

**`Q_RESIDENT` 策略**：`__init__` 中自动强制 `full_load=True`

**可视化**：`utils/visualizer.py` 顶部使用 `matplotlib.use('Agg')`（无显示器环境必须），输出 PNG 到 `outputs/`

---

## Code Style

- 注释使用**中文**，类和重要方法提供文档字符串
- 函数参数和返回值需类型注解，使用 `typing` 模块（`List`, `Dict`, `Tuple`, `Optional`）
- 命名：类 PascalCase，函数/变量 snake_case，常量/枚举成员 UPPER_CASE
- 本地导入使用非相对导入（`from core import ...`，项目根目录需在 `sys.path`）

---

## Core Principles

1. **遇问题先探索代码库，不下意识简化代码**——定位问题点后只修复该部分
2. **充分了解后再决策**——不轻易下结论，不直接开始实现
3. **禁止降级简化**——不能因"能跑"就降低优化标准
