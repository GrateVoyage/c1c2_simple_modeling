# c1c2_modeling Project Memory
> 此文件是 Claude Code 的项目级 memory，跨会话持久保存开发进度和关键决策。

## Project Overview
Flash Attention C1V1C2V2 pipeline performance modeling tool for Ascend NPU.

## Key Architecture
- `core/enums.py` — BoundType, DataType, LoadOrder, InterCorePipeline, InnerCorePipeline
- `core/hardware_config.py` — HardwareConfig
- `modelers/c1_modeler.py` — Main C1Modeler class (~2000 lines)
- `examples/playground.py` — 推荐调试入口，所有参数可配
- `tests/` — 52 tests (all passing)

## Current C1Modeler API
```python
C1Modeler(
    s1_total, s2_total, d_total,
    s1_base_size, s2_base_size, d_base_size,
    baseM_C1=128, baseN_C1=128, baseK_C1=128,
    baseM_C2=128, baseN_C2=128, baseK_C2=128,
    q_data_type=DataType.FP16,
    kv_data_type=DataType.FP16,
    is_l2cache=True, use_dn=False,
    L1_db=True, L0_db=True, full_load=False,
    load_order=LoadOrder.LOAD_Q_FIRST,
    inter_core_pipeline=InterCorePipeline.DEFAULT,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)
```
- `Q_RESIDENT` 策略在 `__init__` 中自动强制 `full_load=True`

## Key Design Decisions
- **matmulFull**: single MTE2→MTE1→MAC→FIXPIPE
- **matmulK**: ONE MTE2 → per-sub-tile MTE1+MAC (L0C accumulate) → ONE FIXPIPE
- **matmulN**: ONE MTE2 → per-sub-tile MTE1+MAC (L0C diff columns) → ONE FIXPIPE
  - 两者均用 `sub_slot = i % len(l0_slots)` 实现 L0_db 流水
- **VECTOR_V1 和 VECTOR_V2 共用一个 `"VECTOR"` resource key** → 强制串行
- FIXPIPE-P = s1_base × s2_base × 4 bytes (FP32)
- FIXPIPE-O = s1_base × d_base × 4 bytes (FP32)
- MTE3-P = s1_base × s2_base × kv_elem_size
- MAC C1: q 和 kv 均为 FP8 才用 FP8 吞吐; MAC C2: kv 决定
- 切分优先级: N → K → Full（C1/C2 相同）

## Visualizer
- `matplotlib.use('Agg')` at top (WSL 无显示器环境必须)
- Y轴标签: VECTOR_V2 / VECTOR_V1 / MTE3 / FIXPIPE / MAC / MTE1(L0B) / MTE1(L0A) / MTE2 / MTE2
- 标注逻辑: substring matching; 阈值 `total_cycles * 0.003`

## Test Run
```
cd /mnt/g/WSL/chuqihang/model/c1c2_modeling && python -m pytest tests/ -q
```
52 tests, ~0.7s.
