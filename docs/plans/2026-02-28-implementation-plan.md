# C1C2建模器完善 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 按照 reference.md 完整实现 InterCorePipeline/InnerCorePipeline 枚举、N-Buffer 流水、矩阵乘切分参数、q/kv 数据类型分离，重构 C1Modeler API。

**Architecture:** 干净重构 C1Modeler，替换旧参数（preload→InterCorePipeline，data_type→q_data_type+kv_data_type），新增 baseM/N/K 切分参数；枚举定义在 core/enums.py；所有改动同步更新 examples 和 templates。

**Tech Stack:** Python 3.x，pytest，matplotlib（已有依赖）

**设计文档:** `docs/plans/2026-02-28-c1c2-improvements-design.md`

---

## 前置准备

### 安装 pytest（如未安装）

```bash
pip install pytest
```

### 创建 tests/ 目录和 conftest.py

**文件:** 新建 `tests/conftest.py`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

验证：
```bash
cd /mnt/g/WSL/chuqihang/model/c1c2_modeling
python -m pytest tests/ -v --tb=short
```
预期：`no tests ran`（目录存在，conftest 加载正常）

---

## Task 1: 添加 InterCorePipeline 和 InnerCorePipeline 枚举

**Files:**
- Modify: `core/enums.py`
- Modify: `core/__init__.py`
- Create: `tests/test_enums.py`

**Step 1: 写失败测试**

```python
# tests/test_enums.py
import pytest
from core import InterCorePipeline, InnerCorePipeline

def test_inter_core_pipeline_values():
    assert InterCorePipeline.DEFAULT.value == "default"
    assert InterCorePipeline.PRELOAD.value == "preload"
    assert InterCorePipeline.N_BUFFER.value == "n_buffer"

def test_inner_core_pipeline_values():
    assert InnerCorePipeline.DEFAULT.value == "default"

def test_all_inter_core_pipeline_members():
    members = {e.value for e in InterCorePipeline}
    assert members == {"default", "preload", "n_buffer"}
```

**Step 2: 运行确认失败**

```bash
python -m pytest tests/test_enums.py -v
```
预期：`ImportError: cannot import name 'InterCorePipeline'`

**Step 3: 在 core/enums.py 末尾追加两个枚举**

```python
class InterCorePipeline(Enum):
    """核间流水线模式"""
    DEFAULT  = "default"    # 顺序流水: C1V1C2V2 → C1V1C2V2 → ...
    PRELOAD  = "preload"    # 渐进式:   V在K加载后立即预加载，C2省去V的MTE2等待
    N_BUFFER = "n_buffer"   # N=2批次:  C1C1→V1V1→C2C2→V2V2

class InnerCorePipeline(Enum):
    """核内流水线模式"""
    DEFAULT = "default"     # Q常驻L1，KP共用2块144K，V使用2块32K
```

**Step 4: 更新 core/__init__.py**

```python
from .enums import BoundType, DataType, LoadOrder, InterCorePipeline, InnerCorePipeline
from .dataclasses import TimelineEvent
from .hardware_config import HardwareConfig

__all__ = [
    'BoundType', 'DataType', 'LoadOrder',
    'InterCorePipeline', 'InnerCorePipeline',
    'TimelineEvent', 'HardwareConfig',
]
```

**Step 5: 运行确认通过**

```bash
python -m pytest tests/test_enums.py -v
```
预期：3 passed

**Step 6: Commit**

```bash
git add core/enums.py core/__init__.py tests/conftest.py tests/test_enums.py
git commit -m "feat: add InterCorePipeline and InnerCorePipeline enums"
```

---

## Task 2: 扩展 HardwareConfig — 分离 C1/C2 MAC 吞吐量查询

**Files:**
- Modify: `core/hardware_config.py`
- Create: `tests/test_hardware_config.py`

**Step 1: 写失败测试**

```python
# tests/test_hardware_config.py
from core import HardwareConfig, DataType

hw = HardwareConfig()

def test_mac_throughput_c1_both_fp16():
    """两者都是FP16 → FP16吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP16, DataType.FP16) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c1_both_fp8():
    """两者都是FP8 → FP8吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP8, DataType.FP8) == hw.MAC_THROUGHPUT_FP8

def test_mac_throughput_c1_mixed():
    """一个FP16一个FP8 → 仍为FP16吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP16, DataType.FP8) == hw.MAC_THROUGHPUT_FP16
    assert hw.get_mac_throughput_c1(DataType.FP8, DataType.FP16) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c2_fp16():
    assert hw.get_mac_throughput_c2(DataType.FP16) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c2_fp8():
    assert hw.get_mac_throughput_c2(DataType.FP8) == hw.MAC_THROUGHPUT_FP8
```

**Step 2: 运行确认失败**

```bash
python -m pytest tests/test_hardware_config.py -v
```
预期：`AttributeError: 'HardwareConfig' object has no attribute 'get_mac_throughput_c1'`

**Step 3: 在 core/hardware_config.py 中新增两个方法**（在现有 `get_mac_throughput` 方法后追加）

```python
def get_mac_throughput_c1(self, q_dtype: DataType, kv_dtype: DataType) -> int:
    """C1 MAC吞吐量：Q和KV都为FP8时用FP8，否则用FP16"""
    if q_dtype == DataType.FP8 and kv_dtype == DataType.FP8:
        return self.MAC_THROUGHPUT_FP8
    return self.MAC_THROUGHPUT_FP16

def get_mac_throughput_c2(self, kv_dtype: DataType) -> int:
    """C2 MAC吞吐量：由kv_data_type决定"""
    return self.MAC_THROUGHPUT_FP8 if kv_dtype == DataType.FP8 else self.MAC_THROUGHPUT_FP16
```

**Step 4: 运行确认通过**

```bash
python -m pytest tests/test_hardware_config.py -v
```
预期：5 passed

**Step 5: Commit**

```bash
git add core/hardware_config.py tests/test_hardware_config.py
git commit -m "feat: add get_mac_throughput_c1/c2 with mixed-type support"
```

---

## Task 3: 重构 C1Modeler 参数签名 + 数据类型分离

**Files:**
- Modify: `modelers/c1_modeler.py`（仅 `__init__` 和辅助计算方法）
- Create: `tests/test_c1_modeler_basic.py`

**Step 1: 写失败测试**

```python
# tests/test_c1_modeler_basic.py
import pytest
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(**kwargs):
    defaults = dict(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_instantiation_new_api():
    m = make_modeler()
    assert m.q_data_type == DataType.FP16
    assert m.kv_data_type == DataType.FP16
    assert m.inter_core_pipeline == InterCorePipeline.DEFAULT
    assert m.inner_core_pipeline == InnerCorePipeline.DEFAULT

def test_instantiation_fp8():
    m = make_modeler(q_data_type=DataType.FP8, kv_data_type=DataType.FP8)
    assert m.q_data_type == DataType.FP8

def test_q_element_size_fp16():
    m = make_modeler(q_data_type=DataType.FP16)
    assert m._get_q_element_size() == 2

def test_q_element_size_fp8():
    m = make_modeler(q_data_type=DataType.FP8)
    assert m._get_q_element_size() == 1

def test_kv_element_size_fp16():
    m = make_modeler(kv_data_type=DataType.FP16)
    assert m._get_kv_element_size() == 2

def test_kv_element_size_fp8():
    m = make_modeler(kv_data_type=DataType.FP8)
    assert m._get_kv_element_size() == 1
```

**Step 2: 运行确认失败**

```bash
python -m pytest tests/test_c1_modeler_basic.py -v
```
预期：`TypeError: __init__() got an unexpected keyword argument 'q_data_type'`

**Step 3: 重构 C1Modeler.__init__**

将 `modelers/c1_modeler.py` 的 `__init__` 参数替换为新 API。关键变更：

- 删除 `data_type: DataType = DataType.FP16`
- 删除 `preload: int = 0`
- 删除 `two_buffer: bool = False`（合并进 PRELOAD 模式的已有逻辑，two_buffer 保留但由 N_BUFFER 场景控制）
- 新增：
  ```python
  q_data_type: DataType = DataType.FP16,
  kv_data_type: DataType = DataType.FP16,
  baseM_C1: int = 128,
  baseN_C1: int = 128,
  baseK_C1: int = 128,
  baseM_C2: int = 128,
  baseN_C2: int = 128,
  baseK_C2: int = 128,
  inter_core_pipeline: InterCorePipeline = InterCorePipeline.DEFAULT,
  inner_core_pipeline: InnerCorePipeline = InnerCorePipeline.DEFAULT,
  ```
- `self.q_data_type = q_data_type`
- `self.kv_data_type = kv_data_type`
- `self.inter_core_pipeline = inter_core_pipeline`
- `self.inner_core_pipeline = inner_core_pipeline`
- `self.baseM_C1, self.baseN_C1, self.baseK_C1 = baseM_C1, baseN_C1, baseK_C1`
- `self.baseM_C2, self.baseN_C2, self.baseK_C2 = baseM_C2, baseN_C2, baseK_C2`

将 `_get_element_size()` 拆分为：

```python
def _get_q_element_size(self) -> int:
    return 2 if self.q_data_type == DataType.FP16 else 1

def _get_kv_element_size(self) -> int:
    return 2 if self.kv_data_type == DataType.FP16 else 1
```

同时更新 `run_simulation` 中的打印输出和分支判断，将 `self.preload` 替换为 `self.inter_core_pipeline`。

**Step 4: 运行确认通过**

```bash
python -m pytest tests/test_c1_modeler_basic.py -v
```
预期：6 passed

**Step 5: Commit**

```bash
git add modelers/c1_modeler.py tests/test_c1_modeler_basic.py
git commit -m "refactor: replace data_type/preload with q/kv_data_type and InterCorePipeline enum"
```

---

## Task 4: 更新所有数据大小计算 — Q/KV 分离 + FIXPIPE 改 FP32

**Files:**
- Modify: `modelers/c1_modeler.py`（所有 `_get_element_size()` 调用处）
- Create: `tests/test_c1_data_sizes.py`

**Step 1: 写失败测试**

```python
# tests/test_c1_data_sizes.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(**kwargs):
    defaults = dict(
        s1_total=128, s2_total=256, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_fixpipe_p_is_fp32():
    """FIXPIPE P 大小 = s1_base × s2_base × 4 (FP32)，与数据类型无关"""
    m = make_modeler(kv_data_type=DataType.FP16)
    # FP16 模式下，FIXPIPE P = 128 * 256 * 4 = 131072 bytes
    expected = 128 * 256 * 4
    assert m._calc_fixpipe_p_size() == expected

def test_fixpipe_o_is_fp32():
    """FIXPIPE O 大小 = s1_base × d_base × 4 (FP32)"""
    m = make_modeler()
    expected = 128 * 128 * 4
    assert m._calc_fixpipe_o_size() == expected

def test_mte3_p_uses_kv_dtype():
    """MTE3 P 大小 = s1_base × s2_base × kv_elem_size"""
    m_fp16 = make_modeler(kv_data_type=DataType.FP16)
    m_fp8  = make_modeler(kv_data_type=DataType.FP8)
    assert m_fp16._calc_mte3_p_size() == 128 * 256 * 2
    assert m_fp8._calc_mte3_p_size()  == 128 * 256 * 1

def test_q_mte_uses_q_dtype():
    """Q 搬运大小 = s1_base × d_base × q_elem_size"""
    m_fp16 = make_modeler(q_data_type=DataType.FP16)
    m_fp8  = make_modeler(q_data_type=DataType.FP8)
    assert m_fp16._calc_q_size() == 128 * 128 * 2
    assert m_fp8._calc_q_size()  == 128 * 128 * 1

def test_k_mte_uses_kv_dtype():
    """K 搬运大小 = s2_base × d_base × kv_elem_size"""
    m_fp16 = make_modeler(kv_data_type=DataType.FP16)
    m_fp8  = make_modeler(kv_data_type=DataType.FP8)
    assert m_fp16._calc_k_size() == 256 * 128 * 2
    assert m_fp8._calc_k_size()  == 256 * 128 * 1
```

**Step 2: 运行确认失败**

```bash
python -m pytest tests/test_c1_data_sizes.py -v
```

**Step 3: 在 C1Modeler 中新增辅助方法并更新所有计算处**

新增 6 个辅助方法：

```python
def _calc_q_size(self) -> int:
    return self.s1_base * self.d_base * self._get_q_element_size()

def _calc_k_size(self) -> int:
    return self.s2_base * self.d_base * self._get_kv_element_size()

def _calc_v_size(self) -> int:
    return self.s2_base * self.d_base * self._get_kv_element_size()

def _calc_fixpipe_p_size(self) -> int:
    """FIXPIPE P: L0C→UB，L0C 输出恒为 FP32"""
    return self.s1_base * self.s2_base * 4

def _calc_fixpipe_o_size(self) -> int:
    """FIXPIPE O: L0C→UB，L0C 输出恒为 FP32"""
    return self.s1_base * self.d_base * 4

def _calc_mte3_p_size(self) -> int:
    """MTE3 P: UB→CUBE，P 经 V1 后为 kv_data_type 格式"""
    return self.s1_base * self.s2_base * self._get_kv_element_size()
```

然后在全文中将所有旧的 `_get_element_size()` 调用替换为对应的新方法：
- Q 相关 size → `_calc_q_size()`
- K 相关 size → `_calc_k_size()`
- V 相关 size → `_calc_v_size()`
- FIXPIPE P size → `_calc_fixpipe_p_size()`
- FIXPIPE O size → `_calc_fixpipe_o_size()`
- MTE3 P size → `_calc_mte3_p_size()`

同时将 `_calc_mac_cycles` 中的 `self.hw_config.get_mac_throughput(self.data_type)` 拆分为：
- C1 MAC: `self.hw_config.get_mac_throughput_c1(self.q_data_type, self.kv_data_type)`
- C2 MAC: `self.hw_config.get_mac_throughput_c2(self.kv_data_type)`

**Step 4: 运行确认通过**

```bash
python -m pytest tests/test_c1_data_sizes.py -v
```
预期：5 passed

**Step 5: 运行所有测试确保无回归**

```bash
python -m pytest tests/ -v
```

**Step 6: Commit**

```bash
git add modelers/c1_modeler.py tests/test_c1_data_sizes.py
git commit -m "feat: separate q/kv data types, fix FIXPIPE to use FP32 size"
```

---

## Task 5: 实现 N_BUFFER 流水模式（C1C1→V1V1→C2C2→V2V2）

**Files:**
- Modify: `modelers/c1_modeler.py`（`run_simulation` + 新增 `_process_n_buffer_group`）
- Create: `tests/test_n_buffer.py`

**Step 1: 写失败测试**

```python
# tests/test_n_buffer.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(pipeline):
    return C1Modeler(
        s1_total=128, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=pipeline,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
    )

def test_n_buffer_runs_without_error():
    m = make_modeler(InterCorePipeline.N_BUFFER)
    timeline, bound_type, unit_times, total_cycles = m.run_simulation()
    assert total_cycles > 0
    assert len(timeline) > 0

def test_n_buffer_has_all_units():
    m = make_modeler(InterCorePipeline.N_BUFFER)
    timeline, _, unit_times, _ = m.run_simulation()
    units = {e.unit for e in timeline}
    assert "MAC" in units
    assert "VECTOR_V1" in units
    assert "VECTOR_V2" in units
    assert "MTE2" in units

def test_n_buffer_v1_overlaps_with_c1():
    """
    N_BUFFER 的关键：VECTOR_V1[k0] 应与 MAC C1[k1] 有时间重叠。
    即 VECTOR_V1[k0].start_time < MAC_C1[k1].end_time
    """
    m = make_modeler(InterCorePipeline.N_BUFFER)
    timeline, _, _, _ = m.run_simulation()

    mac_c1_events = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    v1_events = [e for e in timeline if e.unit == "VECTOR_V1"]

    if len(mac_c1_events) >= 2 and len(v1_events) >= 1:
        # V1[k0] 开始时间 < MAC_C1[k1] 结束时间 → 存在重叠
        v1_k0_start = v1_events[0].start_time
        mac_c1_k1_end = mac_c1_events[1].end_time
        assert v1_k0_start < mac_c1_k1_end, (
            f"V1[k0] should overlap with MAC C1[k1]: "
            f"V1 start={v1_k0_start:.1f}, MAC C1[k1] end={mac_c1_k1_end:.1f}"
        )
```

**Step 2: 运行确认失败**

```bash
python -m pytest tests/test_n_buffer.py -v
```
预期：`AssertionError` 或 N_BUFFER 走到未实现的分支报错

**Step 3: 在 run_simulation 中添加 N_BUFFER 分支**

```python
elif self.inter_core_pipeline == InterCorePipeline.N_BUFFER:
    n_size = 2
    for q_idx in range(self.q_block_count):
        self._load_q_block(q_idx, map_q, resource_free_time,
                           q_l1_ready_times, q_l0_ready_times)
        for group_start in range(0, self.k_block_count, n_size):
            group = list(range(group_start,
                               min(group_start + n_size, self.k_block_count)))
            self._process_n_buffer_group(
                q_idx, group, map_k, resource_free_time, q_l0_ready_times
            )
```

**Step 4: 实现 `_process_n_buffer_group`**

```python
def _process_n_buffer_group(
    self, q_idx: int, k_group: list, map_k: Dict,
    resource_free_time: Dict, q_l0_ready_times: Dict
):
    """
    N_BUFFER 组内按阶段批次执行：
    Phase1: 组内所有 k 的 C1（K-load + MTE1 + MAC + FIXPIPE）
    Phase2: 组内所有 k 的 V1（VECTOR_V1）
    Phase3: 组内所有 k 的 C2（V-load + MTE3 + MTE1 + MAC C2 + FIXPIPE）
    Phase4: 组内所有 k 的 V2（VECTOR_V2）

    V1[k_i] 和 MAC C1[k_{i+1}] 使用不同硬件，resource_free_time 自然处理重叠。
    """
    l1_name_k = map_k['l1']
    l0_name_k = map_k['l0']
    l1_slots_k = map_k['l1_slots']
    l0_slots_k = map_k['l0_slots']

    fixpipe_p_ready = {}   # k_idx → FIXPIPE P 完成时间
    v1_ready = {}          # k_idx → V1 完成时间

    # ── Phase 1: C1 ──────────────────────────────────────────────────
    for k_idx in k_group:
        use_l2_for_k = self.is_l2cache and (q_idx > 0)
        size_k = self._calc_k_size()
        dur_k_l1 = self._calc_mte2_cycles(size_k, use_l2=use_l2_for_k)
        slot_l1_k = k_idx % len(l1_slots_k)

        start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
        end_l1_k = start_l1_k + dur_k_l1
        self.timeline.append(TimelineEvent(
            "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
            start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
            q_idx, k_idx, is_l2_hit=use_l2_for_k
        ))
        resource_free_time["MTE2"] = end_l1_k

        dur_k_l0 = self._calc_mte1_cycles(size_k)
        slot_l0_k = k_idx % len(l0_slots_k)
        start_l0_k = max(resource_free_time["MTE1"], end_l1_k, l0_slots_k[slot_l0_k])
        end_l0_k = start_l0_k + dur_k_l0
        self.timeline.append(TimelineEvent(
            "MTE1", f"Load {l0_name_k} (K{k_idx+1})",
            start_l0_k, end_l0_k, dur_k_l0, l0_name_k, q_idx, k_idx
        ))
        resource_free_time["MTE1"] = end_l0_k
        l1_slots_k[slot_l1_k] = end_l0_k

        dur_mac_c1 = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, self.d_base)
        start_mac_c1 = max(resource_free_time["MAC"], q_l0_ready_times[q_idx], end_l0_k)
        end_mac_c1 = start_mac_c1 + dur_mac_c1
        self.timeline.append(TimelineEvent(
            "MAC", "P", start_mac_c1, end_mac_c1, dur_mac_c1, "L0C", q_idx, k_idx
        ))
        resource_free_time["MAC"] = end_mac_c1
        l0_slots_k[slot_l0_k] = end_mac_c1

        size_fp = self._calc_fixpipe_p_size()
        dur_fix_p = self._calc_fixpipe_cycles(size_fp)
        start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_c1)
        end_fix_p = start_fix_p + dur_fix_p
        self.timeline.append(TimelineEvent(
            "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p, "UB", q_idx, k_idx
        ))
        resource_free_time["FIXPIPE"] = end_fix_p
        fixpipe_p_ready[k_idx] = end_fix_p

    # ── Phase 2: V1 ──────────────────────────────────────────────────
    for k_idx in k_group:
        dur_v1 = self._calc_vector_v1_cycles()
        start_v1 = max(resource_free_time["VECTOR_V1"], fixpipe_p_ready[k_idx])
        end_v1 = start_v1 + dur_v1
        self.timeline.append(TimelineEvent(
            "VECTOR_V1", "P", start_v1, end_v1, dur_v1, "CUBE", q_idx, k_idx
        ))
        resource_free_time["VECTOR_V1"] = end_v1
        v1_ready[k_idx] = end_v1

    # ── Phase 3: C2 ──────────────────────────────────────────────────
    if self.use_dn:
        l1_name_v, l0_name_v = "L1A", "L0A"
        l1_slots_v, l0_slots_v = map_k['l1_slots'], map_k['l0_slots']
    else:
        l1_name_v, l0_name_v = "L1B", "L0B"
        l1_slots_v, l0_slots_v = map_k['l1_slots'], map_k['l0_slots']

    fixpipe_o_ready = {}

    for k_idx in k_group:
        p_ready = v1_ready[k_idx]
        size_p_mte3 = self._calc_mte3_p_size()
        dur_mte3 = self._calc_mte3_cycles(size_p_mte3)
        start_mte3 = max(resource_free_time["MTE3"], p_ready)
        end_mte3 = start_mte3 + dur_mte3
        self.timeline.append(TimelineEvent(
            "MTE3", "P", start_mte3, end_mte3, dur_mte3, "CUBE", q_idx, k_idx
        ))
        resource_free_time["MTE3"] = end_mte3

        size_v = self._calc_v_size()
        dur_v_l1 = self._calc_mte2_cycles(size_v, use_l2=False)
        slot_l1_v = k_idx % len(l1_slots_v)
        start_l1_v = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], p_ready)
        end_l1_v = start_l1_v + dur_v_l1
        self.timeline.append(TimelineEvent(
            "MTE2", f"Load {l1_name_v} (V{k_idx+1})",
            start_l1_v, end_l1_v, dur_v_l1, l1_name_v, q_idx, k_idx
        ))
        resource_free_time["MTE2"] = end_l1_v

        dur_v_l0 = self._calc_mte1_cycles(size_v)
        slot_l0_v = k_idx % len(l0_slots_v)
        start_l0_v = max(resource_free_time["MTE1"], end_l1_v, l0_slots_v[slot_l0_v])
        end_l0_v = start_l0_v + dur_v_l0
        self.timeline.append(TimelineEvent(
            "MTE1", f"Load {l0_name_v} (V{k_idx+1})",
            start_l0_v, end_l0_v, dur_v_l0, l0_name_v, q_idx, k_idx
        ))
        resource_free_time["MTE1"] = end_l0_v
        l1_slots_v[slot_l1_v] = end_l0_v

        dur_mac_c2 = self._calc_mac_cycles_c2(self.s1_base, self.d_base, self.s2_base)
        start_mac_c2 = max(resource_free_time["MAC"], end_mte3, end_l0_v)
        end_mac_c2 = start_mac_c2 + dur_mac_c2
        self.timeline.append(TimelineEvent(
            "MAC", "O", start_mac_c2, end_mac_c2, dur_mac_c2, "L0C", q_idx, k_idx
        ))
        resource_free_time["MAC"] = end_mac_c2
        l0_slots_v[slot_l0_v] = end_mac_c2

        size_fo = self._calc_fixpipe_o_size()
        dur_fix_o = self._calc_fixpipe_cycles(size_fo)
        start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_c2)
        end_fix_o = start_fix_o + dur_fix_o
        self.timeline.append(TimelineEvent(
            "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o, "UB", q_idx, k_idx
        ))
        resource_free_time["FIXPIPE"] = end_fix_o
        fixpipe_o_ready[k_idx] = end_fix_o

    # ── Phase 4: V2 ──────────────────────────────────────────────────
    for k_idx in k_group:
        dur_v2 = self._calc_vector_v2_cycles()
        start_v2 = max(resource_free_time["VECTOR_V2"], fixpipe_o_ready[k_idx])
        end_v2 = start_v2 + dur_v2
        self.timeline.append(TimelineEvent(
            "VECTOR_V2", "O", start_v2, end_v2, dur_v2, "CUBE", q_idx, k_idx
        ))
        resource_free_time["VECTOR_V2"] = end_v2
```

注意：同时将原有 `_calc_mac_cycles` 拆分为 `_calc_mac_cycles_c1(m,n,k)` 和 `_calc_mac_cycles_c2(m,n,k)` 分别调用对应 MAC 吞吐量方法。

**Step 5: 运行确认通过**

```bash
python -m pytest tests/test_n_buffer.py -v
```
预期：3 passed

**Step 6: Commit**

```bash
git add modelers/c1_modeler.py tests/test_n_buffer.py
git commit -m "feat: implement N_BUFFER inter-core pipeline (C1C1-V1V1-C2C2-V2V2)"
```

---

## Task 6: 验证/修正 PRELOAD 渐进式流水

**Files:**
- Modify: `modelers/c1_modeler.py`（`run_simulation` 的 PRELOAD 分支）
- Create: `tests/test_preload_pipeline.py`

**Step 1: 写测试**

```python
# tests/test_preload_pipeline.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(pipeline, two_buffer=False):
    return C1Modeler(
        s1_total=128, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=pipeline,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False, two_buffer=two_buffer,
    )

def test_preload_runs():
    m = make_modeler(InterCorePipeline.PRELOAD)
    _, _, _, total = m.run_simulation()
    assert total > 0

def test_preload_v_loaded_before_v1_ends():
    """
    PRELOAD 的核心：V 在 L1 中就绪时间 应早于 V1 完成时间（V 与 C1/V1 并行加载）。
    即第一个 MTE2-V 的 end_time < 第一个 VECTOR_V1 的 end_time
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()

    mte2_v = [e for e in timeline if e.unit == "MTE2" and "V" in e.operation]
    v1_events = [e for e in timeline if e.unit == "VECTOR_V1"]

    assert len(mte2_v) > 0 and len(v1_events) > 0
    assert mte2_v[0].end_time < v1_events[0].end_time, (
        f"V should be loaded before V1 ends: "
        f"V end={mte2_v[0].end_time:.1f}, V1 end={v1_events[0].end_time:.1f}"
    )

def test_preload_c2_has_no_mte2_v_wait():
    """
    PRELOAD 时 C2 阶段不应有 MTE2 V 加载事件（V 已在 L1，直接 MTE1）。
    验证：MTE2-V[k] 的 end_time 应 <= 对应 VECTOR_V1[k] 的 end_time（V1 开始前已准备好）
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()

    mte2_v = [e for e in timeline if e.unit == "MTE2" and "V" in e.operation]
    v1_events = [e for e in timeline if e.unit == "VECTOR_V1"]

    for v_evt, v1_evt in zip(mte2_v, v1_events):
        if v_evt.k_block_idx == v1_evt.k_block_idx:
            # V 就绪时间 <= V1 完成时间（而非 > V1 完成时间）
            assert v_evt.end_time <= v1_evt.end_time + 1.0  # +1 tolerrance

def test_preload_faster_than_default():
    """PRELOAD 应比 DEFAULT 周期数少或相当（V 预加载减少等待）"""
    m_default = make_modeler(InterCorePipeline.DEFAULT)
    m_preload = make_modeler(InterCorePipeline.PRELOAD)
    _, _, _, t_default = m_default.run_simulation()
    _, _, _, t_preload = m_preload.run_simulation()
    assert t_preload <= t_default, (
        f"PRELOAD should be faster: preload={t_preload:.0f}, default={t_default:.0f}"
    )
```

**Step 2: 运行，确认哪些失败**

```bash
python -m pytest tests/test_preload_pipeline.py -v
```

**Step 3: 修正 run_simulation 中 PRELOAD 分支**

确保 PRELOAD 分支调用 `_process_c1_with_v_preload` + `_process_c2_with_preloaded_v`，同时将旧 `self.preload == 1` 改为 `self.inter_core_pipeline == InterCorePipeline.PRELOAD`：

```python
elif self.inter_core_pipeline == InterCorePipeline.PRELOAD:
    if self.two_buffer:
        v_l1_slots = [0.0] * num_l1_slots
    else:
        v_l1_slots = map_k['l1_slots']

    for q_idx in range(self.q_block_count):
        self._load_q_block(q_idx, map_q, resource_free_time,
                           q_l1_ready_times, q_l0_ready_times)
        for k_idx in range(self.k_block_count):
            p_ready, v_l1_ready = self._process_c1_with_v_preload(
                q_idx, k_idx, map_k, v_l1_slots,
                resource_free_time, q_l0_ready_times
            )
            self._process_c2_with_preloaded_v(
                q_idx, k_idx, map_k, v_l1_slots,
                resource_free_time, p_ready, v_l1_ready
            )
```

同时更新 `_process_c1_with_v_preload` 和 `_process_c2_with_preloaded_v` 中的数据大小调用，改用新辅助方法。

**Step 4: 运行确认通过**

```bash
python -m pytest tests/test_preload_pipeline.py -v
```
预期：4 passed

**Step 5: Commit**

```bash
git add modelers/c1_modeler.py tests/test_preload_pipeline.py
git commit -m "fix: update PRELOAD to use InterCorePipeline enum and fix data size helpers"
```

---

## Task 7: 实现矩阵乘切分 — matmulK（沿 K 轴，L0C 累加，FIXPIPE 一次）

**Files:**
- Modify: `modelers/c1_modeler.py`（新增切分判断 + `_schedule_matmul_k_split`）
- Create: `tests/test_matmul_split.py`

**Step 1: 写失败测试**

```python
# tests/test_matmul_split.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(s2_base=256, d_base=128,
                 baseN_C1=256, baseK_C1=128,
                 baseN_C2=128, baseK_C2=256):
    return C1Modeler(
        s1_total=128, s2_total=256, d_total=256,
        s1_base_size=128, s2_base_size=s2_base, d_base_size=d_base,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=baseN_C1, baseK_C1=baseK_C1,
        baseM_C2=128, baseN_C2=baseN_C2, baseK_C2=baseK_C2,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
    )

def test_matmul_full_single_mac():
    """matmulFull：单个 k block 只有 1 个 MAC C1 事件"""
    # d_base=128 <= baseK_C1=128, s2_base=256 <= baseN_C1=256 → matmulFull
    m = make_modeler(s2_base=256, d_base=128, baseN_C1=256, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    assert len(mac_c1) == 1  # 1 q block × 1 k block × 1 sub-MAC

def test_matmulK_two_sub_macs():
    """matmulK：d_base=256 > baseK_C1=128 → 切成 2 个子 MAC，1 个 FIXPIPE"""
    # d_base=256, baseK_C1=128 → sub_count=2
    m = make_modeler(s2_base=256, d_base=256, baseN_C1=256, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    fixpipe_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    # 1 q block × 1 k block × 2 sub-MACs = 2 MAC events
    assert len(mac_c1) == 2
    # FIXPIPE 只有 1 次（等所有 sub-MACs 完成后）
    assert len(fixpipe_p) == 1

def test_matmulK_fixpipe_after_all_macs():
    """matmulK：FIXPIPE 开始时间 >= 最后一个 MAC C1 结束时间"""
    m = make_modeler(s2_base=256, d_base=256, baseN_C1=256, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    fixpipe_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    last_mac_end = max(e.end_time for e in mac_c1)
    assert fixpipe_p[0].start_time >= last_mac_end - 0.001
```

**Step 2: 运行确认失败**

```bash
python -m pytest tests/test_matmul_split.py::test_matmulK_two_sub_macs -v
```
预期：FAIL（当前没有切分逻辑）

**Step 3: 实现切分判断辅助方法**

在 C1Modeler 中新增：

```python
def _get_c1_split(self) -> tuple:
    """
    返回 C1 的切分类型和子块数。
    Returns: ('full', 1) | ('N', sub_count) | ('K', sub_count)
    优先检查 N，再检查 K。
    """
    if self.s2_base > self.baseN_C1:
        return 'N', math.ceil(self.s2_base / self.baseN_C1)
    if self.d_base > self.baseK_C1:
        return 'K', math.ceil(self.d_base / self.baseK_C1)
    return 'full', 1

def _get_c2_split(self) -> tuple:
    """
    返回 C2 的切分类型和子块数。
    C2: P(s1_base, s2_base) @ V(s2_base, d_base)，M=s1_base, N=d_base, K=s2_base
    """
    if self.d_base > self.baseN_C2:
        return 'N', math.ceil(self.d_base / self.baseN_C2)
    if self.s2_base > self.baseK_C2:
        return 'K', math.ceil(self.s2_base / self.baseK_C2)
    return 'full', 1
```

记得在文件顶部添加 `import math`。

**Step 4: 重构 C1 阶段的 MAC+FIXPIPE 部分**

在 `_process_k_blocks` 和 `_process_c1_stage` 中，将原来的单次 MAC→FIXPIPE 替换为：

```python
split_type, sub_count = self._get_c1_split()
last_mac_end = 0.0

if split_type == 'K':
    sub_k = math.ceil(self.d_base / sub_count)
    for i in range(sub_count):
        actual_k = min(sub_k, self.d_base - i * sub_k)
        # MTE1 Q_sub[i]: s1_base × actual_k × q_elem_size
        size_q_sub = self.s1_base * actual_k * self._get_q_element_size()
        dur_q_sub = self._calc_mte1_cycles(size_q_sub)
        start_q_sub = max(resource_free_time["MTE1"], q_l0_ready_times[q_idx])
        end_q_sub = start_q_sub + dur_q_sub
        # (可选) 添加 timeline event for Q sub-tile

        # MTE2+MTE1 K_sub[i]: s2_base × actual_k × kv_elem_size
        # ...（类似原 K 加载逻辑，但大小为子块）

        # MAC sub[i]
        dur_mac = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, actual_k)
        start_mac = max(resource_free_time["MAC"], end_q_sub, end_l0_k_sub)
        end_mac = start_mac + dur_mac
        # timeline.append(...)
        resource_free_time["MAC"] = end_mac
        last_mac_end = end_mac

    # 一次 FIXPIPE（累加结果）
    size_fp = self._calc_fixpipe_p_size()
    dur_fix_p = self._calc_fixpipe_cycles(size_fp)
    start_fix_p = max(resource_free_time["FIXPIPE"], last_mac_end)
    end_fix_p = start_fix_p + dur_fix_p
    # timeline.append(...)
    resource_free_time["FIXPIPE"] = end_fix_p

elif split_type == 'N':
    # matmulN：Q 加载一次，K 分段，每段各自 FIXPIPE
    ...（见 Task 8）

else:  # full
    # 原有单次 MAC → FIXPIPE 逻辑
    ...
```

**Step 5: 运行确认通过**

```bash
python -m pytest tests/test_matmul_split.py -v
```
预期：3 passed（matmulN 测试暂时 skip 或留到 Task 8）

**Step 6: Commit**

```bash
git add modelers/c1_modeler.py tests/test_matmul_split.py
git commit -m "feat: implement matmulK splitting for C1/C2 (K-axis, single FIXPIPE)"
```

---

## Task 8: 实现矩阵乘切分 — matmulN（沿 N 轴，分段 FIXPIPE）

**Files:**
- Modify: `modelers/c1_modeler.py`（补全 matmulN 分支 + C2 切分）
- Modify: `tests/test_matmul_split.py`（新增 matmulN 测试）

**Step 1: 补充失败测试（追加到 test_matmul_split.py）**

```python
def test_matmulN_two_fixpipes():
    """matmulN：s2_base=256 > baseN_C1=128 → 切成 2 sub，每 sub 各一次 FIXPIPE"""
    # s2_base=256, baseN_C1=128 → sub_count=2
    m = make_modeler(s2_base=256, d_base=128, baseN_C1=128, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    fixpipe_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    assert len(fixpipe_p) == 2  # 2 sub-tiles，各自 FIXPIPE

def test_c2_matmulK_single_fixpipe():
    """C2 matmulK：s2_base=256 > baseK_C2=128 → 2 sub-MACs，1 FIXPIPE O"""
    # C2: K=s2_base=256 > baseK_C2=128
    m = make_modeler(s2_base=256, d_base=128, baseN_C2=128, baseK_C2=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c2 = [e for e in timeline if e.unit == "MAC" and e.operation == "O"]
    fixpipe_o = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "O"]
    assert len(mac_c2) == 2
    assert len(fixpipe_o) == 1
```

**Step 2: 实现 matmulN 分支**

```python
elif split_type == 'N':
    sub_n = math.ceil(self.s2_base / sub_count)
    # Q 只加载一次到 L0A（Q 对所有 N-sub 相同）
    # K 按 N 分段逐段加载

    for i in range(sub_count):
        actual_n = min(sub_n, self.s2_base - i * sub_n)

        # MTE2+MTE1 K_sub[i]: actual_n × d_base × kv_elem_size
        # ... 加载第 i 段 K

        # MAC sub[i]（Q 已在 L0A，K_sub 在 L0B）
        dur_mac = self._calc_mac_cycles_c1(self.s1_base, actual_n, self.d_base)
        # ...

        # FIXPIPE P_sub[i]（每段各自搬运）
        size_fp_sub = self.s1_base * actual_n * 4  # FP32
        dur_fix_sub = self._calc_fixpipe_cycles(size_fp_sub)
        # ...
```

类似地，实现 C2 的切分（`_get_c2_split()` 已在 Task 7 中添加）。

**Step 3: 运行确认通过**

```bash
python -m pytest tests/test_matmul_split.py -v
```
预期：全部通过

**Step 4: Commit**

```bash
git add modelers/c1_modeler.py tests/test_matmul_split.py
git commit -m "feat: implement matmulN splitting for C1/C2 (N-axis, per-tile FIXPIPE)"
```

---

## Task 9: 更新配置模板（3 个文件）

**Files:**
- Modify: `modelers/templates/standard.py`
- Modify: `modelers/templates/dn_mode.py`
- Modify: `modelers/templates/full_load.py`

**StandardConfig 新版本：**

```python
@dataclass
class StandardConfig:
    # 矩阵维度
    s1_total: int = 256
    s2_total: int = 1024
    d_total: int = 128
    s1_base_size: int = 128
    s2_base_size: int = 256
    d_base_size: int = 128

    # 数据类型（分离）
    q_data_type: DataType = DataType.FP16
    kv_data_type: DataType = DataType.FP16

    # 矩阵乘基本块
    baseM_C1: int = 128
    baseN_C1: int = 128
    baseK_C1: int = 128
    baseM_C2: int = 128
    baseN_C2: int = 128
    baseK_C2: int = 128

    # 流水线模式
    inter_core_pipeline: InterCorePipeline = InterCorePipeline.DEFAULT
    inner_core_pipeline: InnerCorePipeline = InnerCorePipeline.DEFAULT

    # 缓存和优化
    is_l2cache: bool = False
    use_dn: bool = False
    L1_db: bool = False
    L0_db: bool = False
    full_load: bool = False
    two_buffer: bool = False

    def to_dict(self):
        return {
            's1_total': self.s1_total,
            's2_total': self.s2_total,
            'd_total': self.d_total,
            's1_base_size': self.s1_base_size,
            's2_base_size': self.s2_base_size,
            'd_base_size': self.d_base_size,
            'q_data_type': self.q_data_type,
            'kv_data_type': self.kv_data_type,
            'baseM_C1': self.baseM_C1,
            'baseN_C1': self.baseN_C1,
            'baseK_C1': self.baseK_C1,
            'baseM_C2': self.baseM_C2,
            'baseN_C2': self.baseN_C2,
            'baseK_C2': self.baseK_C2,
            'inter_core_pipeline': self.inter_core_pipeline,
            'inner_core_pipeline': self.inner_core_pipeline,
            'is_l2cache': self.is_l2cache,
            'use_dn': self.use_dn,
            'L1_db': self.L1_db,
            'L0_db': self.L0_db,
            'full_load': self.full_load,
            'two_buffer': self.two_buffer,
        }
```

DNModeConfig 和 FullLoadConfig 继承 StandardConfig，删除旧有参数引用即可。

导入更新：
```python
from core import DataType, InterCorePipeline, InnerCorePipeline
```

验证：
```bash
python -c "from modelers.templates import StandardConfig, DNModeConfig, FullLoadConfig; print('ok')"
```

**Commit:**
```bash
git add modelers/templates/
git commit -m "refactor: update templates to new C1Modeler API"
```

---

## Task 10: 更新 examples（3 个文件）

**Files:**
- Modify: `examples/run_c1_examples.py`
- Modify: `examples/test_full_pipeline.py`
- Modify: `examples/test_preload.py`

**run_c1_examples.py 示例改动：**

```python
from core import DataType, InterCorePipeline, InnerCorePipeline

def example_standard():
    modeler = C1Modeler(
        s1_total=256, s2_total=1024, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    ...

def example_n_buffer():
    """N-Buffer 流水示例"""
    modeler = C1Modeler(
        s1_total=256, s2_total=1024, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=InterCorePipeline.N_BUFFER,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    ...
```

test_preload.py 的三个测试更新为使用 `InterCorePipeline.PRELOAD` 替代 `preload=1`。

验证所有 examples 无报错：
```bash
python examples/run_c1_examples.py
python examples/test_full_pipeline.py
python examples/test_preload.py
```

**Commit:**
```bash
git add examples/
git commit -m "refactor: update examples to use new InterCorePipeline API"
```

---

## Task 11: 更新 reference.md section 7（移除"待修改"）

**Files:**
- Modify: `reference.md`（section 7 使用示例）

将 section 7 从：
```python
modeler = MatmulModeler(
    ...
    data_type=DataType.FP16,
    inter_core_pipeline=InterCorePipeline.DEFAULT
    inner_core_pipeline=InnerCorePipeline.DEFAULT
)
```

更新为最终 API：
```python
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

modeler = C1Modeler(
    s1_total=256, s2_total=512, d_total=256,
    s1_base_size=128, s2_base_size=128, d_base_size=256,

    # C1 矩阵乘基本块参数
    baseM_C1=128, baseN_C1=128, baseK_C1=128,
    # C2 矩阵乘基本块参数
    baseM_C2=128, baseN_C2=128, baseK_C2=128,

    q_data_type=DataType.FP16,
    kv_data_type=DataType.FP16,
    is_l2cache=True,
    use_dn=False,
    L1_db=True,
    L0_db=True,
    full_load=False,
    inter_core_pipeline=InterCorePipeline.DEFAULT,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)
```

**Commit:**
```bash
git add reference.md
git commit -m "docs: update reference.md section 7 with final API"
```

---

## Task 12: 全量验证 + 更新进度文档

**Step 1: 运行全部测试**

```bash
python -m pytest tests/ -v
```
预期：全部通过

**Step 2: 运行全部 examples**

```bash
python examples/run_c1_examples.py
python examples/test_full_pipeline.py
python examples/test_preload.py
```
预期：无异常，生成 PNG 时间线图

**Step 3: 更新设计文档进度表**

将 `docs/plans/2026-02-28-c1c2-improvements-design.md` 中"实现进度"表格所有 ⬜ 改为 ✅。

**Step 4: 最终 Commit**

```bash
git add docs/
git commit -m "docs: mark all implementation tasks as completed"
```

---

## 注意事项

1. **`_process_k_blocks`、`_process_c1_stage`、`_process_c2_stage`** 中的数据大小计算必须全部改用新辅助方法，避免遗漏。

2. **matmulK 的 Q 子块加载**：K 轴切分时，Q 也需按 sub_k 切分为子块分别 MTE1（因为 L0A 只存当前 sub-tile）。用 `_calc_mte1_cycles(s1_base * actual_k * q_elem_size)` 计算。

3. **matmulN 的 Q 加载**：N 轴切分时 Q 不变，Q 只在该 k block 第一次进入 L0A（和现有逻辑一致），无需重复加载。

4. **C2 切分的 P 处理**：C2 matmulK 时，P 按 K 轴（s2_base）切分，P 的每个子块通过 MTE3 分段搬回 CUBE；C2 matmulN 时，P 全量在 L0A，V 按 N 轴（d_base）分段。

5. **`two_buffer` 参数**：在 PRELOAD 模式中保留 `two_buffer=True/False` 选项（控制 V 是否使用独立 L1 slot）。N_BUFFER 模式暂不支持 `two_buffer`。
