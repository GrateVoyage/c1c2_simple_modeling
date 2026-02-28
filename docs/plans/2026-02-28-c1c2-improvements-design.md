# C1C2建模器完善设计文档

**日期**: 2026-02-28
**状态**: 已批准，待实现
**参考**: reference.md

---

## 背景

当前 `C1Modeler` 已实现 C1V1C2V2 完整流水线基础功能，但与 reference.md 规格存在以下差距：

1. 缺少 `InterCorePipeline` / `InnerCorePipeline` 枚举
2. `preload=1` 的行为语义与 reference.md 描述的渐进式流水不一致
3. 缺少 N-Buffer 流水模式（C1C1V1V1C2C2V2V2）
4. 缺少矩阵乘内部切分参数（baseM/N/K_C1, baseM/N/K_C2）
5. Q 和 KV 的数据类型未分离（现为单一 `data_type`）
6. FIXPIPE 大小未按 FP32 计算（L0C 输出恒为 FP32）

---

## 设计决策

**方案选择**: 干净重构 C1Modeler（方案二），替换旧参数，同步更新 examples。
**理由**: 项目处于早期阶段，API 清晰度优先于向后兼容。

---

## 一、枚举扩展（core/enums.py）

### 新增 InterCorePipeline

```python
class InterCorePipeline(Enum):
    DEFAULT  = "default"    # 顺序流水: C1V1C2V2 → C1V1C2V2 → ...
    PRELOAD  = "preload"    # 渐进式:   C1 → C1V1C2 → C1V1C2V2 → ... → V2
    N_BUFFER = "n_buffer"   # N=2批次:  C1C1V1V1C2C2V2V2 → C1C1V1V1C2C2V2V2
```

### 新增 InnerCorePipeline

```python
class InnerCorePipeline(Enum):
    DEFAULT = "default"     # Q常驻L1，KP共用2块144K，V使用2块32K
```

---

## 二、C1Modeler 参数变更

### 参数替换对照表

| 旧参数 | 新参数 | 说明 |
|--------|--------|------|
| `data_type: DataType` | `q_data_type: DataType` | Q矩阵数据类型 |
| —（同上） | `kv_data_type: DataType` | K/V矩阵数据类型 |
| `preload: int` | `inter_core_pipeline: InterCorePipeline` | 核间流水模式 |
| 无 | `inner_core_pipeline: InnerCorePipeline` | 核内流水模式 |
| 无 | `baseM_C1, baseN_C1, baseK_C1` | C1矩阵乘基本块最大单位 |
| 无 | `baseM_C2, baseN_C2, baseK_C2` | C2矩阵乘基本块最大单位 |

### 新接口示例

```python
modeler = C1Modeler(
    s1_total=256, s2_total=512, d_total=256,
    s1_base_size=128, s2_base_size=128, d_base_size=256,
    baseM_C1=128, baseN_C1=128, baseK_C1=128,
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

---

## 三、核间流水实现逻辑

### DEFAULT（顺序流水）

现有逻辑不变。对每个 (q, k) 块顺序执行 C1→V1→C2→V2。
V-load 在 V1 完成后发起（`start_l1_v = max(..., end_vector_v1)`）。

### PRELOAD（渐进式流水）

V 在 K-load 完成后立即预加载，与 C1/V1 计算并行：

```
for k_idx:
    MTE2 K[k] → L1
    MTE2 V[k] → L1   ← 紧接 K-load，不等 V1
    MTE1 K[k] → L0
    MAC  C1[k]        ← Q @ K^T
    FIXPIPE P[k] → UB
    VECTOR_V1 P[k]    ← Softmax（V已在L1，并行等待）
    MTE3 P[k] → CUBE
    MTE1 V[k] → L0    ← 直接从L1取，省去MTE2
    MAC  C2[k]        ← P @ V
    FIXPIPE O[k] → UB
    VECTOR_V2 O[k]
```

支持 `two_buffer=True`：V 使用独立 L1 slot，与 K slot 无冲突。

### N_BUFFER（N=2 批次流水）

按 k 块分组（每组 N=2），组内按阶段批次执行：

```
for group (k0, k1):
    Phase 1 [C1]: k0: MTE2 K → MTE1 → MAC C1 → FIXPIPE
                  k1: MTE2 K → MTE1 → MAC C1 → FIXPIPE
    Phase 2 [V1]: k0: VECTOR_V1  ← 与 Phase1-k1 的 MAC 并行（不同硬件）
                  k1: VECTOR_V1
    Phase 3 [C2]: k0: MTE2 V → MTE3 → MTE1 → MAC C2 → FIXPIPE
                  k1: MTE2 V → MTE3 → MTE1 → MAC C2 → FIXPIPE
    Phase 4 [V2]: k0: VECTOR_V2
                  k1: VECTOR_V2
```

关键并行：V1[k0] 与 MAC C1[k1] 使用不同硬件，可通过 resource_free_time 自然重叠。

---

## 四、矩阵乘切分逻辑

### 切分判断（以 C1 为例）

```
matmulFull: s2_base <= baseN_C1 AND d_base <= baseK_C1
matmulN:    s2_base > baseN_C1   (沿 N 轴切分，sub_count = ceil(s2_base / baseN_C1))
matmulK:    d_base  > baseK_C1   (沿 K 轴切分，sub_count = ceil(d_base  / baseK_C1))
```

C2 同理，用 baseM/N/K_C2，矩阵维度为 (s1_base, d_base, s2_base)。

### matmulK 调度（L0C 累加，FIXPIPE 一次）

```
for i in range(sub_count):
    MTE2 K_sub[i](s2_base × baseK_C1) → L1
    MTE1 Q_sub[i](s1_base × baseK_C1) → L0A  (Q 从 L1 重新搬子块)
    MTE1 K_sub[i] → L0B
    MAC  C1_sub[i]  ← L0C 累加
FIXPIPE P (s1_base × s2_base × 4 bytes, FP32)  ← 所有 sub-MAC 完成后
```

### matmulN 调度（分段 FIXPIPE）

```
for i in range(sub_count):
    MTE1 Q(全量) → L0A  (仅第一次，或每次按需)
    MTE2 K_sub[i](s2_base/sub_count × d_base) → L1
    MTE1 K_sub[i] → L0B
    MAC  C1_sub[i]   ← L0C 独立分块
    FIXPIPE P_sub[i] (baseN_C1 × s1_base × 4 bytes, FP32)  ← 每个子块搬运
```

---

## 五、数据类型分离

### 元素大小

```python
q_elem_size  = 2 if q_data_type  == DataType.FP16 else 1
kv_elem_size = 2 if kv_data_type == DataType.FP16 else 1
```

### 各单元数据大小

| 单元 | 数据 | 大小公式 |
|------|------|---------|
| MTE2 Q | Q | s1_base × d_base × q_elem_size |
| MTE1 Q | Q | s1_base × d_base × q_elem_size |
| MTE2 K | K | s2_base × d_base × kv_elem_size |
| MTE1 K | K | s2_base × d_base × kv_elem_size |
| MTE2 V | V | s2_base × d_base × kv_elem_size |
| MTE1 V | V | s2_base × d_base × kv_elem_size |
| FIXPIPE P | P (L0C→UB) | s1_base × s2_base × **4** (FP32) |
| MTE3 P | P (UB→CUBE) | s1_base × s2_base × kv_elem_size |
| FIXPIPE O | O (L0C→UB) | s1_base × d_base × **4** (FP32) |

### MAC 吞吐量规则

- C1（Q @ K^T）：q_data_type **AND** kv_data_type 都为 FP8 → FP8 吞吐；否则 FP16
- C2（P @ V）：kv_data_type 决定（P 和 V 类型一致）

---

## 六、hardware_config.py 扩展

新增方法：

```python
def get_mac_throughput_c1(self, q_dtype: DataType, kv_dtype: DataType) -> int:
    if q_dtype == DataType.FP8 and kv_dtype == DataType.FP8:
        return self.MAC_THROUGHPUT_FP8
    return self.MAC_THROUGHPUT_FP16

def get_mac_throughput_c2(self, kv_dtype: DataType) -> int:
    return self.MAC_THROUGHPUT_FP8 if kv_dtype == DataType.FP8 else self.MAC_THROUGHPUT_FP16
```

---

## 七、文件修改清单

| 文件 | 变更 |
|------|------|
| `core/enums.py` | 新增 InterCorePipeline, InnerCorePipeline |
| `core/__init__.py` | 导出新枚举 |
| `core/hardware_config.py` | 新增 get_mac_throughput_c1/c2 |
| `modelers/c1_modeler.py` | 全面重构：新参数、N-Buffer、matmul切分、数据类型分离 |
| `modelers/templates/standard.py` | 更新为新 API |
| `modelers/templates/dn_mode.py` | 更新为新 API |
| `modelers/templates/full_load.py` | 更新为新 API |
| `examples/run_c1_examples.py` | 更新示例代码 |
| `examples/test_full_pipeline.py` | 更新为新 API |
| `examples/test_preload.py` | 更新为新 API |
| `reference.md` section 7 | 移除"待修改"，更新示例 |

---

## 八、进度追踪

> 本节由实现者持续更新，方便后续接手。

### 状态说明
- ⬜ 未开始
- 🔄 进行中
- ✅ 已完成
- ❌ 有问题

### 实现进度

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | core/enums.py — 新增枚举 | ✅ | |
| 2 | core/__init__.py — 导出 | ✅ | |
| 3 | core/hardware_config.py — 新增方法 | ✅ | |
| 4 | c1_modeler.py — 替换参数签名 | ✅ | |
| 5 | c1_modeler.py — 数据类型分离 | ✅ | |
| 6 | c1_modeler.py — FIXPIPE 改为 FP32 大小 | ✅ | |
| 7 | c1_modeler.py — PRELOAD 渐进式流水（验证/修正） | ✅ | |
| 8 | c1_modeler.py — N_BUFFER 流水实现 | ✅ | |
| 9 | c1_modeler.py — matmulK 切分 | ✅ | |
| 10 | c1_modeler.py — matmulN 切分 | ✅ | |
| 11 | 模板文件更新（3个） | ✅ | |
| 12 | examples 更新（3个文件） | ✅ | |
| 13 | reference.md section 7 更新 | ✅ | |
| 14 | 运行所有 examples 验证无报错 | ✅ | |

### 已知问题 / 待确认

- 已确认：MTE3 P 大小使用 kv_elem_size（P 经 V1 后保持 kv_data_type 格式）
- 已确认：matmulN 时 Q 全量加载一次，K/V 按子块在循环内逐次加载

---

*设计文档版本: v1.0 — 2026-02-28*
