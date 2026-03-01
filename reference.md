## 🎯 实现目标

以Cube的视角实现昇腾NPU上Flash Attention算子的完整C1V1C2V2流水线建模。


## 1. 项目结构(可修改)

```
c1c2_modeling/
├── core/                          核心定义
│   ├── enums.py                  枚举类型
│   ├── dataclasses.py            数据结构
│   └── hardware_config.py        硬件配置
│
├── modelers/                      建模器
│   ├── c1_modeler.py             完整流水线建模器
│   └── templates/                配置模板
│       ├── standard.py
│       ├── dn_mode.py
│       └── full_load.py
│
├── utils/                         工具
│   ├── visualizer.py             可视化 (已更新)
│   └── logger.py                 日志
│
├── examples/                      示例
│   ├── playground.py             单一可配参数调试入口 (推荐)
│   ├── run_c1_examples.py        配置示例
│   └── test_full_pipeline.py     完整流水线测试
│
└── 文档
    ├── README.md                 主文档 (已更新)
    ├── CHANGELOG_V2.md           更新日志
    ├── QUICK_REF_V2.py           快速参考
    └── FINAL_SUMMARY.md          
```


## 2. 完整流水线实现
### 2.1 Flash Attention 算子完整流程
```
C1: Q @ K^T → S (Attention Score) - 第一次矩阵乘法
V1: Softmax S → P (Attention Probability) - 第一次向量操作
C2: P @ V → O (Output) - 第二次矩阵乘法
V2: 后处理 - 第二次向量操作
```
### 2.2 流水线详细步骤

```
C1阶段: Q @ K^T → P (Attention Score)
  ├─ MTE2: 加载Q和K
  ├─ MTE1: 搬运到L0
  ├─ MAC: 矩阵乘法
  └─ FIXPIPE: 搬P到UB

V1阶段: Softmax等操作
  └─ VECTOR_V1: 处理P矩阵 (800 cycles)

转换阶段:
  ├─ MTE3: P从UB搬回CUBE
  ├─ MTE2: 并行加载V矩阵
  └─ MTE1: P/V搬到L0

C2阶段: P @ V → O (Output)
  ├─ MAC: 矩阵乘法
  └─ FIXPIPE: 搬O到UB

V2阶段: 后处理
  └─ VECTOR_V2: 处理O矩阵 (800 cycles)
```
## 3. 硬件单元说明
### 3.1 搬运及计算单元
```
HARDWARE_UNITS = {
    "MTE2": {
        "功能": "DRAM/L2 → L1",
        "带宽_DRAM": "1600 GB/s / 32",
        "带宽_L2": "5400 GB/s / 32",
        "用途": "加载Q, K, V矩阵"
    },
    "MTE1": {
        "功能": "L1 → L0",
        "带宽": "256 bytes/cycle",
        "用途": "搬运Q, K, V到L0"
    },
    "MTE3": {
        "功能": "UB → L1",
        "带宽": "256 bytes/cycle",
        "用途": "P矩阵从UB搬回L1，再经MTE1到L0用于C2"
    },
    "MAC": {
        "功能": "矩阵乘法",
        "吞吐_FP16": "16*16*16*2",
        "吞吐_FP8": "16*32*16*2",
        "用途": "C1: Q@K^T→P, C2: P@V→O"
    },
    "FIXPIPE": {
        "功能": "L0C → UB",
        "带宽": "128 bytes/cycle",
        "用途": "搬运P和O矩阵"
    },
    "VECTOR_V1": {
        "功能": "向量操作",
        "周期": "1600 cycles (固定)",
        "用途": "Softmax等处理P"
    },
    "VECTOR_V2": {
        "功能": "向量操作",
        "周期": "400 cycles (固定)",
        "用途": "后处理O"
    }
}
```
### 3.2 存储单元
```
STORE_UNITS = {
    "L1 Buffer": {
        "大小": "512 KB",
        "槽位": "由 核内流水 决定",
        "缓冲区名称": "MTE2A / MTE2B (原L1A / L1B)",
    },
    "L0A Buffer (MTE1A)": {
        "大小": "64 KB",
        "槽位": "由 L0_db 决定。False=1槽，True=2槽",
        "用途": "标准模式存Q/P；DN模式存K/V"
    },
    "L0B Buffer (MTE1B)": {
        "大小": "64 KB",
        "槽位": "由 L0_db 决定。False=1槽，True=2槽",
        "用途": "标准模式存K/V；DN模式存Q/P"
    },
    "L0C Buffer": {
        "大小": "64 KB",
        "槽位": "由 L0c_db 决定",
        "输出类型": "恒为FP32 (4 bytes/element)"
    },
}
```
### 3.3 搬运及计算约束
1. L0A和L0B必须串行（共用MTE1资源），L1A和L1B也必须串行（共用MTE2资源）。
   - 实现中 MTE2A/MTE2B 表示L1A/L1B缓冲区，MTE1A/MTE1B 表示L0A/L0B缓冲区。
2. **VECTOR_V1 和 VECTOR_V2 必须串行**（共用同一个 VECTOR 向量单元资源）。
   - V2 的启动时间 = max(VECTOR 单元空闲时间, FIXPIPE-O 完成时间)。
3. C2阶段P的完整路由：FIXPIPE(L0C→UB) → VECTOR_V1(UB处理) → MTE3(UB→L1) → MTE1(L1→L0A/B) → MAC-O。
   - MAC-O 同时依赖 MTE1-P（P已在L0）和 MTE1-V（V已在L0）两个条件。
4. CUBE核内依赖关系如下，_表示后面组件等前面组件搬运或计算结束。
- C1
    - 消费者等生产者：
    MTE2_MTE1、MAC_FIXPIPE、MTE1_MAC
    - 生产者等消费者：
    MTE1_MTE2、FIXPIPE_MAC、MAC_MTE1
- C2
    - 消费者等生产者：
    MTE3_MTE1(P路由)、MTE2_MTE1(V加载)、MAC_FIXPIPE、MTE1_MAC
    - 生产者等消费者：
    MTE1_MTE3、MTE1_MTE2、FIXPIPE_MAC、MAC_MTE1

5. CUBE和VECTOR核间，每个基本块的C1V1C2V2都是有依赖关系的，不同基本块之间的C1V1C2V2没有依赖关系。
6. 所有组件之间如果没有依赖关系均可并行，以最早的时间执行。
6. L0C输出恒为FP32：FIXPIPE-P搬运大小 = s1×s2×4 bytes；FIXPIPE-O大小 = s1×d×4 bytes。


## 4. 模式特性支持
### 4.1 DN模式支持

**标准模式路径**:
- Q → MTE2A → MTE1A (L0A)
- K → MTE2B → MTE1B (L0B)
- V → MTE2B → MTE1B (L0B)
- P → MTE3 → MTE2A(L1) → MTE1A (L0A)

**DN模式路径**:
- Q → MTE2B → MTE1B (L0B)
- K → MTE2A → MTE1A (L0A)
- V → MTE2A → MTE1A (L0A)
- P → MTE3 → MTE2B(L1) → MTE1B (L0B)

### 4.2 核间流水
实际执行时默认为顺序流水，可以根据输入标志不同进行选择
1. 顺序流水(默认情况)

**执行顺序**: C1V1C2V2 -> C1V1C2V2 ->... -> C1V1C2V2 -> C1V1C2V2

2. Preload流水
Preload = 1

**执行顺序**: C1 -> C1V1C2 -> C1V1C2V2 -> C1V1C2V2 -> ... -> V1C2V2 -> C2V2 -> V2

3. N-Bufer流水
N=2
**执行顺序**: C1C1V1V1C2C2V2V2 -> C1C1V1V1C2C2V2V2 -> ... -> C1C1V1V1C2C2V2V2

### 4.3 核内流水
支持两种策略（inner_core_pipeline 参数）：

**DEFAULT（默认，基础L1容量追踪）**:
- L1总容量 512 KB，采用平坦LRU池追踪所有块（Q/K/V/P）。
- 若块未被驱逐仍在L1中，跳过MTE2，直接用MTE1从L1搬到L0。
- 任何块在当前被MTE1使用后，占用L1直至被驱逐（空间不足时LRU淘汰）。

**Q_RESIDENT（Q常驻）**:
- L1分区固定（总512 KB）：
  - Q槽：1 × 144 KB（永驻，不参与LRU淘汰）
  - KP池：2 × 144 KB（K块和P块共享，LRU替换）
  - V池：2 × 32 KB（LRU替换）
- 若块在对应槽中命中，跳过MTE2直接用MTE1。
- 超出容量时按LRU策略替换对应槽中的最旧块。

### 4.4 矩阵乘切分

**输入参数**：baseM_C1/baseN_C1/baseK_C1 是 C1 MAC 的最大子块尺寸；baseM_C2/baseN_C2/baseK_C2 是 C2 MAC 的最大子块尺寸。

**切分判定优先级**（先判 N，再判 K，否则 Full）：
```
C1: s2_base > baseN_C1 → matmulN (sub_count = ceil(s2_base / baseN_C1))
    d_base  > baseK_C1 → matmulK (sub_count = ceil(d_base  / baseK_C1))
    否则               → matmulFull

C2: d_base  > baseN_C2 → matmulN (sub_count = ceil(d_base  / baseN_C2))
    s2_base > baseK_C2 → matmulK (sub_count = ceil(s2_base / baseK_C2))
    否则               → matmulFull
```

**通用规则**：不论哪种切分，MTE2 始终对完整基本块做一次整体加载（DRAM/L2 → L1），L0 分块由 MTE1 按子块搬运。

---

#### 1. matmulN（切 N 维，即右矩阵的列）

```
C1: 右矩阵 K 形状 [s2_base, d_base]，切 s2_base → 子块 [actual_n, d_base]
C2: 右矩阵 V 形状 [s2_base, d_base]，切 d_base  → 子块 [s2_base, actual_n]
```

N 维子块写入 L0C 的不同列区域，全部子块 MAC 完成后发出一次 FIXPIPE：
```
MTE2 (完整块, 1次)
└─ for i in range(sub_count):
       MTE1_i  (子块 L1 → L0)
       MAC_i   (i=0: 等 MTE1_i + 左矩阵就绪;  i>0: 等 MTE1_i)   ← L0C 不同列区域
FIXPIPE (1次, 等所有 MAC 完成)
```

L0 槽位：`sub_slot = i % len(l0_slots)` 轮换，与 matmulK 相同。

**L0_db 对 matmulN 的影响**：
- `L0_db=False`（1 槽）：sub_slot 恒为 0，MTE1_i+1 必须等 MAC_i 释放槽位，MTE1 与 MAC **串行**。
- `L0_db=True` （2 槽）：sub_slot 在 0/1 间轮换，MTE1_i+1 可在 MAC_i 运行时写入另一槽，MTE1 与 MAC **可流水**。

---

#### 2. matmulK（切 K 维，即左矩阵的列 / 右矩阵的行）

```
C1: 右矩阵 K 形状 [s2_base, d_base]，切 d_base  → 子块 [s2_base, actual_k]
C2: 右矩阵 V 形状 [s2_base, d_base]，切 s2_base → 子块 [actual_k, d_base]
```

K 维累加到 L0C，全部子块 MAC 完成后发出一次 FIXPIPE：
```
MTE2 (完整块, 1次)
└─ for i in range(sub_count):
       sub_slot = i % len(l0_slots)        # 轮换 L0 槽位
       MTE1_i  (子块 L1 → L0[sub_slot])
       MAC_i   (i=0: 等 MTE1_i + 左矩阵就绪;  i>0: 等 MTE1_i)   ← L0C 累加
       l0_slots[sub_slot] = MAC_i 结束时间
l1_slots = MTE1 全部完成后释放
FIXPIPE (1次, 等所有 MAC 完成)
```

**L0_db 对 matmulK 的影响**：
- `L0_db=False`（1 槽）：sub_slot 恒为 0，MTE1_i+1 必须等 MAC_i 释放槽位，MTE1 与 MAC **串行**。
- `L0_db=True` （2 槽）：sub_slot 在 0/1 间轮换，MTE1_i+1 可在 MAC_i 运行时写入另一槽，MTE1 与 MAC **可流水**。

---

#### 3. matmulFull（不切分）

```
MTE2 (完整块, 1次) → MTE1 (完整块, 1次) → MAC (1次) → FIXPIPE (1次)
```

L0 槽位：使用固定的 `slot_l0`，MTE1 等槽位空闲后搬入，MAC 完成时释放。

## 5. 特殊逻辑:
1. full_load=True: Q 矩阵在模拟开始前全部加载到 L1，后续不再触发 Q 的 MTE2。
2. q_data_type=DataType.FP16/DataType.FP8，输入Q矩阵的数据类型，影响C1的Q搬运和QK计算。
3. kv_data_type=DataType.FP16/DataType.FP8，输入K矩阵的数据类型，影响C1C2的K/P/V搬运和PV计算。
4. L0C的得到的矩阵乘结果，即fixpipe搬运的数据恒为FP32。
5. is_l2cache=True：相同基本块第一次加载使用DRAM带宽，下次加载使用L2带宽。不使能时每次均为DRAM带宽。
   - K 块：第二个及以上 q_idx 复用时使用L2带宽（`use_l2 = is_l2cache and q_idx > 0`）
   - V 块：同K块，第二个及以上 q_idx 使用L2带宽（`use_l2 = is_l2cache and q_idx > 0`）
6. matmulK/matmulN 切分：切分时所有子块 MAC 完成后才发出一次 FIXPIPE（详见4.4节）。
   - matmulK：K 维累加到 L0C，子块结果在 L0C 内叠加，最后统一 FIXPIPE。
   - matmulN：N 维子块写入 L0C 的不同列区域，全部完成后统一 FIXPIPE。

## 6. 可视化增强

### Y轴布局 (从下到上)
```
VECTOR_V2  (橙色 #E59866)  ■■■ O11 ■■■ O12
VECTOR_V1  (黄色 #F7DC6F)  ■■■ P11 ■■■ P12
MTE3       (灰色 #95A5A6)  ■ P11 ■ P12        (UB → L1缓冲)
FIXPIPE    (绿色 #96CEB4)  ■ P11/O11
MAC        (蓝色 #45B7D1)  ■■ P11/O11
MTE1(L0B)  (青色 #4ECDC4)  ■ K1/V1
MTE1(L0A)  (青色 #4ECDC4)  ■ Q1/P1
MTE2       (红橙)          ■■ K1/V1            (对应 L1B)
MTE2       (红橙)          ■■ Q1               (对应 L1A)
```

### 性能测试结果示例
```
总周期数: 29,679
MTE2:      61.1% (瓶颈)
MAC:       27.6%
VECTOR_V1: 10.8%
VECTOR_V2: 10.8%
MTE3:      3.5%
MTE1:      7.8%
FIXPIPE:   5.2%
```

## 7. 使用示例

```python
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

# 标准模式（顺序流水）
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

timeline, _, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "timeline.png")

# PRELOAD 渐进式流水
modeler_preload = C1Modeler(
    s1_total=256, s2_total=512, d_total=256,
    s1_base_size=128, s2_base_size=128, d_base_size=256,
    baseM_C1=128, baseN_C1=128, baseK_C1=128,
    baseM_C2=128, baseN_C2=128, baseK_C2=128,
    q_data_type=DataType.FP16,
    kv_data_type=DataType.FP16,
    inter_core_pipeline=InterCorePipeline.PRELOAD,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)

# N_BUFFER 流水（N=2批次）
modeler_nbuffer = C1Modeler(
    s1_total=256, s2_total=512, d_total=256,
    s1_base_size=128, s2_base_size=128, d_base_size=256,
    baseM_C1=128, baseN_C1=128, baseK_C1=128,
    baseM_C2=128, baseN_C2=128, baseK_C2=128,
    q_data_type=DataType.FP8,
    kv_data_type=DataType.FP8,
    inter_core_pipeline=InterCorePipeline.N_BUFFER,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)
```