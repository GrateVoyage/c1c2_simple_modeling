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
        "功能": "UB → CUBE",
        "带宽": "256 bytes/cycle",
        "用途": "P矩阵搬回CUBE用于C2"
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
        "周期": "800 cycles (固定)",
        "用途": "Softmax等处理P"
    },
    "VECTOR_V2": {
        "功能": "向量操作",
        "周期": "800 cycles (固定)",
        "用途": "后处理O"
    }
}
```
### 3.2 存储单元
```
STORE_UNITS = {
    "L1 Buffer": {
        "大小": "512 KB",
        "槽位": 由 核内流水 决定。
    },
    "L0A Buffer": {
        "大小": "64 KB",
        "槽位": 由 L0_db 决定，当单个矩阵乘基本块大小小于等于32KB可开启duouble buffer。False=1个槽位，True=2个槽位。
    },
    "L0B Buffer": {
        "大小": "64 KB",
        "槽位": 由 L0_db 决定，当单个矩阵乘基本块大小小于等于32KB可开启duouble buffer。False=1个槽位，True=2个槽位。
    },
    "L0C Buffer": {
        "大小": "64 KB",
        "槽位": 由 L0c_db 决定，当单个矩阵乘结果基本块大小小于等于128KB可开启duouble buffer。False=1个槽位，True=2个槽位。
    },
}
```
### 3.3 搬运及计算约束
1. L0A和L0B必须串行，L1A和L1B必须串行。
2. CUBE核内依赖关系如下，_表示后面组件等前面组件搬运或计算结束。
- C1
    - 消费者等生产者：
    MTE2_MTE1、MAC_FIXPIP、MTE1_MAC
    - 生产者等消费者：
    MTE1_MTE2、FIXPIP_MAC、MAC_MTE1
- C2
    - 消费者等生产者：
    MTE3_MTE1、MTE2_MTE1、MAC_FIXPIP、MTE1_MAC
    - 生产者等消费者：
    MTE1_MTE3、MTE1_MTE2、FIXPIP_MAC、MAC_MTE1

3. CUBE和VECTOR核间，每个基本块的C1V1C2V2都是有依赖关系的，不同基本块之间的C1V1C2V2没有依赖关系。
4. 所有组件之间如果没有依赖关系均可并行，以最早的时间执行。


## 4. 模式特性支持
### 4.1 DN模式支持

**标准模式路径**:
- Q → L1A → L0A
- K → L1B → L0B
- V → L1B → L0B

**DN模式路径**:
- Q → L1B → L0B
- K → L1A → L0A
- V → L1A → L0A

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
暂时只考虑默认情况，Q常驻流水L1。L1=512K，Q分配1块144K的Buffer，在循环内常驻；KP共用2块144K的Buffer；V使用2块32K的Buffer，如果基本块加载数据量没有达到上限就可以一直加载，某块重复使用不需要再次用MTE2加载，可以直接使用MTE1从L1加载。如果达到上限就会从最开始加载的基本块进行替换，如果再次需要该块只能再次用MTE2加载搬运。

### 4.4 矩阵乘切分
加上两组输入参数baseM_C1，baseN_C1,baseK_C1,这是c1进行矩阵乘的基本块的最大单位，参数baseM_C2，baseN_C2,baseK_C2,这是c2进行矩阵乘的基本块的最大单位。

比如baseM_C1=128，baseN_C1=128，baseK_C1=128，如果[s1_base_size,d_base_size]=[128,128]，那么单个C1就会L0A搬运一次只计算一次，如果[s1_base_size,d_base_size]=[128,256]，此时256/128=2，那么这个基本块单个C1内会被切成2个[128,128]进行计算。fixpip会等这两次算完进行累加才进行搬运，因为这是一个C1的全流程。C2同理。

只会有以下三种情况：
1. matmulN
矩阵乘切N，即右矩阵的列

2. matmulK
矩阵乘切K，即左矩阵的列，右矩阵的行

3. matmulFull
不切分直接计算

## 5. 特殊逻辑:
1. full_load=True: Q 矩阵在模拟开始前全部加载到 L1，后续不再触发 Q 的 MTE2。
2. q_data_type=DataType.FP16/DataType.FP8，输入Q矩阵的数据类型，影响C1的Q搬运和QK计算。
3. kv_data_type=DataType.FP16/DataType.FP8，输入K矩阵的数据类型，影响C1C2的K/P/V搬运和PV计算。
4. L0C的得到的矩阵乘结果，即fixpip搬运的数据恒为FP32。
5. is_l2cache=True：相同基本块第一次加载使用DRAM带宽，下次加载使用L2带宽。不使能时每次均为DRAM带宽。

## 6. 可视化增强

### Y轴布局 (从下到上)
```
VECTOR_V2 (橙色 #E59866)  ■■■ O11 ■■■ O12
VECTOR_V1 (黄色 #F7DC6F)  ■■■ P11 ■■■ P12
MTE3      (灰色 #95A5A6)  ■ P11 ■ P12
FIXPIPE   (绿色 #96CEB4)  ■ P11/O11
MAC       (蓝色 #45B7D1)  ■■ P11/O11
L0B       (青色 #4ECDC4)  ■ K1/V1
L0A       (青色 #4ECDC4)  ■ Q1
L1B       (红橙)          ■■ K1/V1
L1A       (红橙)          ■■ Q1
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