# C1V1C2V2 完整流水线实现 - 最终总结

## 🎯 实现目标

实现昇腾NPU上Flash Attention算子的完整C1V1C2V2流水线建模。

## ✅ 已完成的工作

### 1. 完整流水线实现

```
C1阶段: Q @ K^T → P (Attention Score)
  ├─ MTE2: 加载Q和K
  ├─ MTE1: 搬运到L0
  ├─ MAC: 矩阵乘法
  └─ FIXPIPE: 搬P到UB

V1阶段: Softmax等操作
  └─ VECTOR_V1: 处理P矩阵 (800 cycles)

转换阶段:
  ├─ MTE3: P从UB搬回CUBE (新增)
  ├─ MTE2: 并行加载V矩阵
  └─ MTE1: V搬到L0

C2阶段: P @ V → O (Output)
  ├─ MAC: 矩阵乘法
  └─ FIXPIPE: 搬O到UB

V2阶段: 后处理
  └─ VECTOR_V2: 处理O矩阵 (800 cycles)
```

### 2. 新增硬件单元

- **MTE3**: UB→CUBE搬运，256 bytes/cycle
- **VECTOR_V1**: C1后的Softmax操作
- **VECTOR_V2**: C2后的后处理操作

### 3. V矩阵加载

- 支持V矩阵通过MTE2/MTE1加载
- DN模式自动路径切换
- 与MTE3并行执行

### 4. 标注格式优化

| 阶段 | 单元 | 标注格式 | 示例 |
|-----|------|---------|------|
| C1 | MAC/FIXPIPE | P矩阵 | P11, P12, P21 |
| V1 | VECTOR_V1 | P矩阵 | P11, P12 |
| V加载 | MTE2/MTE1 | V矩阵 | V1, V2 |
| C2 | MAC/FIXPIPE | O矩阵 | O11, O12, O21 |
| V2 | VECTOR_V2 | O矩阵 | O11, O12 |

**去除了VEC前缀**，Vector操作直接显示P或O。

### 5. 可视化增强

#### Y轴布局 (从下到上)
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

#### 颜色方案
- MTE2/MTE1: 红色/青色
- MTE3: 灰色 (新增)
- MAC: 蓝色
- FIXPIPE: 绿色
- VECTOR_V1: 黄色
- VECTOR_V2: 橙色 (新增)

### 6. DN模式支持

**标准模式路径**:
- Q → L1A → L0A
- K → L1B → L0B
- V → L1B → L0B

**DN模式路径**:
- Q → L1B → L0B
- K → L1A → L0A
- V → L1A → L0A

## 📊 性能测试结果

### 小规模测试 (256x512)
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

### 大规模测试 (512x2048)
```
总周期数: 385,197
MTE2:      69.0% (瓶颈)
MAC:       34.0%
VECTOR_V1: 6.6%
VECTOR_V2: 6.6%
```

## 🧪 验证方法

### 运行测试
```bash
# 完整流水线测试
python examples/test_full_pipeline.py

# 所有配置测试
python examples/run_c1_examples.py
```

### 关键验证点
- ✅ C1阶段P矩阵生成和标注
- ✅ VECTOR_V1处理P
- ✅ MTE3搬P回CUBE
- ✅ V矩阵并行加载
- ✅ C2阶段O矩阵生成和标注
- ✅ VECTOR_V2处理O
- ✅ 两个Vector阶段分别显示

## 📁 项目结构

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
    └── FINAL_SUMMARY.md          本文档
```

## 🔧 关键代码修改

### modelers/c1_modeler.py
```python
# 新增MTE3计算
def _calc_mte3_cycles(self, size_bytes: int) -> float:
    return size_bytes / 256.0

# 分离VECTOR_V1和VECTOR_V2
resource_free_time["VECTOR_V1"] = 0.0
resource_free_time["VECTOR_V2"] = 0.0

# C1阶段 → V1阶段 → MTE3搬运 → V加载 → C2阶段 → V2阶段
```

### utils/visualizer.py
```python
# 新增颜色
COLORS = {
    "MTE3": "#95A5A6",
    "VECTOR_V1": "#F7DC6F",
    "VECTOR_V2": "#E59866"
}

# 更新Y轴
Y_MAP = {
    "VECTOR_V2": 0,
    "VECTOR_V1": 0.5,
    "MTE3": 1,
    ...
}

# 标注逻辑
if event.operation == "P":
    label = f"P{q_idx}{k_idx}"
elif event.operation == "O":
    label = f"O{q_idx}{k_idx}"
```

## 📈 性能优化建议

1. **MTE2瓶颈** (最常见)
   - 启用L2缓存
   - Full Load预加载Q
   - 增加块大小

2. **MAC瓶颈**
   - 优化矩阵分块
   - 使用FP8数据类型

3. **流水线优化**
   - 双缓冲 (L1_db=True, L0_db=True)
   - DN模式提高并行度

## 🎓 使用示例

```python
from modelers import C1Modeler
from core import DataType

modeler = C1Modeler(
    s1_total=512,
    s2_total=2048,
    d_total=256,
    data_type=DataType.FP16,
    use_dn=False,
    L1_db=True,
    L0_db=True
)

timeline, _, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "output.png")
```

## 🔍 技术细节

### 时间计算公式

| 单元 | 公式 |
|-----|------|
| MTE2 | size_bytes / MTE2_BYTES_PER_CYCLE |
| MTE1 | size_bytes / 256 |
| MTE3 | size_bytes / 256 |
| MAC(C1) | s1 * s2 * d * 2 / THROUGHPUT |
| MAC(C2) | s1 * d * s2 * 2 / THROUGHPUT |
| FIXPIPE | size_bytes / 256 |
| VECTOR | 800 (固定) |

### 矩阵大小

| 矩阵 | 维度 | 大小 (FP16) |
|-----|------|-------------|
| Q | s1 × d | s1 * d * 2 bytes |
| K | s2 × d | s2 * d * 2 bytes |
| P | s1 × s2 | s1 * s2 * 2 bytes |
| V | s2 × d | s2 * d * 2 bytes |
| O | s1 × d | s1 * d * 2 bytes |

## 📝 下一步扩展

1. ✅ 完整C1V1C2V2流水线 (已完成)
2. ⏳ 参数化Vector周期数
3. ⏳ 支持不同的Softmax实现
4. ⏳ 多头注意力建模
5. ⏳ 内存优化分析

## 🎉 总结

成功实现了Flash Attention算子的完整C1V1C2V2流水线建模，包括：

- ✅ 四阶段完整流水线
- ✅ MTE3和双Vector单元
- ✅ V矩阵加载支持
- ✅ DN模式自动适配
- ✅ 优化的可视化
- ✅ 清晰的标注格式

所有测试通过，文档完善，可直接使用！
