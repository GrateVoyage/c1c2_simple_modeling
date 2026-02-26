# 更新日志 - V3: Vector周期优化 + Preload模式

## 更新内容

### 1. Vector周期数优化

根据实际硬件测量调整Vector操作周期：

- **VECTOR_V1** (Softmax): 800 → **1600 cycles**
- **VECTOR_V2** (后处理): 800 → **400 cycles**

**修改代码**:
```python
def _calc_vector_v1_cycles(self) -> float:
    """V1: Softmax等操作"""
    return 1600.0

def _calc_vector_v2_cycles(self) -> float:
    """V2: 后处理操作"""
    return 400.0
```

### 2. 紧凑Y轴布局

优化可视化Y轴间距，使图表更紧凑：

**之前** (0.5均匀间距):
```python
Y_MAP = {
    "VECTOR_V2": 0,
    "VECTOR_V1": 0.5,
    "MTE3": 1,
    "FIXPIPE": 1.5,
    "MAC": 2,
    "L0B": 2.5,
    "L0A": 3,
    "L1B": 3.5,
    "L1A": 4
}
```

**现在** (紧凑间距):
```python
Y_MAP = {
    "VECTOR_V2": 0,      # 0.3间距
    "VECTOR_V1": 0.3,    # 0.5间距
    "MTE3": 0.8,         # 0.5间距
    "FIXPIPE": 1.3,      # 0.5间距
    "MAC": 1.8,          # 0.5间距
    "L0B": 2.3,          # 0.3间距
    "L0A": 2.6,          # 0.5间距
    "L1B": 3.1,          # 0.3间距
    "L1A": 3.4
}
```

### 3. Preload模式

新增`preload`参数，支持两种执行模式：

**参数**:
- `preload=0` (默认): 正常模式，C1V1C2V2连续流水线执行
- `preload=1`: Preload模式，先执行所有C1+V1，再执行所有C2+V2

**实现**:
```python
class C1Modeler:
    def __init__(self, ..., preload: int = 0):
        self.preload = preload

    def run_simulation(self):
        if self.preload == 1:
            # 阶段1: 所有C1+V1
            for q_idx in range(self.q_block_count):
                for k_idx in range(self.k_block_count):
                    self._process_c1_stage(...)

            # 阶段2: 所有C2+V2
            for q_idx in range(self.q_block_count):
                for k_idx in range(self.k_block_count):
                    self._process_c2_stage(...)
        else:
            # 正常流水线
            for q_idx in range(self.q_block_count):
                self._process_k_blocks(...)  # C1V1C2V2连续
```

**新增方法**:
- `_process_c1_stage()`: 只执行C1+V1阶段
- `_process_c2_stage()`: 只执行C2+V2阶段

## 性能对比

### 测试配置
- 矩阵: 256x512, d=128
- 块大小: Q=128, K=256, d=128
- 无双缓冲

### Preload=0 (正常模式)
```
总周期数: 32,991 cycles
MTE2:      54.9% (瓶颈)
VECTOR_V1: 19.4% (1600 cycles × 4 = 6400)
VECTOR_V2: 4.8%  (400 cycles × 4 = 1600)
```

**执行顺序**: C1→V1→C2→V2→C1→V1→C2→V2...

### Preload=1 (Preload模式)
```
总周期数: 21,471 cycles ✓ 提升35%
MTE2:      84.4% (瓶颈)
VECTOR_V1: 29.8%
VECTOR_V2: 7.5%
```

**执行顺序**: C1→C1→C1→C1→V1→V1→V1→V1→C2→C2→C2→C2→V2→V2→V2→V2

**优势**: 批量执行相同操作，减少流水线切换，提高硬件利用率

## 验证结果

### Vector周期数验证
```
✓ VECTOR_V1: 1600 cycles (4个事件 = 6400 total)
✓ VECTOR_V2: 400 cycles (4个事件 = 1600 total)
```

### Preload模式验证
```
最后C1完成: 11,862.8 cycles
首个C2开始: 13,109.0 cycles
✓ C2确实在所有C1完成后执行
```

### Y轴布局验证
```
✓ 更紧凑的间距
✓ 所有标签清晰可见
✓ 图表高度减小约15%
```

## 使用示例

```python
from modelers import C1Modeler
from core import DataType

# Preload模式
modeler = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    data_type=DataType.FP16,
    use_dn=False,
    L1_db=True,
    L0_db=True,
    preload=1  # ← 启用Preload模式
)

timeline, _, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles)
```

## 测试命令

```bash
# 完整Preload测试
python examples/test_preload.py

# 对比两种模式
python examples/run_c1_examples.py
```

## 文件修改

### 修改的文件
- `modelers/c1_modeler.py`
  - 新增 `preload` 参数
  - 拆分 `_process_c1_stage()` 和 `_process_c2_stage()`
  - 更新 `_calc_vector_v1_cycles()` 和 `_calc_vector_v2_cycles()`

- `utils/visualizer.py`
  - 更新 `Y_MAP` 为紧凑间距

- `modelers/templates/standard.py`
  - 添加 `preload` 参数

### 新增的文件
- `examples/test_preload.py` - Preload模式测试

## 性能建议

**何时使用Preload模式**:

✅ **推荐使用** (preload=1):
- 批量计算场景
- 内存带宽充足
- 希望最大化硬件利用率
- 对延迟不敏感

❌ **不推荐** (preload=0):
- 实时推理场景
- 内存受限
- 需要低延迟
- 流式处理

## 总结

三项优化全部完成：

1. ✅ Vector周期数更新 (V1=1600, V2=400)
2. ✅ Y轴紧凑布局
3. ✅ Preload模式 (提升35%性能)

所有测试通过，可视化正常，文档完善！
