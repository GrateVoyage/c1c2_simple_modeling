# Flash Attention C1V1 建模更新日志

## 最新更新

### 新增 Vector 流水线 (V1阶段)

根据FA算子设计文档，Flash Attention包含C1V1C2V2四个阶段。当前版本实现了C1和V1阶段的完整建模。

#### 主要变更

1. **新增Vector流水线**
   - 在FIXPIPE之后执行
   - 固定周期数: 800 cycles
   - 模拟Softmax等向量操作
   - 可与其他流水线并行

2. **标注格式更新**
   - MAC输出: 从"Q1K1"改为"P11"
   - FIXPIPE输出: 从"Out"改为"P11"
   - Vector操作: 显示为"V111"

3. **可视化改进**
   - Y轴新增VECTOR行（最下方）
   - 颜色: 黄色 (#F7DC6F)
   - 清晰显示V1阶段的执行时间

4. **性能统计**
   - 新增VECTOR单元利用率统计
   - 支持识别VECTOR为瓶颈

#### 示例输出

```
=== 性能分析 ===
总周期数: 140578.9
MTE2       总耗时: 136962.9 (利用率: 97.4%)
MTE1       总耗时: 17408.0  (利用率: 12.4%)
MAC        总耗时: 65536.0  (利用率: 46.6%)
FIXPIPE    总耗时: 8192.0   (利用率: 5.8%)
VECTOR     总耗时: 25600.0  (利用率: 18.2%)  ← 新增
瓶颈分析: MTE2
```

#### 时间线图表

Y轴布局（从下到上）：
- VECTOR (V1阶段)
- FIXPIPE
- MAC
- L0B / L0A
- L1B / L1A

#### 流水线依赖关系

```
MTE2 → MTE1 → MAC → FIXPIPE → VECTOR
                              (V1阶段)
```

VECTOR必须等待FIXPIPE完成，但可以与下一轮的MTE2/MTE1/MAC并行执行。

#### 测试验证

运行测试脚本验证功能：
```bash
python examples/test_vector.py
```

检查要点：
- ✓ MAC的operation为'P'
- ✓ FIXPIPE的operation为'P'
- ✓ VECTOR的operation为'V1'
- ✓ VECTOR在FIXPIPE之后执行
- ✓ 图表显示P11, V111等标注

#### 向后兼容

原有代码完全兼容，所有示例和测试正常运行。
