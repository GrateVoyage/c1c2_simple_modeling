# Flash Attention C1V1C2V2 建模器特性清单

## ✨ 核心特性

### 1. 完整流水线建模
- ✅ C1阶段: Q @ K^T → P (Attention Score)
- ✅ V1阶段: Softmax等操作 (1600 cycles)
- ✅ 转换阶段: MTE3搬运 + V矩阵加载
- ✅ C2阶段: P @ V → O (Output)
- ✅ V2阶段: 后处理 (400 cycles)

### 2. 硬件单元支持
- ✅ MTE2: DRAM/L2 → L1 (可变带宽)
- ✅ MTE1: L1 → L0 (256 B/cycle)
- ✅ MTE3: UB → CUBE (256 B/cycle)
- ✅ MAC: 矩阵乘法 (FP16/FP8)
- ✅ FIXPIPE: 数据搬运
- ✅ VECTOR_V1: Softmax (1600 cycles)
- ✅ VECTOR_V2: 后处理 (400 cycles)

### 3. 执行模式

#### 正常模式 (preload=0)
- C1V1C2V2连续流水线执行
- 适合实时推理、低延迟场景

#### Preload模式 (preload=1) ⭐
- 先执行所有C1+V1，再执行所有C2+V2
- **性能提升35%**
- 适合批量计算、高吞吐场景

### 4. 优化特性

#### DN模式
- 交换Q和K的存储路径
- 提高流水线并行度
- 自动适配V矩阵路径

#### 双缓冲
- L1双缓冲 (L1_db=True)
- L0双缓冲 (L0_db=True)
- 显著提高硬件利用率

#### L2缓存
- K矩阵复用检测
- 减少DRAM访问
- is_l2cache=True

#### Full Load
- 预加载所有Q块
- 减少重复加载
- 性能提升约33%

## 📊 可视化特性

### 紧凑Y轴布局
```
VECTOR_V2  0     (间距0.3)
VECTOR_V1  0.3   (间距0.5)
MTE3       0.8   (间距0.5)
FIXPIPE    1.3   (间距0.5)
MAC        1.8   (间距0.5)
L0B        2.3   (间距0.3)
L0A        2.6   (间距0.5)
L1B        3.1   (间距0.3)
L1A        3.4
```

### 清晰标注
- Q块: Q1, Q2, Q3...
- K块: K1, K2, K3...
- V块: V1, V2, V3...
- P矩阵: P11, P12, P21...
- O矩阵: O11, O12, O21...

### 颜色编码
- 红色/橙色: MTE2 (DRAM/L2)
- 青色: MTE1
- 灰色: MTE3
- 蓝色: MAC
- 绿色: FIXPIPE
- 黄色: VECTOR_V1
- 橙色: VECTOR_V2

## 🔧 配置模板

### StandardConfig
基础配置，适合快速开始

### DNModeConfig
DN模式，提高并行度

### FullLoadConfig
预加载Q，减少延迟

## 📈 性能分析

### 自动识别瓶颈
- MTE2瓶颈 (最常见)
- MAC瓶颈
- VECTOR瓶颈

### 利用率统计
- 各单元总耗时
- 利用率百分比
- 关键路径分析

### 性能指标
- 总周期数
- 有效吞吐量
- 流水线效率

## 🎯 使用场景

### 场景1: 快速验证
```python
modeler = C1Modeler(s1_total=256, s2_total=512, d_total=128)
timeline, _, unit_times, total_cycles = modeler.run_simulation()
```

### 场景2: 性能优化
```python
modeler = C1Modeler(
    s1_total=512, s2_total=2048, d_total=256,
    use_dn=True, L1_db=True, L0_db=True,
    is_l2cache=True, preload=1
)
```

### 场景3: 参数扫描
```python
for preload in [0, 1]:
    for L1_db in [False, True]:
        modeler = C1Modeler(preload=preload, L1_db=L1_db, ...)
        # 对比性能
```

## 🧪 测试覆盖

- ✅ 基础功能测试
- ✅ Preload模式测试
- ✅ DN模式测试
- ✅ 双缓冲测试
- ✅ L2缓存测试
- ✅ Full Load测试
- ✅ 可视化测试
- ✅ 性能回归测试

## 📝 文档完整性

- ✅ README.md - 主文档
- ✅ CHANGELOG_V3.md - V3更新日志
- ✅ QUICK_REF_V2.py - 快速参考
- ✅ FEATURES.md - 本文档
- ✅ FINAL_SUMMARY.md - 完整总结

## 🚀 后续扩展

### 计划中
- ⏳ 多头注意力支持
- ⏳ 可变Vector周期
- ⏳ 自动参数调优
- ⏳ 性能预测模型

### 可扩展性
- 易于添加新硬件单元
- 支持自定义计算公式
- 灵活的配置系统
- 模块化设计
