# C1V1C2V2 完整流水线更新日志

## 最新更新 - 完整Flash Attention流水线实现

### 新增功能

#### 1. 完整的C1V1C2V2流水线

实现了Flash Attention算子的完整四阶段流水线：

```
C1: Q @ K^T → S (Attention Score)
V1: Softmax S → P (Attention Probability)
C2: P @ V → O (Output)
V2: 后处理
```

#### 2. 新增硬件单元

**MTE3单元**
- 功能: 从UB搬运P矩阵回CUBE
- 带宽: 256 bytes/cycle
- 时机: V1阶段完成后

**VECTOR_V1和VECTOR_V2分离**
- VECTOR_V1: C1阶段后的Softmax等操作
- VECTOR_V2: C2阶段后的后处理操作
- 在可视化中分别占据不同Y轴高度

#### 3. V矩阵加载流水线

- V矩阵通过MTE2加载到L1
- DN模式: V使用L1A路径
- 标准模式: V使用L1B路径
- MTE1搬运V到L0用于C2计算

#### 4. 标注格式优化

- **C1阶段**: MAC/FIXPIPE/VECTOR_V1显示P11, P12, P21, P22...
- **C2阶段**: MAC/FIXPIPE/VECTOR_V2显示O11, O12, O21, O22...
- **V矩阵加载**: MTE2/MTE1显示V1, V2...
- **去除VEC标注**: Vector操作只显示P或O，不再显示V111等格式

#### 5. 可视化增强

Y轴布局（从下到上）：
```
VECTOR_V2   (橙色) ■■■ O11 ■■■ O12
VECTOR_V1   (黄色) ■■■ P11 ■■■ P12
MTE3        (灰色) ■ P11 ■ P12
FIXPIPE     (绿色) ■ P11/O11 ■ P12/O12
MAC         (蓝色) ■■ P11/O11 ■■ P12/O12
L0B         (青色) ■ K1/V1 ■ K2/V2
L0A         (青色) ■ Q1 ■ Q2
L1B         (红色) ■■ K1/V1 ■■ K2/V2
L1A         (红色) ■■ Q1 ■■ Q2
```

#### 6. 流水线依赖关系

```
C1阶段:
  MTE2(Q,K) → MTE1 → MAC → FIXPIPE → VECTOR_V1
                      ↓
                     P矩阵

转换阶段:
  VECTOR_V1 → MTE3(P回CUBE)
           ↘
             MTE2(V) → MTE1

C2阶段:
  MTE3 + MTE1(V) → MAC → FIXPIPE → VECTOR_V2
                    ↓
                   O矩阵
```

### 性能统计示例

```
=== 性能分析 ===
总周期数: 29679.4
MTE2       总耗时: 18127.4  (利用率: 61.1%)
MTE1       总耗时: 2304.0   (利用率: 7.8%)
MTE3       总耗时: 1024.0   (利用率: 3.5%)  ← 新增
MAC        总耗时: 8192.0   (利用率: 27.6%)
FIXPIPE    总耗时: 1536.0   (利用率: 5.2%)
VECTOR_V1  总耗时: 3200.0   (利用率: 10.8%) ← 分离
VECTOR_V2  总耗时: 3200.0   (利用率: 10.8%) ← 分离
瓶颈分析: MTE2
```

### 关键事件示例

```
MAC        P      P11      (C1阶段)
FIXPIPE    P      P11
VECTOR_V1  P      P11
MTE3       P      P11      (搬回CUBE)
MTE2       Load   V1       (加载V矩阵)
MTE1       Load   V1
MAC        O      O11      (C2阶段)
FIXPIPE    O      O11
VECTOR_V2  O      O11
```

### 测试验证

运行完整流水线测试：
```bash
python examples/test_full_pipeline.py
```

检查要点：
- ✓ C1阶段产生P矩阵，标注为P11, P12...
- ✓ VECTOR_V1处理P矩阵
- ✓ MTE3搬P回CUBE
- ✓ 并行加载V矩阵
- ✓ C2阶段产生O矩阵，标注为O11, O12...
- ✓ VECTOR_V2处理O矩阵
- ✓ 两个Vector阶段在不同高度显示

### 向后兼容

所有原有配置和示例仍然可用，输出格式已更新为完整流水线。

## 技术细节

### DN模式下的路径变化

**标准模式**:
- Q → L1A → L0A
- K → L1B → L0B
- V → L1B → L0B (复用K路径)

**DN模式**:
- Q → L1B → L0B
- K → L1A → L0A
- V → L1A → L0A (复用K路径)

### 时间计算公式

- MTE2: `size_bytes / MTE2_BYTES_PER_CYCLE`
- MTE1: `size_bytes / 256`
- MTE3: `size_bytes / 256`
- MAC (C1): `s1 * s2 * d * 2 / MAC_THROUGHPUT`
- MAC (C2): `s1 * d * s2 * 2 / MAC_THROUGHPUT`
- FIXPIPE: `size_bytes / 256`
- VECTOR: `800 cycles` (固定)
