"""
完整Flash Attention C1V1C2V2流水线 - 快速参考
"""

# ============================================================
# Flash Attention 算子完整流程
# ============================================================

"""
C1: Q @ K^T → S (Attention Score) - 第一次矩阵乘法
V1: Softmax S → P (Attention Probability) - 第一次向量操作
C2: P @ V → O (Output) - 第二次矩阵乘法
V2: 后处理 - 第二次向量操作
"""

# ============================================================
# 流水线详细步骤
# ============================================================

"""
【C1阶段】生成Attention Score矩阵P
1. MTE2: 加载Q到L1A, K到L1B (DN模式相反)
2. MTE1: 搬Q到L0A, K到L0B
3. MAC:  计算 Q @ K^T → P矩阵 (标注: P11, P12...)
4. FIXPIPE: 搬P到UB

【V1阶段】Softmax处理
5. VECTOR_V1: Softmax等操作 (800 cycles, 标注: P11)

【转换阶段】准备C2
6. MTE3: 搬P从UB回到CUBE (256 B/cycle)
7. MTE2: 并行加载V矩阵到L1B (DN模式为L1A)
8. MTE1: 搬V到L0B

【C2阶段】生成最终输出O
9. MAC:  计算 P @ V → O矩阵 (标注: O11, O12...)
10. FIXPIPE: 搬O到UB

【V2阶段】后处理
11. VECTOR_V2: 后处理操作 (800 cycles, 标注: O11)
"""

# ============================================================
# 硬件单元说明
# ============================================================

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
        "带宽": "256 bytes/cycle",
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

# ============================================================
# DN模式路径映射
# ============================================================

"""
标准模式:
  Q → L1A → L0A
  K → L1B → L0B
  V → L1B → L0B (复用K路径)

DN模式:
  Q → L1B → L0B
  K → L1A → L0A
  V → L1A → L0A (复用K路径)
"""

# ============================================================
# 可视化Y轴布局
# ============================================================

Y_AXIS_LAYOUT = """
从下到上:

VECTOR_V2  ■■■ O11  ■■■ O12  (橙色 #E59866)
VECTOR_V1  ■■■ P11  ■■■ P12  (黄色 #F7DC6F)
MTE3       ■ P11 ■ P12        (灰色 #95A5A6)
FIXPIPE    ■ P11/O11          (绿色 #96CEB4)
MAC        ■■ P11/O11         (蓝色 #45B7D1)
L0B        ■ K1/V1            (青色 #4ECDC4)
L0A        ■ Q1               (青色 #4ECDC4)
L1B        ■■ K1/V1           (红色/橙色)
L1A        ■■ Q1              (红色/橙色)
"""

# ============================================================
# 标注格式
# ============================================================

LABEL_FORMAT = {
    "Q块": "Q1, Q2, Q3...",
    "K块": "K1, K2, K3...",
    "V块": "V1, V2, V3...",
    "P矩阵": "P11, P12, P21, P22... (C1和V1阶段)",
    "O矩阵": "O11, O12, O21, O22... (C2和V2阶段)",
}

# ============================================================
# 使用示例
# ============================================================

from modelers import C1Modeler
from modelers.templates import StandardConfig
from core import DataType

# 创建建模器
modeler = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    s1_base_size=128,
    s2_base_size=256,
    d_base_size=128,
    data_type=DataType.FP16,
    use_dn=False,
    L1_db=True,
    L0_db=True
)

# 运行模拟
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()

# 输出性能
modeler.print_performance(unit_times, total_cycles)

# 生成图表
modeler.plot_timeline(timeline, unit_times, total_cycles, "output.png")

# ============================================================
# 性能输出示例
# ============================================================

"""
=== 性能分析 ===
总周期数: 29679.4

MTE2       总耗时: 18127.4  (利用率: 61.1%) ← 瓶颈
MTE1       总耗时: 2304.0   (利用率: 7.8%)
MTE3       总耗时: 1024.0   (利用率: 3.5%)  ← 新增
MAC        总耗时: 8192.0   (利用率: 27.6%)
FIXPIPE    总耗时: 1536.0   (利用率: 5.2%)
VECTOR_V1  总耗时: 3200.0   (利用率: 10.8%) ← V1阶段
VECTOR_V2  总耗时: 3200.0   (利用率: 10.8%) ← V2阶段

瓶颈分析: MTE2
"""

# ============================================================
# 关键事件流
# ============================================================

EVENT_FLOW_EXAMPLE = """
事件序列 (Q1K1为例):

C1阶段:
  MTE2  Load L1A (Q1)    ← 加载Q
  MTE2  Load L1B (K1)    ← 加载K
  MTE1  Load L0A (Q1)    ← Q到L0
  MTE1  Load L0B (K1)    ← K到L0
  MAC   P      P11       ← 矩阵乘 Q@K^T
  FIXPIPE P    P11       ← P到UB

V1阶段:
  VECTOR_V1 P  P11       ← Softmax处理

转换阶段:
  MTE3  P      P11       ← P回CUBE
  MTE2  Load L1B (V1)    ← 加载V (并行)
  MTE1  Load L0B (V1)    ← V到L0

C2阶段:
  MAC   O      O11       ← 矩阵乘 P@V
  FIXPIPE O    O11       ← O到UB

V2阶段:
  VECTOR_V2 O  O11       ← 后处理
"""

# ============================================================
# 测试命令
# ============================================================

"""
# 运行完整流水线测试
python examples/test_full_pipeline.py

# 运行所有配置示例
python examples/run_c1_examples.py

# 向后兼容测试
python single_mm_modeling.py
"""

# ============================================================
# 性能优化建议
# ============================================================

OPTIMIZATION_TIPS = """
1. MTE2瓶颈 (最常见):
   - 启用L2缓存: is_l2cache=True
   - 增加块大小以提高计算强度
   - 考虑Full Load预加载Q

2. MAC瓶颈:
   - 检查矩阵维度是否合理
   - 考虑使用FP8提高吞吐

3. VECTOR瓶颈:
   - 检查Vector操作周期数设置
   - 优化Softmax实现

4. 流水线优化:
   - 启用L1和L0双缓冲
   - 利用DN模式提高并行度
"""

# ============================================================
# 文件说明
# ============================================================

FILES = {
    "README.md": "项目说明文档",
    "CHANGELOG_V2.md": "C1V1C2V2更新日志",
    "QUICK_REF_V2.py": "本文档 - 完整流水线参考",

    "examples/test_full_pipeline.py": "完整流水线测试",
    "examples/run_c1_examples.py": "配置示例集",

    "modelers/c1_modeler.py": "主建模器实现",
    "modelers/templates/": "配置模板",

    "utils/visualizer.py": "可视化工具",
    "core/": "核心定义"
}
