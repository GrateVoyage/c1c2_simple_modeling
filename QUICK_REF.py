"""
快速参考指南
"""

# ============================================================
# 1. 快速开始
# ============================================================

# 方法1: 使用配置模板 (推荐)
from modelers import C1Modeler
from modelers.templates import StandardConfig

config = StandardConfig(s1_total=512, s2_total=2048, d_total=256)
modeler = C1Modeler(**config.to_dict())
timeline, _, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "output.png")

# 方法2: 直接实例化
modeler = C1Modeler(
    s1_total=512,
    s2_total=2048,
    d_total=256,
    use_dn=True,
    L1_db=True,
    L0_db=True
)

# ============================================================
# 2. 配置模板
# ============================================================

from modelers.templates import StandardConfig, DNModeConfig, FullLoadConfig

# 标准配置
StandardConfig()

# DN模式 (L1A/L0A存储K, L1B/L0B存储Q)
DNModeConfig()

# Full Load (预加载所有Q块)
FullLoadConfig()

# ============================================================
# 3. 参数说明
# ============================================================

C1Modeler(
    # 矩阵维度
    s1_total=256,        # Q矩阵序列长度
    s2_total=1024,       # K矩阵序列长度
    d_total=128,         # 特征维度

    # 块大小
    s1_base_size=128,    # Q块大小
    s2_base_size=256,    # K块大小
    d_base_size=128,     # D块大小

    # 数据类型
    data_type=DataType.FP16,  # FP16 或 FP8

    # 流水线配置
    use_dn=False,        # DN模式
    L1_db=False,         # L1双缓冲
    L0_db=False,         # L0双缓冲
    is_l2cache=False,    # L2缓存
    full_load=False,     # Q矩阵预加载
)

# ============================================================
# 4. 流水线阶段
# ============================================================

"""
Flash Attention算子完整流程:
    C1: Q @ K^T → S      (已实现)
    V1: Softmax等 S → P  (已实现)
    C2: P @ V → O        (待实现)
    V2: 后处理            (待实现)

当前C1V1流水线:
    MTE2 → MTE1 → MAC → FIXPIPE → VECTOR
     ↓      ↓      ↓       ↓         ↓
    L1     L0     P矩阵   UB       V1操作
                          (800 cycles)
"""

# ============================================================
# 5. 运行示例
# ============================================================

# 运行所有示例
$ python examples/run_c1_examples.py

# 运行Vector测试
$ python examples/test_vector.py

# 使用原始文件 (向后兼容)
$ python single_mm_modeling.py

# ============================================================
# 6. 输出解读
# ============================================================

"""
=== 性能分析 ===
总周期数: 140578.9

MTE2       总耗时: 136962.9 (利用率: 97.4%)  ← DRAM/L2带宽瓶颈
MTE1       总耗时: 17408.0  (利用率: 12.4%)  ← L1→L0搬运
MAC        总耗时: 65536.0  (利用率: 46.6%)  ← 矩阵乘法计算
FIXPIPE    总耗时: 8192.0   (利用率: 5.8%)   ← 数据搬运到UB
VECTOR     总耗时: 25600.0  (利用率: 18.2%)  ← V1向量操作

瓶颈分析: MTE2  ← 主要瓶颈

优化建议:
  - MTE2瓶颈: 启用L2缓存 (is_l2cache=True)
  - MAC瓶颈: 增加块大小，提高计算强度
  - VECTOR瓶颈: 检查向量操作优化
"""

# ============================================================
# 7. 时间线图表
# ============================================================

"""
Y轴布局 (从下到上):
  VECTOR   ■■■■■■  V111  ■■■■■■  V112      黄色
  FIXPIPE  ■■■  P11  ■■■  P12               绿色
  MAC      ■■■■■■  P11  ■■■■■■  P12        蓝色
  L0B      ■■  K1  ■■  K2                   青色
  L0A      ■  Q1  ■  Q2                     青色
  L1B      ■■■  K1  ■■■  K2                红色/橙色
  L1A      ■■  Q1  ■■  Q2                   红色/橙色

标注格式:
  Q块: Q1, Q2, Q3...
  K块: K1, K2, K3...
  P矩阵: P11, P12, P21, P22...
  Vector: V111, V112, V121...
  L2命中: (L2)
"""

# ============================================================
# 8. 自定义硬件配置
# ============================================================

from core import HardwareConfig

hw_config = HardwareConfig(
    CHIP_FREQ_GHZ=1.65,              # 芯片频率
    MTE2_DRAM_BW_GBPS=1600,          # DRAM带宽
    MTE2_L2_BW_GBPS=5400,            # L2带宽
    MTE1_FIXPIPE_BYTES_PER_CYCLE=256 # MTE1带宽
)

modeler = C1Modeler(..., hw_config=hw_config)

# ============================================================
# 9. 文件说明
# ============================================================

"""
README.md       - 项目说明文档
CHANGELOG.md    - Vector更新日志
SUMMARY.py      - 重构总结
USAGE.py        - 使用示例
QUICK_REF.py    - 本文档

examples/
  run_c1_examples.py  - 完整示例集
  test_vector.py      - Vector测试

single_mm_modeling.py - 原始文件 (向后兼容)
"""

# ============================================================
# 10. 常见问题
# ============================================================

# Q: 如何修改Vector的周期数?
# A: 修改 modelers/c1_modeler.py 中的 _calc_vector_cycles() 方法

# Q: 如何添加C2建模器?
# A: 在 modelers/ 下创建 c2_modeler.py，参考 c1_modeler.py 的结构

# Q: 导入错误怎么办?
# A: 确保在项目根目录运行，或调整 sys.path

# Q: 图表不显示怎么办?
# A: 检查matplotlib安装，或直接保存PNG文件

# Q: 如何理解DN模式?
# A: DN模式交换了Q和K的存储路径，L1A/L0A存K，L1B/L0B存Q
