"""
项目重构与Vector流水线总结
"""

# ============================================================
# 第一阶段：项目重构
# ============================================================

原始结构:
    single_mm_modeling.py  (单文件，约387行)

重构后结构:
    core/                   - 核心定义
    ├── enums.py           - 枚举类型
    ├── dataclasses.py     - 数据结构
    └── hardware_config.py - 硬件配置

    modelers/              - 建模器
    ├── c1_modeler.py      - C1建模器
    └── templates/         - 配置模板
        ├── standard.py    - 标准配置
        ├── dn_mode.py     - DN模式
        └── full_load.py   - Full Load

    utils/                 - 工具模块
    ├── visualizer.py      - 可视化
    └── logger.py          - 日志

    examples/              - 示例代码
    ├── run_c1_examples.py - 完整示例
    └── test_vector.py     - 测试脚本

重构优势:
    ✓ 模块化设计，职责清晰
    ✓ 易于扩展 (可添加C2建模器)
    ✓ 配置模板化，方便复用
    ✓ 保持向后兼容

# ============================================================
# 第二阶段：新增Vector流水线
# ============================================================

Flash Attention算子结构:
    C1: Q @ K^T → S (Attention Score)
    V1: Softmax等操作 S → P (Attention Probability)  ← 本次新增
    C2: P @ V → O (Output)
    V2: 后处理操作

新增功能:
    1. Vector流水线单元
       - 固定周期: 800 cycles
       - 在FIXPIPE后执行
       - 可与其他流水线并行

    2. 标注格式优化
       - MAC: "Q1K1" → "P11"
       - FIXPIPE: "Out" → "P11"
       - VECTOR: "V111"

    3. 可视化增强
       - Y轴布局: VECTOR(底部) → FIXPIPE → MAC → L0 → L1(顶部)
       - 黄色表示Vector操作
       - 清晰展示流水线依赖

代码修改:
    modelers/c1_modeler.py:
        + def _calc_vector_cycles(self) -> float: return 800.0
        + resource_free_time["VECTOR"] = 0.0
        + TimelineEvent("VECTOR", "V1", ...)

    utils/visualizer.py:
        + COLORS["VECTOR"] = "#F7DC6F"
        + Y_MAP["VECTOR"] = 0
        + 标注逻辑支持P和V1

测试验证:
    ✓ 所有示例运行正常
    ✓ Vector在FIXPIPE后执行
    ✓ 标注格式正确 (P11, V111)
    ✓ 性能统计包含VECTOR
    ✓ 图表显示正确

# ============================================================
# 使用示例
# ============================================================

from modelers import C1Modeler
from modelers.templates import DNModeConfig

# 使用配置模板
config = DNModeConfig(s1_total=512, s2_total=2048, d_total=256)
modeler = C1Modeler(**config.to_dict())
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles)

# 输出示例:
# === 性能分析 ===
# 总周期数: 140578.9
# MTE2       总耗时: 136962.9 (利用率: 97.4%)
# MTE1       总耗时: 17408.0  (利用率: 12.4%)
# MAC        总耗时: 65536.0  (利用率: 46.6%)
# FIXPIPE    总耗时: 8192.0   (利用率: 5.8%)
# VECTOR     总耗时: 25600.0  (利用率: 18.2%)
# 瓶颈分析: MTE2

# ============================================================
# 后续扩展建议
# ============================================================

1. 添加C2建模器 (modelers/c2_modeler.py)
   - 实现 P @ V → O
   - 支持V矩阵加载
   - 整合C1和C2的流水线

2. 添加C1C2完整流水线 (modelers/c1c2_modeler.py)
   - C1 → V1 → C2 → V2 完整流程
   - 支持流水线重叠优化
   - 考虑不同优化策略

3. 参数化Vector周期数
   - 目前固定800 cycles
   - 可根据实际硬件测试调整
   - 支持不同算子的Vector耗时

4. 添加更多配置模板
   - L2 Cache优化配置
   - 大矩阵优化配置
   - FP8数据类型优化配置

5. 性能优化分析
   - 自动识别瓶颈
   - 给出优化建议
   - 参数调优工具
