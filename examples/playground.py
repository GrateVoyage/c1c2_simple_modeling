"""
Flash Attention C1V1C2V2 流水线建模 - 参数调试入口

修改下方参数后直接运行即可查看时间线和性能分析。
输出图片保存至 outputs/playground_timeline.png
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline, LoadOrder

# ============================================================
# 矩阵规模
# ============================================================
s1_total     = 128      # Q 的 sequence 长度（总行数），需被 s1_base_size 整除
s2_total     = 1024     # K/V 的 sequence 长度（总列数），需被 s2_base_size 整除
d_total      = 128      # 特征维度，需被 d_base_size 整除

# ============================================================
# 基本块大小（核内每次处理的分块大小）
# ============================================================
s1_base_size = 128      # Q 块行数
s2_base_size = 256      # K/V 块列数
d_base_size  = 128      # 特征维度块大小

# ============================================================
# 数据类型
# ============================================================
q_data_type  = DataType.FP16    # Q 矩阵数据类型:  DataType.FP16 / DataType.FP8
kv_data_type = DataType.FP16    # K/V 矩阵数据类型: DataType.FP16 / DataType.FP8

# ============================================================
# 矩阵乘切分（基本块内的 MAC 最大块尺寸）
#   * 若 s2_base_size > baseN_C1  → matmulN（沿 N 方向切，每子块独立 FIXPIPE）
#   * 若 d_base_size  > baseK_C1  → matmulK（沿 K 方向切，所有子块共用一次 FIXPIPE）
#   * 否则                        → matmulFull（不切分）
#   C2 同理，对应 P@V 计算
# ============================================================
baseM_C1 = 128      # C1 阶段 MAC 的 M 维最大块（对应 s1_base 方向）
baseN_C1 = 128      # C1 阶段 MAC 的 N 维最大块（对应 s2_base 方向）
baseK_C1 = 128      # C1 阶段 MAC 的 K 维最大块（对应 d_base  方向）

baseM_C2 = 128      # C2 阶段 MAC 的 M 维最大块
baseN_C2 = 128      # C2 阶段 MAC 的 N 维最大块（对应 d_base  方向）
baseK_C2 = 128      # C2 阶段 MAC 的 K 维最大块（对应 s2_base 方向）

# ============================================================
# 缓存与缓冲区特性
# ============================================================
is_l2cache = True  # True: K/V 首次从 DRAM 加载，后续 q_idx>0 复用 L2 带宽
                    # False: 每次均使用 DRAM 带宽

L1_db      = False  # True: L1 双缓冲（MTE2A/MTE2B 各两个槽位，可流水加载）
                    # False: 单槽位

L0_db      = True  # True: L0 双缓冲（MTE1A/MTE1B 各两个槽位，MTE1 与 MAC 可流水）
                    # False: 单槽位

full_load  = False  # True: 模拟开始前将 Q 矩阵全量加载到 L1，后续跳过 Q 的 MTE2
                    # False: Q 按需加载

# ============================================================
# DN 模式（交换 Q/K 的搬运路径）
#   标准: Q→MTE2A→MTE1A(L0A),  K/V→MTE2B→MTE1B(L0B),  P→MTE3→MTE2A→MTE1A
#   DN:   Q→MTE2B→MTE1B(L0B),  K/V→MTE2A→MTE1A(L0A),  P→MTE3→MTE2B→MTE1B
# ============================================================
use_dn = False      # True: 使能 DN 模式 / False: 标准模式

# ============================================================
# 加载顺序（当 L1_db=False 时有效）
# ============================================================
load_order = LoadOrder.LOAD_Q_FIRST     # LOAD_Q_FIRST: 先加载 Q 再加载 K
                                        # LOAD_K_FIRST: 先加载 K 再加载 Q

# ============================================================
# 核间流水线模式（不同 q_block 之间的调度策略）
#   DEFAULT:  C1V1C2V2 → C1V1C2V2 → ...（顺序执行，等上一个完全结束再开始下一个）
#   PRELOAD:  C1 → C1V1C2 → C1V1C2V2 → ... → V1C2V2 → C2V2 → V2
#             （渐进启动：下一个 C1 在当前 MAC 完成后立即发射，不等 V2）
#   N_BUFFER: C1C1 → V1V1 → C2C2 → V2V2（批量流水，N=2 组同时推进）
# ============================================================
inter_core_pipeline = InterCorePipeline.PRELOAD
# inter_core_pipeline = InterCorePipeline.PRELOAD
# inter_core_pipeline = InterCorePipeline.N_BUFFER
# inter_core_pipeline = InterCorePipeline.DEFAULT

# ============================================================
# 核内流水线（L1 缓存替换策略）
#   DEFAULT:    L1 总容量 512KB，平坦 LRU，所有块（Q/K/V/P）共享池
#               若块仍在 L1 中则跳过 MTE2，直接 MTE1
#   Q_RESIDENT: Q 永驻（1×144KB），KP 共享池（2×144KB LRU），V 池（2×32KB LRU）
# ============================================================
inner_core_pipeline = InnerCorePipeline.DEFAULT
# inner_core_pipeline = InnerCorePipeline.Q_RESIDENT
# inner_core_pipeline = InnerCorePipeline.DEFAULT

# ============================================================
# 运行模拟
# ============================================================
OUTPUT_FILE = "outputs/playground_timeline.png"

modeler = C1Modeler(
    s1_total=s1_total,
    s2_total=s2_total,
    d_total=d_total,
    s1_base_size=s1_base_size,
    s2_base_size=s2_base_size,
    d_base_size=d_base_size,
    q_data_type=q_data_type,
    kv_data_type=kv_data_type,
    baseM_C1=baseM_C1,
    baseN_C1=baseN_C1,
    baseK_C1=baseK_C1,
    baseM_C2=baseM_C2,
    baseN_C2=baseN_C2,
    baseK_C2=baseK_C2,
    is_l2cache=is_l2cache,
    use_dn=use_dn,
    L1_db=L1_db,
    L0_db=L0_db,
    load_order=load_order,
    full_load=full_load,
    inter_core_pipeline=inter_core_pipeline,
    inner_core_pipeline=inner_core_pipeline,
)

timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, OUTPUT_FILE)
