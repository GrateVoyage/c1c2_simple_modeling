"""
Flash Attention C1V1C2V2 流水线建模 — 快速参考脚本

直接运行此文件可看到各特性的简短演示：
  python QUICK_REF_V2.py

更完整的示例见 examples/ 目录。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

# ──────────────────────────────────────────────────────────────
# 参数速查
# ──────────────────────────────────────────────────────────────
#
# 矩阵维度
#   s1_total / s2_total / d_total       — Q/K/d 总长度
#   s1_base_size / s2_base_size / d_base_size  — 基本块大小
#
# 矩阵乘切分（超出 base 上限自动切分）
#   baseM_C1/baseN_C1/baseK_C1          — C1 (Q@K^T) 切分上限
#   baseM_C2/baseN_C2/baseK_C2          — C2 (P@V)   切分上限
#   切分规则:  K 轴超限 → matmulK（L0C 累加，单次 FIXPIPE）
#              N 轴超限 → matmulN（每子块独立 FIXPIPE）
#
# 数据类型
#   q_data_type  = DataType.FP16 / DataType.FP8   — Q 精度
#   kv_data_type = DataType.FP16 / DataType.FP8   — K/V 精度
#   MAC C1 吞吐：q 和 kv 同时 FP8 → FP8 吞吐，否则 FP16
#   MAC C2 吞吐：由 kv_data_type 决定
#   FIXPIPE 输出大小恒为 FP32（与数据类型无关）
#
# 核间流水（inter_core_pipeline）
#   DEFAULT    顺序: C1V1C2V2 → C1V1C2V2 → ...
#   PRELOAD_1  渐进(2WS): C1[k] 与前一轮 V1C2V2 并行；UB 2个64KB Workspace
#   PRELOAD_2  渐进(3WS): V1 紧跟 C1，C2 延迟 2 个 K；UB 3个64KB Workspace
#
# 存储策略
#   is_l2cache  重复加载同一块用 L2 带宽（否则用 DRAM）
#   use_dn      DN 模式（Q/K 路径互换）
#   L1_db       L1 双缓冲
#   L0_db       L0A/L0B 双缓冲
#   full_load   仿真开始前 Q 全量加载到 L1
#
# ──────────────────────────────────────────────────────────────


def demo_default():
    """DEFAULT 顺序流水 — 最基础用法"""
    modeler = C1Modeler(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16,
        kv_data_type=DataType.FP16,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    _, _, unit_times, total_cycles = modeler.run_simulation()
    print(f"  DEFAULT    总周期: {total_cycles:>10.1f}")
    return total_cycles


def demo_preload1():
    """PRELOAD_1 渐进式流水(2WS) — C1[k] 与前一轮 V1C2V2 并行"""
    modeler = C1Modeler(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16,
        kv_data_type=DataType.FP16,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.PRELOAD_1,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    _, _, unit_times, total_cycles = modeler.run_simulation()
    print(f"  PRELOAD_1  总周期: {total_cycles:>10.1f}")
    return total_cycles


def demo_preload2():
    """PRELOAD_2 渐进式流水(3WS) — V1 紧跟 C1，C2 延迟 2 个 K"""
    modeler = C1Modeler(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16,
        kv_data_type=DataType.FP16,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.PRELOAD_2,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    _, _, unit_times, total_cycles = modeler.run_simulation()
    print(f"  PRELOAD_2  总周期: {total_cycles:>10.1f}")
    return total_cycles


def demo_fp8():
    """FP8 数据类型 — 更高 MAC 吞吐"""
    modeler = C1Modeler(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP8,
        kv_data_type=DataType.FP8,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    _, _, unit_times, total_cycles = modeler.run_simulation()
    print(f"  FP8+FP8    总周期: {total_cycles:>10.1f}")
    return total_cycles


def demo_matmul_split():
    """矩阵乘切分 — d_base > baseK_C1 触发 matmulK"""
    modeler = C1Modeler(
        s1_total=128, s2_total=128, d_total=256,
        s1_base_size=128, s2_base_size=128, d_base_size=256,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,   # d_base(256) > baseK_C1(128) → matmulK
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16,
        kv_data_type=DataType.FP16,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    timeline, _, unit_times, total_cycles = modeler.run_simulation()
    mac_p = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    fix_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    print(f"  matmulK    MAC×{len(mac_p)} FIXPIPE×{len(fix_p)}  总周期: {total_cycles:>10.1f}")
    return total_cycles


if __name__ == "__main__":
    print("=" * 50)
    print("Flash Attention C1V1C2V2 — 快速特性演示")
    print("=" * 50)

    print("\n【核间流水对比】")
    c_default  = demo_default()
    c_preload1 = demo_preload1()
    c_preload2 = demo_preload2()
    print(f"\n  PRELOAD_1 vs DEFAULT:  {(c_default - c_preload1) / c_default * 100:+.1f}%")
    print(f"  PRELOAD_2 vs DEFAULT:  {(c_default - c_preload2) / c_default * 100:+.1f}%")

    print("\n【数据类型对比】")
    demo_default()
    demo_fp8()

    print("\n【矩阵乘切分】")
    demo_matmul_split()

    print("\n更多示例:")
    print("  python examples/run_c1_examples.py      # 各配置+时间线图")
    print("  python examples/test_full_pipeline.py   # 完整流水线事件流")
    print("  python examples/test_preload.py         # PRELOAD_1 / PRELOAD_2 对比")
    print("  python -m pytest tests/ -q              # 运行全部测试")
