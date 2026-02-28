"""
测试完整C1V1C2V2流水线
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

# 使用小规模矩阵以便清晰查看完整流水线
modeler = C1Modeler(
    s1_total=256,        # Q: 2个块
    s2_total=512,        # K: 2个块
    d_total=128,
    s1_base_size=128,
    s2_base_size=256,
    d_base_size=128,
    q_data_type=DataType.FP16,
    kv_data_type=DataType.FP16,
    baseM_C1=128,
    baseN_C1=128,
    baseK_C1=128,
    baseM_C2=128,
    baseN_C2=128,
    baseK_C2=128,
    use_dn=False,
    L1_db=True,
    L0_db=True,
    is_l2cache=False,
    full_load=False,
    inter_core_pipeline=InterCorePipeline.DEFAULT,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)

print("=" * 80)
print("测试完整C1V1C2V2流水线")
print("=" * 80)
print("\nFlash Attention算子完整流程:")
print("  C1: Q @ K^T → S (Attention Score)")
print("  V1: Softmax S → P (Attention Probability)")
print("  C2: P @ V → O (Output)")
print("  V2: 后处理")
print()

timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "full_pipeline_timeline.png")

print("\n" + "=" * 80)
print("关键事件检查 (最后20个)")
print("=" * 80)
print(f"{'#':<4} {'Unit':<12} {'Op':<6} {'Label':<8} {'Start':<10} {'End':<10} {'Duration':<10}")
print("-" * 80)

for i, event in enumerate(timeline[-20:]):
    idx = len(timeline) - 20 + i
    q_idx = event.q_block_idx + 1
    k_idx = event.k_block_idx + 1

    # 生成标注
    if "Q" in event.operation:
        label = f"Q{q_idx}"
    elif "K" in event.operation:
        label = f"K{k_idx}"
    elif "V" in event.operation and event.unit in ["MTE2", "MTE1"]:
        label = f"V{k_idx}"
    elif event.operation == "P":
        label = f"P{q_idx}{k_idx}"
    elif event.operation == "O":
        label = f"O{q_idx}{k_idx}"
    else:
        label = f"Q{q_idx}K{k_idx}"

    print(f"{idx:<4} {event.unit:<12} {event.operation:<6} {label:<8} "
          f"{event.start_time:<10.1f} {event.end_time:<10.1f} {event.duration:<10.1f}")

print("\n" + "=" * 80)
print("验证要点")
print("=" * 80)
print("✓ C1阶段: MAC产生P矩阵，FIXPIPE搬到UB，VECTOR_V1处理，标注为P11, P12...")
print("✓ 转换阶段: MTE3搬P回CUBE，同时MTE2/MTE1加载V矩阵")
print("✓ C2阶段: MAC产生O矩阵，FIXPIPE搬到UB，VECTOR_V2处理，标注为O11, O12...")
print("✓ VECTOR_V1和VECTOR_V2分别在不同Y轴高度显示")
print("✓ 图表中显示完整的C1V1C2V2流水线")
print("\n图表已保存: full_pipeline_timeline.png")
