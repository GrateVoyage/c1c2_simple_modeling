"""
测试InterCorePipeline模式
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

print("=" * 80)
print("测试1: DEFAULT (顺序流水 - C1V1C2V2连续，V在V1后加载)")
print("=" * 80)

modeler = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    s1_base_size=128,
    s2_base_size=128,
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
    L1_db=False,
    L0_db=False,
    inter_core_pipeline=InterCorePipeline.DEFAULT,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)

timeline, _, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "outputs/preload_0_timeline.png")

print("\n事件序列 (前20个):")
print("-" * 80)
for i, e in enumerate(timeline[:20]):
    if e.operation == "P":
        label = f"P{e.q_block_idx+1}{e.k_block_idx+1}"
    elif e.operation == "O":
        label = f"O{e.q_block_idx+1}{e.k_block_idx+1}"
    elif "V" in e.operation and e.unit in ["MTE2", "MTE1"]:
        label = f"V{e.k_block_idx+1}"
    elif "Q" in e.operation:
        label = f"Q{e.q_block_idx+1}"
    elif "K" in e.operation:
        label = f"K{e.k_block_idx+1}"
    else:
        label = ""
    print(f"{i:3} {e.unit:12} {e.operation:6} {label:6} [{e.start_time:7.1f}-{e.end_time:7.1f}] {e.duration:6.1f}")

print("\n" + "=" * 80)
print("测试2: PRELOAD_1 (渐进式2WS流水 - C1提前发射，C2延迟1个K)")
print("=" * 80)

modeler2 = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    s1_base_size=128,
    s2_base_size=128,
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
    L1_db=False,
    L0_db=False,
    inter_core_pipeline=InterCorePipeline.PRELOAD_1,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)

timeline2, _, unit_times2, total_cycles2 = modeler2.run_simulation()
modeler2.print_performance(unit_times2, total_cycles2)
modeler2.plot_timeline(timeline2, unit_times2, total_cycles2, "outputs/preload_1_timeline.png")

print("\n事件序列 (前24个):")
print("-" * 80)
for i, e in enumerate(timeline2[:24]):
    if e.operation == "P":
        label = f"P{e.q_block_idx+1}{e.k_block_idx+1}"
    elif e.operation == "O":
        label = f"O{e.q_block_idx+1}{e.k_block_idx+1}"
    elif "V" in e.operation and e.unit in ["MTE2", "MTE1"]:
        label = f"V{e.k_block_idx+1}"
    elif "Q" in e.operation:
        label = f"Q{e.q_block_idx+1}"
    elif "K" in e.operation:
        label = f"K{e.k_block_idx+1}"
    else:
        label = ""
    print(f"{i:3} {e.unit:12} {e.operation:6} {label:6} [{e.start_time:7.1f}-{e.end_time:7.1f}] {e.duration:6.1f}")

print("\n" + "=" * 80)
print("测试3: PRELOAD_2 (渐进式3WS流水 - V1紧跟C1，C2延迟2个K)")
print("=" * 80)

modeler3 = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    s1_base_size=128,
    s2_base_size=128,
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
    L1_db=False,
    L0_db=False,
    inter_core_pipeline=InterCorePipeline.PRELOAD_2,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)

timeline3, _, unit_times3, total_cycles3 = modeler3.run_simulation()
modeler3.print_performance(unit_times3, total_cycles3)
modeler3.plot_timeline(timeline3, unit_times3, total_cycles3, "outputs/preload_2_timeline.png")

print("\n" + "=" * 80)
print("性能对比")
print("=" * 80)
print(f"DEFAULT   (顺序流水):      总周期 = {total_cycles:.1f}")
print(f"PRELOAD_1 (2WS渐进式): 总周期 = {total_cycles2:.1f}  ({(total_cycles-total_cycles2)/total_cycles*100:+.1f}%)")
print(f"PRELOAD_2 (3WS渐进式): 总周期 = {total_cycles3:.1f}  ({(total_cycles-total_cycles3)/total_cycles*100:+.1f}%)")

print("\n验证要点:")
print("1. PRELOAD_1: C1[k] 在前一轮 V1C2V2 期间并行发射，C2 延迟 1 个 K，2个WS")
print("2. PRELOAD_2: V1 紧跟 C1 发射（无需等 C2），C2 延迟 2 个 K，3个WS")
print("3. 执行序列 PRELOAD_1: C1 -> C1V1C2 -> C1V1C2V2 -> ... -> V1C2V2 -> C2V2 -> V2")
print("4. 执行序列 PRELOAD_2: C1V1 -> C1V1 -> C1V1C2V2 -> ... -> C2V2 -> C2V2")
print("\n图表已保存:")
print("  - outputs/preload_0_timeline.png  (DEFAULT顺序流水)")
print("  - outputs/preload_1_timeline.png  (PRELOAD_1 渐进式 2WS)")
print("  - outputs/preload_2_timeline.png  (PRELOAD_2 渐进式 3WS)")
