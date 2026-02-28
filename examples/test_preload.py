"""
测试InterCorePipeline模式和TWOBUFFER特性
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
    L1_db=False,
    L0_db=False,
    inter_core_pipeline=InterCorePipeline.DEFAULT,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
)

timeline, _, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "preload_0_timeline.png")

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
print("测试2: PRELOAD (渐进式流水)，TWOBUFFER=False (V在K加载后立即预加载，共享L1B slot)")
print("=" * 80)

modeler2 = C1Modeler(
    s1_total=256,
    s2_total=512,
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
    L1_db=False,
    L0_db=False,
    inter_core_pipeline=InterCorePipeline.PRELOAD,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
    two_buffer=False
)

timeline2, _, unit_times2, total_cycles2 = modeler2.run_simulation()
modeler2.print_performance(unit_times2, total_cycles2)
modeler2.plot_timeline(timeline2, unit_times2, total_cycles2, "preload_1_timeline.png")

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
print("测试3: PRELOAD (渐进式流水)，TWOBUFFER=True (V独立L1 slot，与K无slot冲突)")
print("=" * 80)

modeler3 = C1Modeler(
    s1_total=256,
    s2_total=512,
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
    L1_db=False,
    L0_db=False,
    inter_core_pipeline=InterCorePipeline.PRELOAD,
    inner_core_pipeline=InnerCorePipeline.DEFAULT,
    two_buffer=True
)

timeline3, _, unit_times3, total_cycles3 = modeler3.run_simulation()
modeler3.print_performance(unit_times3, total_cycles3)
modeler3.plot_timeline(timeline3, unit_times3, total_cycles3, "preload_1_twobuffer_timeline.png")

print("\n事件序列 (前24个):")
print("-" * 80)
for i, e in enumerate(timeline3[:24]):
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
print("性能对比")
print("=" * 80)
print(f"DEFAULT (顺序流水):                总周期 = {total_cycles:.1f}")
print(f"PRELOAD (渐进式), TWOBUFFER=False: 总周期 = {total_cycles2:.1f}  ({(total_cycles-total_cycles2)/total_cycles*100:+.1f}%)")
print(f"PRELOAD (渐进式), TWOBUFFER=True:  总周期 = {total_cycles3:.1f}  ({(total_cycles-total_cycles3)/total_cycles*100:+.1f}%)")

print("\n验证要点:")
print("1. PRELOAD模式时，每个k block的V在K加载完成后立即预加载（不等V1）")
print("2. TWOBUFFER=True时，V使用独立L1 slot，与K的slot无冲突，V可更早开始加载")
print("3. C2阶段跳过MTE2 V加载步骤，直接MTE1从L1→L0")
print("\n图表已保存:")
print("  - preload_0_timeline.png         (DEFAULT顺序流水)")
print("  - preload_1_timeline.png         (PRELOAD渐进式, TWOBUFFER=False)")
print("  - preload_1_twobuffer_timeline.png (PRELOAD渐进式, TWOBUFFER=True)")
