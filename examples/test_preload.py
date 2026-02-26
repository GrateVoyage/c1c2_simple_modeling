"""
测试Preload模式和新的Vector周期数
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelers import C1Modeler
from core import DataType

print("=" * 80)
print("测试1: Preload=0 (正常模式 - C1V1C2V2连续)")
print("=" * 80)

modeler = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    s1_base_size=128,
    s2_base_size=256,
    d_base_size=128,
    data_type=DataType.FP16,
    use_dn=False,
    L1_db=False,
    L0_db=False,
    preload=0
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
print("测试2: Preload=1 (先执行所有C1，再执行所有C2)")
print("=" * 80)

modeler2 = C1Modeler(
    s1_total=256,
    s2_total=512,
    d_total=128,
    s1_base_size=128,
    s2_base_size=256,
    d_base_size=128,
    data_type=DataType.FP16,
    use_dn=False,
    L1_db=False,
    L0_db=False,
    preload=1
)

timeline2, _, unit_times2, total_cycles2 = modeler2.run_simulation()
modeler2.print_performance(unit_times2, total_cycles2)
modeler2.plot_timeline(timeline2, unit_times2, total_cycles2, "preload_1_timeline.png")

print("\n事件序列 (前20个):")
print("-" * 80)
for i, e in enumerate(timeline2[:20]):
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
print("验证要点")
print("=" * 80)
print("1. V1周期数: 1600 cycles")
print("2. V2周期数: 400 cycles")
print("3. Y轴间距: 紧凑模式 (0.3/0.5)")
print("4. Preload=0: C1V1C2V2连续执行")
print("5. Preload=1: 先所有C1+V1，再所有C2+V2")

# 检查C1和C2的分离
if modeler2.preload == 1:
    c1_events = [e for e in timeline2 if e.operation == "P" and e.unit == "MAC"]
    c2_events = [e for e in timeline2 if e.operation == "O" and e.unit == "MAC"]

    if c1_events and c2_events:
        last_c1_time = max(e.end_time for e in c1_events)
        first_c2_time = min(e.start_time for e in c2_events)

        print(f"\n6. Preload验证:")
        print(f"   - 最后C1结束: {last_c1_time:.1f}")
        print(f"   - 首个C2开始: {first_c2_time:.1f}")
        if first_c2_time >= last_c1_time:
            print("   ✓ C2确实在所有C1完成后执行")
        else:
            print("   ✗ 时序有问题")

print("\n图表已保存:")
print("  - preload_0_timeline.png (正常模式)")
print("  - preload_1_timeline.png (Preload模式)")
