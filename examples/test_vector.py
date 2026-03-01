"""
快速测试 - 验证Vector流水线和P标注
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

# 使用较小的矩阵规模以便清晰查看标注
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

print("运行小规模测试以验证P标注和Vector流水线...")
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "outputs/test_vector_timeline.png")

print("\n检查timeline中的事件:")
print("=" * 80)
for i, event in enumerate(timeline[-10:]):  # 显示最后10个事件
    print(f"Event {len(timeline)-10+i}: {event.unit:8s} | {event.operation:6s} | "
          f"Q{event.q_block_idx+1}K{event.k_block_idx+1} | "
          f"[{event.start_time:.1f} - {event.end_time:.1f}] {event.duration:.1f} cycles")

print("\n验证要点:")
print("1. MAC的operation应为 'P'")
print("2. FIXPIPE的operation应为 'P'")
print("3. VECTOR的operation应为 'V1'")
print("4. VECTOR应在FIXPIPE之后执行")
print("5. 图表中应显示 P11, P12, P21, P22 和 V111, V112, V121, V122 等标注")
