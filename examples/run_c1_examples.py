"""
C1建模器运行示例
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelers import C1Modeler
from modelers.templates import StandardConfig, DNModeConfig, FullLoadConfig
from core import DataType

def example_standard():
    """标准配置示例"""
    print("\n" + "="*60)
    print("示例1: 标准配置")
    print("="*60)

    config = StandardConfig(
        s1_total=256,
        s2_total=1024,
        d_total=128,
        s1_base_size=128,
        s2_base_size=256,
        d_base_size=128,
        data_type=DataType.FP16,
    )

    modeler = C1Modeler(**config.to_dict())
    timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
    modeler.print_performance(unit_times, total_cycles)
    modeler.plot_timeline(timeline, unit_times, total_cycles, "standard_timeline.png")

def example_dn_mode():
    """DN模式示例"""
    print("\n" + "="*60)
    print("示例2: DN模式配置")
    print("="*60)

    config = DNModeConfig(
        s1_total=512,
        s2_total=2048,
        d_total=256,
        s1_base_size=128,
        s2_base_size=256,
        d_base_size=256,
        data_type=DataType.FP16,
    )

    modeler = C1Modeler(**config.to_dict())
    timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
    modeler.print_performance(unit_times, total_cycles)
    modeler.plot_timeline(timeline, unit_times, total_cycles, "dn_mode_timeline.png")

def example_full_load():
    """Full Load模式示例"""
    print("\n" + "="*60)
    print("示例3: Full Load配置")
    print("="*60)

    config = FullLoadConfig(
        s1_total=512,
        s2_total=2048,
        d_total=256,
        s1_base_size=128,
        s2_base_size=256,
        d_base_size=256,
        data_type=DataType.FP16,
    )

    modeler = C1Modeler(**config.to_dict())
    timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
    modeler.print_performance(unit_times, total_cycles)
    modeler.plot_timeline(timeline, unit_times, total_cycles, "full_load_timeline.png")

def example_custom():
    """自定义配置示例"""
    print("\n" + "="*60)
    print("示例4: 自定义配置 (L1+L0双缓冲)")
    print("="*60)

    modeler = C1Modeler(
        s1_total=256,
        s2_total=1024,
        d_total=128,
        s1_base_size=128,
        s2_base_size=256,
        d_base_size=128,
        data_type=DataType.FP16,
        is_l2cache=False,
        use_dn=False,
        L1_db=True,
        L0_db=True,
        full_load=False
    )

    timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
    modeler.print_performance(unit_times, total_cycles)
    modeler.plot_timeline(timeline, unit_times, total_cycles, "custom_timeline.png")

def main():
    """运行所有示例"""
    print("C1建模器示例集")

    # 运行各个示例
    example_standard()
    example_dn_mode()
    example_full_load()
    example_custom()

    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)

if __name__ == "__main__":
    main()
