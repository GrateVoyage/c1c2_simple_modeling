"""
使用指南 - 快速上手
"""

# 方法1: 使用配置模板(推荐)
from modelers import C1Modeler
from modelers.templates import StandardConfig, DNModeConfig, FullLoadConfig
from core import DataType

# 1. 标准配置
config = StandardConfig(
    s1_total=256,
    s2_total=1024,
    d_total=128,
)
modeler = C1Modeler(**config.to_dict())
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
modeler.print_performance(unit_times, total_cycles)
modeler.plot_timeline(timeline, unit_times, total_cycles, "output.png")

# 2. DN模式配置
config = DNModeConfig(
    s1_total=512,
    s2_total=2048,
    d_total=256,
)
modeler = C1Modeler(**config.to_dict())
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()

# 3. Full Load配置
config = FullLoadConfig(
    s1_total=512,
    s2_total=2048,
    d_total=256,
)
modeler = C1Modeler(**config.to_dict())
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()


# 方法2: 直接自定义参数
modeler = C1Modeler(
    s1_total=512,
    s2_total=2048,
    d_total=256,
    s1_base_size=128,
    s2_base_size=256,
    d_base_size=256,
    data_type=DataType.FP16,
    use_dn=True,
    L1_db=True,
    L0_db=True,
    is_l2cache=False,
    full_load=False
)
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()


# 方法3: 自定义硬件配置
from core import HardwareConfig

hw_config = HardwareConfig(
    CHIP_FREQ_GHZ=1.65,
    MTE2_DRAM_BW_GBPS=1600,
    MTE2_L2_BW_GBPS=5400,
)

modeler = C1Modeler(
    s1_total=512,
    s2_total=2048,
    d_total=256,
    hw_config=hw_config
)
timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
