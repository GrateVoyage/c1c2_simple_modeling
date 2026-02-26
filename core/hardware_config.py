"""
硬件配置参数
"""
from dataclasses import dataclass
from .enums import DataType

@dataclass
class HardwareConfig:
    """昇腾NPU硬件配置"""

    # 芯片频率
    CHIP_FREQ_GHZ: float = 1.65

    # MTE2 带宽配置 (DRAM)
    MTE2_DRAM_BW_GBPS: float = 1600  # GB/s

    # MTE2 L2缓存带宽
    MTE2_L2_BW_GBPS: float = 5400  # GB/s

    # MTE1 FixPipe带宽 (bytes/cycle)
    MTE1_FIXPIPE_BYTES_PER_CYCLE: float = 256.0

    # MAC吞吐量配置
    MAC_THROUGHPUT_FP16: int = 16 * 16 * 16 * 2  # m * n * k * 2
    MAC_THROUGHPUT_FP8: int = 16 * 32 * 16 * 2

    def __post_init__(self):
        """计算派生参数"""
        self.MTE2_DRAM_BW = self.MTE2_DRAM_BW_GBPS * 1024 * 1024 * 1024 / 32
        self.MTE2_DRAM_BYTES_PER_CYCLE = self.MTE2_DRAM_BW / (self.CHIP_FREQ_GHZ * 1e9)

        self.MTE2_L2_BW = self.MTE2_L2_BW_GBPS * 1024 * 1024 * 1024 / 32
        self.MTE2_L2_BYTES_PER_CYCLE = self.MTE2_L2_BW / (self.CHIP_FREQ_GHZ * 1e9)

    def get_mac_throughput(self, data_type: DataType) -> int:
        """获取MAC吞吐量"""
        return self.MAC_THROUGHPUT_FP16 if data_type == DataType.FP16 else self.MAC_THROUGHPUT_FP8
