"""
标准配置模板
"""
from dataclasses import dataclass
from core import DataType, LoadOrder

@dataclass
class StandardConfig:
    """标准C1配置"""

    # 矩阵维度
    s1_total: int = 256
    s2_total: int = 1024
    d_total: int = 128

    # 块大小
    s1_base_size: int = 128
    s2_base_size: int = 256
    d_base_size: int = 128

    # 数据类型
    data_type: DataType = DataType.FP16

    # 缓存和流水线配置
    is_l2cache: bool = False
    use_dn: bool = False
    L1_db: bool = False
    L0_db: bool = False

    # 加载配置
    load_order: LoadOrder = LoadOrder.LOAD_Q_FIRST
    full_load: bool = False
    preload: int = 0

    def to_dict(self):
        """转换为字典"""
        return {
            's1_total': self.s1_total,
            's2_total': self.s2_total,
            'd_total': self.d_total,
            's1_base_size': self.s1_base_size,
            's2_base_size': self.s2_base_size,
            'd_base_size': self.d_base_size,
            'data_type': self.data_type,
            'is_l2cache': self.is_l2cache,
            'use_dn': self.use_dn,
            'L1_db': self.L1_db,
            'L0_db': self.L0_db,
            'load_order': self.load_order,
            'full_load': self.full_load,
            'preload': self.preload,
        }
