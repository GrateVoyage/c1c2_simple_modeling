"""
标准配置模板
"""
from dataclasses import dataclass
from core import DataType, LoadOrder, InterCorePipeline, InnerCorePipeline

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
    q_data_type: DataType = DataType.FP16
    kv_data_type: DataType = DataType.FP16

    # 矩阵乘切分块大小 - C1阶段
    baseM_C1: int = 128
    baseN_C1: int = 128
    baseK_C1: int = 128

    # 矩阵乘切分块大小 - C2阶段
    baseM_C2: int = 128
    baseN_C2: int = 128
    baseK_C2: int = 128

    # 缓存和流水线配置
    is_l2cache: bool = False
    use_dn: bool = False
    L1_db: bool = False
    L0_db: bool = False

    # 加载配置
    load_order: LoadOrder = LoadOrder.LOAD_Q_FIRST
    full_load: bool = False

    # 流水线模式
    inter_core_pipeline: InterCorePipeline = InterCorePipeline.DEFAULT
    inner_core_pipeline: InnerCorePipeline = InnerCorePipeline.DEFAULT

    def to_dict(self):
        """转换为字典"""
        return {
            's1_total': self.s1_total,
            's2_total': self.s2_total,
            'd_total': self.d_total,
            's1_base_size': self.s1_base_size,
            's2_base_size': self.s2_base_size,
            'd_base_size': self.d_base_size,
            'q_data_type': self.q_data_type,
            'kv_data_type': self.kv_data_type,
            'baseM_C1': self.baseM_C1,
            'baseN_C1': self.baseN_C1,
            'baseK_C1': self.baseK_C1,
            'baseM_C2': self.baseM_C2,
            'baseN_C2': self.baseN_C2,
            'baseK_C2': self.baseK_C2,
            'is_l2cache': self.is_l2cache,
            'use_dn': self.use_dn,
            'L1_db': self.L1_db,
            'L0_db': self.L0_db,
            'load_order': self.load_order,
            'full_load': self.full_load,
            'inter_core_pipeline': self.inter_core_pipeline,
            'inner_core_pipeline': self.inner_core_pipeline,
        }
