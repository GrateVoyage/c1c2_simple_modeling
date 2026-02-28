"""
DN模式配置模板
"""
from dataclasses import dataclass
from core import DataType, LoadOrder, InterCorePipeline, InnerCorePipeline
from .standard import StandardConfig

@dataclass
class DNModeConfig(StandardConfig):
    """DN模式配置 - L1A/L0A存储K, L1B/L0B存储Q"""

    # DN模式特定配置
    use_dn: bool = True
    L1_db: bool = True
    L0_db: bool = True

    # 默认使用L2缓存
    is_l2cache: bool = False

    def __post_init__(self):
        """验证DN模式配置"""
        if not self.use_dn:
            raise ValueError("DN模式配置必须启用use_dn")
