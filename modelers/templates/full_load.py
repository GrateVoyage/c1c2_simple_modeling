"""
Full Load配置模板
"""
from dataclasses import dataclass
from core import DataType, LoadOrder, InterCorePipeline, InnerCorePipeline
from .standard import StandardConfig

@dataclass
class FullLoadConfig(StandardConfig):
    """Full Load配置 - 预加载所有Q块"""

    # Full Load特定配置
    full_load: bool = True
    L1_db: bool = True
    L0_db: bool = True

    # 推荐启用L2缓存以提高性能
    is_l2cache: bool = True

    def __post_init__(self):
        """验证Full Load配置"""
        if not self.full_load:
            raise ValueError("Full Load配置必须启用full_load")
