"""
配置模板模块
"""
from .standard import StandardConfig
from .dn_mode import DNModeConfig
from .full_load import FullLoadConfig

__all__ = [
    'StandardConfig',
    'DNModeConfig',
    'FullLoadConfig',
]
