"""
枚举定义
"""
from enum import Enum

class BoundType(Enum):
    """性能瓶颈类型"""
    MTE2_BOUND = "MTE2_BOUND"
    MTE1_BOUND = "MTE1_BOUND"
    MAC_BOUND = "MAC_BOUND"
    FIXPIPE_BOUND = "FIXPIPE_BOUND"

class DataType(Enum):
    """数据类型"""
    FP16 = "fp16"
    FP8 = "fp8"

class LoadOrder(Enum):
    """加载顺序"""
    LOAD_Q_FIRST = 0
    LOAD_K_FIRST = 1
