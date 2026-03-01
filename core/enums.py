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

class InterCorePipeline(Enum):
    """核间流水线模式"""
    DEFAULT  = "default"    # 顺序流水: C1V1C2V2 → C1V1C2V2 → ...
    PRELOAD  = "preload"    # 渐进式:   V在K加载后立即预加载，C2省去V的MTE2等待
    N_BUFFER = "n_buffer"   # N=2批次:  C1C1→V1V1→C2C2→V2V2

class InnerCorePipeline(Enum):
    """核内流水线模式"""
    DEFAULT    = "default"      # 无策略 (基础L1容量追踪: 512K内不重复MTE2)
    Q_RESIDENT = "q_resident"   # Q常驻: Q=1×144K, KP=2×144K, V=2×32K, LRU
