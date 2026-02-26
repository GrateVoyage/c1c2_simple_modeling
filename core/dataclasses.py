"""
数据结构定义
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class TimelineEvent:
    """时间线事件"""
    unit: str
    operation: str
    start_time: float
    end_time: float
    duration: float
    buffer: Optional[str] = None
    q_block_idx: int = 0
    k_block_idx: int = 0
    is_l2_hit: bool = False
