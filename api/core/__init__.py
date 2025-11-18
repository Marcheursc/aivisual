"""
核心算法模块
包含所有计算机视觉检测算法的统一接口
"""

from ..algorithms.loitering_detection import LoiteringDetector
from ..algorithms.leave_detection import LeaveDetector
from ..algorithms.gather_detection import GatherDetector

__all__ = [
    "LoiteringDetector",
    "LeaveDetector", 
    "GatherDetector"
]