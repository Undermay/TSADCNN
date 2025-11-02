"""
TSADCNN Models Package

包含TSADCNN模型的所有核心组件：
- TSADCNN: 主模型
- TrackEncoder: 时空信息提取编码器
- ProjectionHead: 共享投影头
"""

from .tsadcnn import TSADCNN
from .encoder import TrackEncoder
from .projection import ProjectionHead

__all__ = [
    'TSADCNN',
    'TrackEncoder', 
    'ProjectionHead'
]