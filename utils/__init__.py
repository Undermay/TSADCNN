"""
TSADCNN utils 包入口（精简版）。

保留当前训练/评估实际使用的工具：
- contrastive_data_loader: 对比学习数据集与加载器
- normalization: 归一化工具
- metrics_pak: 正确口径的对比评估指标
"""

from .contrastive_data_loader import ContrastiveTrajectoryDataset, create_contrastive_data_loaders
from .normalization import TrajectoryNormalizer
from .metrics_pak import (
    evaluate_contrastive_model_correct,
    compute_contrastive_association_metrics_correct,
)

__all__ = [
    'ContrastiveTrajectoryDataset',
    'create_contrastive_data_loaders',
    'TrajectoryNormalizer',
    'evaluate_contrastive_model_correct',
    'compute_contrastive_association_metrics_correct',
]