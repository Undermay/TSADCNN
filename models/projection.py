import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    投影头 - 将编码器输出映射到对比学习空间
    
    类似于SimCLR的投影头，但针对TSADCNN的双对比学习进行了优化
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25
    ):
        super(ProjectionHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建多层投影网络
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 编码器输出特征 [batch_size, input_dim]
            
        Returns:
            projected_features: 投影后的特征 [batch_size, output_dim]
        """
        projected = self.projection(x)
        # L2归一化，这对对比学习很重要
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class DualProjectionHead(nn.Module):
    """
    双投影头 - TSADCNN的特色组件
    
    为双对比学习提供两个不同的投影空间：
    1. 时间对比投影
    2. 空间对比投影
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25
    ):
        super(DualProjectionHead, self).__init__()
        
        # 时间对比投影头
        self.temporal_projection = ProjectionHead(
            input_dim, hidden_dim, output_dim, num_layers, dropout
        )
        
        # 空间对比投影头
        self.spatial_projection = ProjectionHead(
            input_dim, hidden_dim, output_dim, num_layers, dropout
        )
        
        # 共享特征提取器（可选）
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 保持维度不变
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 编码器输出特征 [batch_size, input_dim]
            
        Returns:
            temporal_proj: 时间对比投影 [batch_size, output_dim]
            spatial_proj: 空间对比投影 [batch_size, output_dim]
        """
        # 可选的共享特征提取
        shared_feat = self.shared_features(x)
        
        # 双投影
        temporal_proj = self.temporal_projection(x)  # 直接使用原始特征
        spatial_proj = self.spatial_projection(x)    # 直接使用原始特征
        
        return temporal_proj, spatial_proj
    
    def get_temporal_projection(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取时间投影"""
        shared_feat = self.shared_features(x)
        return self.temporal_projection(shared_feat)
    
    def get_spatial_projection(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取空间投影"""
        shared_feat = self.shared_features(x)
        return self.spatial_projection(shared_feat)