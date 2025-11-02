import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    共享投影头 - 将编码器输出映射到对比学习的嵌入空间
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
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        layers.append(nn.Dropout(self.dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
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
        # 共享单投影：不在此处执行归一化，交由上层处理
        return self.projection(x)