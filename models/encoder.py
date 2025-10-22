import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# 新增：空间卷积残差模块
class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return self.relu(out + x)


class TrackEncoder(nn.Module):
    """
    时空信息提取编码器 - TSADCNN的核心组件
    
    与标准SimCLR不同，该编码器专门处理轨迹段数据，提取时空特征
    包含对称相关信息提取和维度提升功能
    """
    
    def __init__(
        self, 
        input_dim: int = 4,  # [x, y, vx, vy] 位置和速度
        hidden_dim: int = 512,
        output_dim: int = 256,
        sequence_length: int = 10,
        num_layers: int = 3
    ):
        super(TrackEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # 时间特征提取 - 使用LSTM处理序列数据
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        # 新增：时间相关性图投影器，将 L×L 相关性图汇聚为 hidden_dim 向量
        self.temporal_corr_projector = nn.Sequential(
            nn.Linear(sequence_length * sequence_length, hidden_dim),
            nn.ReLU()
        )
        
        # 替换空间特征提取为卷积残差结构（3x3 与 1x1）
        mid_channels = hidden_dim // 4
        # 并行分支：3x3 与 1x1，随后相加
        self.spatial_conv3 = nn.Conv2d(in_channels=1, out_channels=mid_channels, kernel_size=3, padding=1)
        self.spatial_conv1 = nn.Conv2d(in_channels=1, out_channels=mid_channels, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.spatial_residual_stack = nn.Sequential(
            ResidualConvBlock(mid_channels),
            ResidualConvBlock(mid_channels),
            ResidualConvBlock(mid_channels)
        )
        # 展平后FC到 hidden_dim
        self.spatial_fc = nn.Sequential(
            nn.Linear(mid_channels * sequence_length * sequence_length, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # 对称相关信息提取（保留但不强依赖）
        self.symmetric_correlation = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 维度提升层 - TSADCNN的关键特性
        self.dimension_lifting = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, track_sequence: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            track_sequence: 轨迹序列 [batch_size, sequence_length, input_dim]
            
        Returns:
            encoded_features: 编码后的特征 [batch_size, output_dim]
        """
        batch_size, seq_len, input_dim = track_sequence.shape
        
        # 1. 时间特征提取（双向LSTM）
        temporal_features, (h_n, c_n) = self.temporal_encoder(track_sequence)
        # 构建时间相关性图：归一化后两两时刻相似度 -> tanh
        Y = F.normalize(temporal_features, p=2, dim=-1)
        temporal_corr = torch.bmm(Y, Y.transpose(1, 2))  # [batch, L, L]
        temporal_corr = torch.tanh(temporal_corr)
        # 将相关性图汇聚为时间向量
        temporal_repr = self.temporal_corr_projector(temporal_corr.view(batch_size, -1))  # [batch, hidden_dim]
        # 使用相关性图路径得到 temporal_repr，已在上方计算
        
        # 2. 空间结构特征提取（L×L 并行3x3/1x1相加 + 3×3残差堆叠 + 展平FC）
        pos = track_sequence[..., :2]
        pos = F.normalize(pos, p=2, dim=-1)
        spatial_corr = torch.bmm(pos, pos.transpose(1, 2))  # [batch, L, L]
        x2d = spatial_corr.unsqueeze(1)  # [batch, 1, L, L]
        x3 = self.act(self.spatial_conv3(x2d))
        x1 = self.act(self.spatial_conv1(x2d))
        spatial = x3 + x1
        spatial = self.spatial_residual_stack(spatial)
        spatial_flat = spatial.view(batch_size, -1)
        spatial_repr = self.spatial_fc(spatial_flat)  # [batch, hidden_dim]
        
        # 3. 特征融合
        fused = torch.cat([temporal_repr, spatial_repr], dim=1)  # [batch, hidden_dim*2]
        fused = self.feature_fusion(fused)  # [batch, output_dim]
        
        # 4. 维度提升
        encoded_features = self.dimension_lifting(fused)
        return encoded_features

    def extract_temporal_features(self, track_sequence: torch.Tensor) -> torch.Tensor:
        """单独提取时间特征（相关性图路径）"""
        temporal_features, (h_n, c_n) = self.temporal_encoder(track_sequence)
        Y = F.normalize(temporal_features, p=2, dim=-1)
        temporal_corr = torch.bmm(Y, Y.transpose(1, 2))
        temporal_corr = torch.tanh(temporal_corr)
        return self.temporal_corr_projector(temporal_corr.view(track_sequence.size(0), -1))
    
    def extract_spatial_features(self, track_sequence: torch.Tensor) -> torch.Tensor:
        """单独提取空间特征（L×L并行3x3/1x1+残差+FC）"""
        batch_size = track_sequence.size(0)
        pos = track_sequence[..., :2]
        pos = F.normalize(pos, p=2, dim=-1)
        spatial_corr = torch.bmm(pos, pos.transpose(1, 2))
        x2d = spatial_corr.unsqueeze(1)
        x3 = self.act(self.spatial_conv3(x2d))
        x1 = self.act(self.spatial_conv1(x2d))
        spatial = x3 + x1
        spatial = self.spatial_residual_stack(spatial)
        spatial_flat = spatial.view(batch_size, -1)
        return self.spatial_fc(spatial_flat)