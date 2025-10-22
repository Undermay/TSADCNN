import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .encoder import TrackEncoder
from .projection import DualProjectionHead


class TSADCNN(nn.Module):
    """
    Track Segment Association with Dual Contrast Neural Network
    
    TSADCNN主模型，实现双对比学习机制用于轨迹段关联
    
    主要特性：
    1. 时空信息提取模块（TrackEncoder）
    2. 双投影头（DualProjectionHead）
    3. 双对比学习损失
    4. 最近邻关联
    """
    
    def __init__(
        self,
        input_dim: int = 4,          # 轨迹点维度 [x, y, vx, vy]
        sequence_length: int = 10,    # 轨迹段长度
        encoder_hidden_dim: int = 512,
        encoder_output_dim: int = 256,
        projection_hidden_dim: int = 512,
        projection_output_dim: int = 128,
        temperature: float = 0.07,
        encoder_layers: int = 3,
        projection_layers: int = 2
    ):
        super(TSADCNN, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.temperature = temperature
        
        # 时空信息提取编码器
        self.encoder = TrackEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            sequence_length=sequence_length,
            num_layers=encoder_layers
        )
        
        # 双投影头
        self.projection_head = DualProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            num_layers=projection_layers
        )
        
    def forward(
        self, 
        track_segments: torch.Tensor,
        return_projections: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            track_segments: 轨迹段数据 [batch_size, sequence_length, input_dim]
            return_projections: 是否返回投影结果
            
        Returns:
            outputs: 包含编码特征和投影结果的字典
        """
        # 1. 时空信息提取
        encoded_features = self.encoder(track_segments)
        
        outputs = {
            'encoded_features': encoded_features
        }
        
        # 2. 双投影（如果需要）
        if return_projections:
            temporal_proj, spatial_proj = self.projection_head(encoded_features)
            outputs.update({
                'temporal_projection': temporal_proj,
                'spatial_projection': spatial_proj
            })
        
        return outputs
    
    def compute_dual_contrastive_loss(
        self,
        temporal_proj: torch.Tensor,
        spatial_proj: torch.Tensor,
        labels: torch.Tensor,
        temporal_weight: float = 0.5,
        spatial_weight: float = 0.5,
        margin: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算双对比学习损失（严格对齐论文公式23/24/25）
        
        - Lc（公式23）：距离优化对比损失 = 1/2*lD² + 1/2*(1-l)[max(0,m-D)]²
        - Ls（公式24）：对称约束损失 = ΣΣ(aij - aji)²
        - 总损失（公式25）：L = 10 * Ls + Lc
        
        Args:
            temporal_proj: 时间投影 [batch_size, projection_dim]
            spatial_proj: 空间投影 [batch_size, projection_dim]
            labels: 标签 [batch_size]
            margin: 距离优化的边界参数 m
        """
        # 距离优化对比损失（Lc，公式23）
        Lc = self._compute_distance_contrastive_loss(temporal_proj, spatial_proj, labels, margin)
        
        # 对称约束损失（Ls，公式24）
        Ls = self._compute_symmetric_constraint_loss(temporal_proj, spatial_proj)
        
        # 总损失（公式25）
        total_loss = 10.0 * Ls + Lc
        
        return {
            'total_loss': total_loss,
            'Ls': Ls,
            'Lc': Lc,
            'distance_loss': Lc,
            'symmetric_loss': Ls,
            # 保持向后兼容
            'temporal_loss': Lc * 0.5,  # 近似分配
            'spatial_loss': Lc * 0.5,
            'cross_loss': Ls
        }
    
    def _compute_distance_contrastive_loss(
        self,
        temporal_proj: torch.Tensor,
        spatial_proj: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """
        计算距离优化对比损失（公式23）
        
        Lc = 1/2 * l * D² + 1/2 * (1-l) * [max(0, m-D)]²
        
        其中：
        - l: 标签（1表示同一目标，0表示不同目标）
        - D: 高维空间中的欧几里得距离
        - m: 边界参数
        """
        batch_size = temporal_proj.shape[0]
        
        # 计算所有样本对的欧几里得距离
        # 使用时间和空间投影的平均作为特征表示
        features = 0.5 * (temporal_proj + spatial_proj)
        
        # 计算距离矩阵 D = ||f(xi) - f(xj)||2
        distance_matrix = torch.cdist(features, features, p=2)
        
        # 创建标签掩码
        labels_expanded = labels.unsqueeze(1)
        same_label_mask = (labels_expanded == labels_expanded.T).float()
        diff_label_mask = 1.0 - same_label_mask
        
        # 移除对角线（自己与自己的距离）
        eye_mask = torch.eye(batch_size, device=features.device)
        same_label_mask = same_label_mask * (1 - eye_mask)
        diff_label_mask = diff_label_mask * (1 - eye_mask)
        
        # 公式23：Lc = 1/2 * l * D² + 1/2 * (1-l) * [max(0, m-D)]²
        positive_loss = 0.5 * same_label_mask * (distance_matrix ** 2)
        negative_loss = 0.5 * diff_label_mask * torch.clamp(margin - distance_matrix, min=0) ** 2
        
        # 计算平均损失（只考虑有效的样本对）
        num_positive_pairs = torch.sum(same_label_mask)
        num_negative_pairs = torch.sum(diff_label_mask)
        
        total_positive_loss = torch.sum(positive_loss)
        total_negative_loss = torch.sum(negative_loss)
        
        # 避免除零
        if num_positive_pairs > 0:
            avg_positive_loss = total_positive_loss / num_positive_pairs
        else:
            avg_positive_loss = torch.tensor(0.0, device=features.device)
            
        if num_negative_pairs > 0:
            avg_negative_loss = total_negative_loss / num_negative_pairs
        else:
            avg_negative_loss = torch.tensor(0.0, device=features.device)
        
        return avg_positive_loss + avg_negative_loss
    
    def _compute_symmetric_constraint_loss(
        self,
        temporal_proj: torch.Tensor,
        spatial_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对称约束损失（公式24）
        
        Ls = ΣΣ(aij - aji)²
        
        其中aij是相关性矩阵中行i列j的元素
        通过时间相关性信息提取模块处理后，特征图中行列的元素表示第i个轨迹点与第j个轨迹点之间的相关性
        """
        # 计算时间-空间交叉相似度矩阵作为相关性矩阵
        correlation_matrix = torch.matmul(temporal_proj, spatial_proj.T) / self.temperature
        
        # 计算对称性约束：ΣΣ(aij - aji)²
        symmetric_diff = correlation_matrix - correlation_matrix.T
        symmetric_loss = torch.sum(symmetric_diff ** 2)
        
        # 归一化（除以矩阵元素总数）
        batch_size = correlation_matrix.shape[0]
        normalized_loss = symmetric_loss / (batch_size * batch_size)
        
        return normalized_loss
    
    def associate_tracks(
        self,
        query_segments: torch.Tensor,
        candidate_segments: torch.Tensor,
        k: int = 1,
        distance_metric: str = 'cosine'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        轨迹段关联 - 使用最近邻方法
        
        Args:
            query_segments: 查询轨迹段 [num_queries, sequence_length, input_dim]
            candidate_segments: 候选轨迹段 [num_candidates, sequence_length, input_dim]
            k: 返回前k个最近邻
            distance_metric: 距离度量方式
            
        Returns:
            indices: 最近邻索引 [num_queries, k]
            distances: 距离值 [num_queries, k]
        """
        with torch.no_grad():
            # 编码查询和候选轨迹段
            query_features = self.encoder(query_segments)
            candidate_features = self.encoder(candidate_segments)
            
            # 计算距离矩阵
            if distance_metric == 'cosine':
                # 余弦距离
                query_norm = F.normalize(query_features, p=2, dim=1)
                candidate_norm = F.normalize(candidate_features, p=2, dim=1)
                similarity_matrix = torch.matmul(query_norm, candidate_norm.T)
                distance_matrix = 1 - similarity_matrix
            elif distance_metric == 'euclidean':
                # 欧几里得距离
                distance_matrix = torch.cdist(query_features, candidate_features, p=2)
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")
            
            # 找到最近邻
            distances, indices = torch.topk(
                distance_matrix, k, dim=1, largest=False
            )
            
            return indices, distances
    
    def get_track_embeddings(self, track_segments: torch.Tensor) -> torch.Tensor:
        """
        获取轨迹段的嵌入表示
        
        Args:
            track_segments: 轨迹段 [batch_size, sequence_length, input_dim]
            
        Returns:
            embeddings: 嵌入表示 [batch_size, encoder_output_dim]
        """
        with torch.no_grad():
            embeddings = self.encoder(track_segments)
        return embeddings