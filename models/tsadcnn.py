from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TrackEncoder
from .projection import ProjectionHead


# -----------------------------
# Config (paper-aligned)
# -----------------------------
@dataclass
class TSADCNNConfig:
    # 输入与序列配置
    input_dim: int = 2                 # 原文以轨迹二维坐标为主，可扩展为4
    sequence_length: int = 13          # 片段长度（按数据集设置）

    # 编码器配置（对应原文 TrackEncoder: 时空特征+相关矩阵）
    encoder_hidden_dim: int = 128
    encoder_output_dim: int = 128
    encoder_layers: int = 2
    dropout: float = 0.1

    # 投影头配置（对比空间）
    projection_hidden_dim: int = 128
    projection_output_dim: int = 64
    projection_layers: int = 2
    use_dual_projection: bool = False  # 默认使用共享单投影头

    # 训练与损失配置（原文对比+对称约束）
    margin: float = 1.0
    pos_weight: float = 1.0
    neg_weight: float = 1.0
    lambda_symmetric: float = 0.1      # 对称约束权重

    # 结构共享（原文默认共享同一编码器）
    share_backbone: bool = True


# -----------------------------
# Losses (paper-aligned)
# -----------------------------
def _pairwise_cosine(z_old: torch.Tensor, z_new: torch.Tensor) -> torch.Tensor:
    # 逐样本配对余弦相似度（z 已 L2 归一化）
    return (z_old * z_new).sum(dim=1)


def _euclidean_distance_from_cos(cos_sim: torch.Tensor) -> torch.Tensor:
    # d = sqrt(2 * (1 - cos))，保证非负，数值稳定
    d_sq = torch.clamp(2.0 * (1.0 - cos_sim), min=0.0)
    return torch.sqrt(d_sq + 1e-8)


def contrastive_loss(
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor:
    # 原文双对比的核心：正样本拉近、负样本推远（margin）
    cos_sim = _pairwise_cosine(z_old, z_new)
    d = _euclidean_distance_from_cos(cos_sim)

    l = labels.float()
    pos_term = l * (d ** 2)
    neg_term = (1.0 - l) * (F.relu(margin - d) ** 2)

    loss = 0.5 * (pos_weight * pos_term + neg_weight * neg_term)
    return loss.mean()


def symmetric_constraint_from_corr(A: torch.Tensor) -> torch.Tensor:
    # 原文的对称相关约束：惩罚关联矩阵非对称性
    # A: [B, L, L]
    asym = A - A.transpose(1, 2)
    return (asym ** 2).mean()


# -----------------------------
# TSADCNN (paper-aligned)
# -----------------------------
class TSADCNN(nn.Module):
    def __init__(self, cfg: TSADCNNConfig):
        super().__init__()
        self.cfg = cfg

        # 编码器：默认共享同一骨干（原文设置）
        self.encoder = TrackEncoder(
            input_dim=cfg.input_dim,
            sequence_length=cfg.sequence_length,
            hidden_dim=cfg.encoder_hidden_dim,
            output_dim=cfg.encoder_output_dim,
            num_layers=cfg.encoder_layers,
        )

        if not cfg.share_backbone:
            self.encoder_new = TrackEncoder(
                input_dim=cfg.input_dim,
                sequence_length=cfg.sequence_length,
                hidden_dim=cfg.encoder_hidden_dim,
                output_dim=cfg.encoder_output_dim,
                num_layers=cfg.encoder_layers,
            )
        else:
            self.encoder_new = None

        # 共享单投影头（推荐默认）
        self.single_projection = ProjectionHead(
            input_dim=cfg.encoder_output_dim,
            hidden_dim=cfg.projection_hidden_dim,
            output_dim=cfg.projection_output_dim,
            num_layers=cfg.projection_layers,
            dropout=cfg.dropout,
        )

    # -------------------------
    # Encoding API (for eval)
    # -------------------------
    @torch.no_grad()
    def encode_trajectory(self, traj_btK: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        输入：traj_btK [B, T, K]，K>=input_dim
        输出：embedding [B, D], correlation_matrix [B, T, T]
        原文接口：编码器产生全局特征 + 时序相关矩阵；投影头将特征映射到对比空间。
        """
        x = traj_btK[:, :, : self.cfg.input_dim]
        feats, corr = self.encoder(x)

        # 共享投影到同一嵌入空间
        z = F.normalize(self.single_projection(feats), p=2, dim=1)

        return z, corr

    # -------------------------
    # Training forward
    # -------------------------
    def forward(
        self,
        old_btK: torch.Tensor,
        new_btK: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 裁剪输入维度以匹配编码器
        old_x = old_btK[:, :, : self.cfg.input_dim]
        new_x = new_btK[:, :, : self.cfg.input_dim]

        # 编码（共享或非共享骨干）
        old_feats, old_corr = self.encoder(old_x)
        if self.encoder_new is not None:
            new_feats, new_corr = self.encoder_new(new_x)
        else:
            new_feats, new_corr = self.encoder(new_x)

        # 共享投影到同一嵌入空间，并进行L2归一化
        z_old = F.normalize(self.single_projection(old_feats), p=2, dim=1)
        z_new = F.normalize(self.single_projection(new_feats), p=2, dim=1)

        # 对比损失（配对距离）
        loss_contrast = contrastive_loss(
            z_old, z_new, labels,
            margin=self.cfg.margin,
            pos_weight=self.cfg.pos_weight,
            neg_weight=self.cfg.neg_weight,
        )

        # 对称相关约束（原文基于相关矩阵的对称性）
        loss_sym_old = symmetric_constraint_from_corr(old_corr)
        loss_sym_new = symmetric_constraint_from_corr(new_corr)
        loss_sym = 0.5 * (loss_sym_old + loss_sym_new)

        loss_total = loss_contrast + self.cfg.lambda_symmetric * loss_sym

        losses = {
            "total": loss_total,
            "contrastive": loss_contrast,
            "symmetric": loss_sym,
        }

        return z_old, z_new, losses