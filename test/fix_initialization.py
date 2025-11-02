#!/usr/bin/env python3
"""
修复余弦相似度异常高的问题
主要解决：权重初始化、数值稳定性、梯度流问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import logging
from models.tsadcnn import TSADCNN, TSADCNNConfig
from utils.contrastive_data_loader import create_contrastive_data_loaders

def xavier_init_weights(m):
    """Xavier/Glorot初始化"""
    if isinstance(m, nn.Linear):
        # Xavier uniform初始化
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            # 小的非零偏置，避免对称性
            nn.init.uniform_(m.bias, -0.01, 0.01)
    elif isinstance(m, nn.BatchNorm1d):
        # BatchNorm参数初始化
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        # 卷积层使用He初始化
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.01, 0.01)

def he_init_weights(m):
    """He/Kaiming初始化（适合ReLU激活）"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.01, 0.01)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.01, 0.01)

def orthogonal_init_weights(m):
    """正交初始化（保持梯度流）"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.01, 0.01)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.01, 0.01)

def add_noise_regularization(embeddings, noise_std=0.01):
    """添加噪声正则化，增加嵌入多样性"""
    if embeddings.requires_grad:
        noise = torch.randn_like(embeddings) * noise_std
        return embeddings + noise
    return embeddings

def stable_cosine_similarity(x, y, eps=1e-8):
    """数值稳定的余弦相似度计算"""
    # 确保输入已归一化
    x_norm = F.normalize(x, p=2, dim=1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=1, eps=eps)
    
    # 计算余弦相似度，限制范围
    cos_sim = torch.sum(x_norm * y_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    
    return cos_sim

def stable_euclidean_distance(cos_sim, eps=1e-8):
    """数值稳定的欧几里得距离计算"""
    # 确保cos_sim在有效范围内
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    
    # 使用稳定的公式：d = sqrt(2 * (1 - cos_sim))
    distance_squared = 2.0 * (1.0 - cos_sim)
    distance_squared = torch.clamp(distance_squared, eps, 4.0)  # 限制在[eps, 4]
    
    return torch.sqrt(distance_squared)

def improved_contrastive_loss(z_old, z_new, labels, margin=0.5, pos_weight=1.0, neg_weight=1.0, eps=1e-8):
    """改进的对比损失函数，增强数值稳定性"""
    # 添加噪声正则化
    z_old = add_noise_regularization(z_old, noise_std=0.01)
    z_new = add_noise_regularization(z_new, noise_std=0.01)
    
    # 稳定的相似度计算
    cos_sim = stable_cosine_similarity(z_old, z_new, eps=eps)
    euclidean_dist = stable_euclidean_distance(cos_sim, eps=eps)
    
    # 对比损失计算
    pos_loss = labels * torch.pow(euclidean_dist, 2)
    neg_loss = (1 - labels) * torch.pow(torch.clamp(margin - euclidean_dist, min=0.0), 2)
    
    # 加权损失
    total_loss = pos_weight * pos_loss + neg_weight * neg_loss
    
    return total_loss.mean(), {
        'cos_sim_mean': cos_sim.mean().item(),
        'cos_sim_std': cos_sim.std().item(),
        'euclidean_dist_mean': euclidean_dist.mean().item(),
        'euclidean_dist_std': euclidean_dist.std().item(),
        'pos_loss_mean': pos_loss.mean().item(),
        'neg_loss_mean': neg_loss.mean().item()
    }

def test_initialization_methods():
    """测试不同初始化方法的效果"""
    print("🔧 测试不同权重初始化方法...")
    
    # 加载配置
    with open("config_improved_v4.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型配置
    model_config = TSADCNNConfig(
        input_dim=config['model']['input_dim'],
        encoder_hidden_dim=config['model']['encoder_hidden_dim'],
        encoder_output_dim=config['model']['encoder_output_dim'],
        projection_hidden_dim=config['model']['projection_hidden_dim'],
        projection_output_dim=config['model']['projection_output_dim'],
        encoder_layers=config['model']['encoder_layers'],
        projection_layers=config['model']['projection_layers'],
        dropout=config['model']['dropout'],
        sequence_length=config['data']['sequence_length'],
        share_backbone=True,
        pos_weight=1.0,
        neg_weight=1.0,
        lambda_symmetric=config['loss']['lambda_sym'],
        margin=config['loss']['margin']
    )
    
    # 加载数据
    try:
        train_loader, test_loader = create_contrastive_data_loaders(
            train_path="data/train_correct.npy",
            test_path="data/test_correct.npy",
            batch_size=config['data']['batch_size'],
            num_workers=0
        )
        print(f"✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 测试不同初始化方法
    init_methods = {
        "默认初始化": None,
        "Xavier初始化": xavier_init_weights,
        "He初始化": he_init_weights,
        "正交初始化": orthogonal_init_weights
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for method_name, init_func in init_methods.items():
        print(f"\n📊 测试 {method_name}:")
        
        # 创建模型
        model = TSADCNN(model_config).to(device)
        
        # 应用初始化
        if init_func is not None:
            model.apply(init_func)
        
        model.eval()
        
        # 测试一个批次
        try:
            batch = next(iter(train_loader))
            old_traj, new_traj, labels = batch
            old_traj = old_traj.to(device)
            new_traj = new_traj.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                # 获取嵌入
                old_emb, _ = model.encode_trajectory(old_traj)
                new_emb, _ = model.encode_trajectory(new_traj)
                
                # 计算相似度统计
                cos_sim = stable_cosine_similarity(old_emb, new_emb)
                euclidean_dist = stable_euclidean_distance(cos_sim)
                
                # 改进的损失计算
                loss, loss_stats = improved_contrastive_loss(
                    old_emb, new_emb, labels, 
                    margin=model_config.margin
                )
                
                print(f"  - 余弦相似度: 均值={cos_sim.mean():.4f}, 标准差={cos_sim.std():.4f}")
                print(f"  - 欧几里得距离: 均值={euclidean_dist.mean():.4f}, 标准差={euclidean_dist.std():.4f}")
                print(f"  - 对比损失: {loss:.6f}")
                print(f"  - 嵌入范数: 均值={old_emb.norm(dim=1).mean():.4f}")
                
                # 检查是否有NaN
                if torch.isnan(loss):
                    print(f"  ❌ 损失为NaN")
                else:
                    print(f"  ✅ 损失正常")
                
                # 分析正负样本分离度
                pos_mask = labels == 1
                neg_mask = labels == 0
                
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_sim = cos_sim[pos_mask].mean()
                    neg_sim = cos_sim[neg_mask].mean()
                    separation = pos_sim - neg_sim
                    print(f"  - 正样本相似度: {pos_sim:.4f}")
                    print(f"  - 负样本相似度: {neg_sim:.4f}")
                    print(f"  - 分离度: {separation:.4f}")
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_initialization_methods()