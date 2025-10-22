#!/usr/bin/env python3
"""
TSADCNN Demo Script

演示如何使用TSADCNN进行轨迹段关联
包括：
1. 数据生成
2. 模型训练
3. 轨迹关联
4. 结果可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from typing import List, Tuple
import logging

from models import TSADCNN
from utils.data_loader import TrackDataset
from utils.augmentation import TrackAugmentation
from utils.metrics import compute_association_metrics


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def generate_demo_data(num_targets: int = 10, segments_per_target: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成演示用的轨迹数据
    
    Args:
        num_targets: 目标数量
        segments_per_target: 每个目标的轨迹段数量
        
    Returns:
        segments: 轨迹段数据 [num_samples, sequence_length, input_dim]
        labels: 标签 [num_samples]
    """
    logging.info(f"Generating demo data: {num_targets} targets, {segments_per_target} segments per target")
    
    sequence_length = 10
    input_dim = 4
    
    all_segments = []
    all_labels = []
    
    for target_id in range(num_targets):
        # 为每个目标生成基础轨迹参数
        np.random.seed(target_id * 42)  # 确保可重复性
        
        base_x = np.random.uniform(-50, 50)
        base_y = np.random.uniform(-50, 50)
        base_vx = np.random.uniform(-5, 5)
        base_vy = np.random.uniform(-5, 5)
        
        for segment_id in range(segments_per_target):
            # 为每个轨迹段添加一些变化
            start_x = base_x + np.random.normal(0, 2)
            start_y = base_y + np.random.normal(0, 2)
            start_vx = base_vx + np.random.normal(0, 0.5)
            start_vy = base_vy + np.random.normal(0, 0.5)
            
            # 生成轨迹段
            segment = []
            x, y, vx, vy = start_x, start_y, start_vx, start_vy
            
            for t in range(sequence_length):
                # 添加噪声和轻微的机动
                noise_x = np.random.normal(0, 0.3)
                noise_y = np.random.normal(0, 0.3)
                noise_vx = np.random.normal(0, 0.1)
                noise_vy = np.random.normal(0, 0.1)
                
                x += vx + noise_x
                y += vy + noise_y
                vx += noise_vx
                vy += noise_vy
                
                # 限制速度
                vx = np.clip(vx, -10, 10)
                vy = np.clip(vy, -10, 10)
                
                segment.append([x, y, vx, vy])
            
            all_segments.append(np.array(segment, dtype=np.float32))
            all_labels.append(target_id)
    
    return np.array(all_segments), np.array(all_labels)


def create_demo_model() -> TSADCNN:
    """创建演示用的TSADCNN模型"""
    model = TSADCNN(
        input_dim=4,
        sequence_length=10,
        encoder_hidden_dim=128,
        encoder_output_dim=64,
        projection_hidden_dim=128,
        projection_output_dim=32,
        temperature=0.07,
        encoder_layers=2,
        projection_layers=2
    )
    
    logging.info(f"Created TSADCNN model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def train_demo_model(
    model: TSADCNN,
    segments: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50
) -> TSADCNN:
    """
    训练演示模型
    
    Args:
        model: TSADCNN模型
        segments: 轨迹段数据
        labels: 标签
        epochs: 训练轮数
        
    Returns:
        trained_model: 训练好的模型
    """
    logging.info(f"Training model for {epochs} epochs...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 转换数据
    segments_tensor = torch.from_numpy(segments).float().to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(segments_tensor, return_projections=True)
        
        # 计算损失
        losses = model.compute_dual_contrastive_loss(
            temporal_proj=outputs['temporal_projection'],
            spatial_proj=outputs['spatial_projection'],
            labels=labels_tensor,
            temporal_weight=0.5,
            spatial_weight=0.5
        )
        
        # 反向传播
        total_loss = losses['total_loss']
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
    
    logging.info("Training completed!")
    return model


def demonstrate_association(
    model: TSADCNN,
    segments: np.ndarray,
    labels: np.ndarray
) -> None:
    """
    演示轨迹关联功能
    
    Args:
        model: 训练好的模型
        segments: 轨迹段数据
        labels: 标签
    """
    logging.info("Demonstrating track association...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    segments_tensor = torch.from_numpy(segments).float().to(device)
    
    # 选择几个查询样本
    query_indices = [0, 5, 10, 15, 20]  # 来自不同目标的样本
    
    with torch.no_grad():
        for i, query_idx in enumerate(query_indices):
            query_segment = segments_tensor[query_idx:query_idx+1]
            query_label = labels[query_idx]
            
            # 进行关联
            indices, distances = model.associate_tracks(
                query_segments=query_segment,
                candidate_segments=segments_tensor,
                k=5,
                distance_metric='cosine'
            )
            
            # 打印结果
            logging.info(f"\nQuery {i+1} (Index: {query_idx}, True Label: {query_label}):")
            
            neighbor_indices = indices[0]
            neighbor_distances = distances[0]
            
            for j, (idx, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
                neighbor_label = labels[idx.item()]
                correct = "✓" if neighbor_label == query_label else "✗"
                logging.info(f"  {j+1}. Index: {idx.item()}, Distance: {dist.item():.4f}, "
                           f"Label: {neighbor_label} {correct}")


def evaluate_demo_model(
    model: TSADCNN,
    segments: np.ndarray,
    labels: np.ndarray
) -> None:
    """
    评估演示模型
    
    Args:
        model: 训练好的模型
        segments: 轨迹段数据
        labels: 标签
    """
    logging.info("Evaluating model performance...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    segments_tensor = torch.from_numpy(segments).float().to(device)
    
    with torch.no_grad():
        # 获取嵌入
        outputs = model(segments_tensor, return_projections=True)
        
        embeddings = outputs['encoded_features'].cpu()
        temporal_proj = outputs['temporal_projection'].cpu()
        spatial_proj = outputs['spatial_projection'].cpu()
        labels_tensor = torch.from_numpy(labels)
        
        # 计算指标
        encoder_metrics = compute_association_metrics(embeddings, labels_tensor, k=5)
        temporal_metrics = compute_association_metrics(temporal_proj, labels_tensor, k=5)
        spatial_metrics = compute_association_metrics(spatial_proj, labels_tensor, k=5)
        
        logging.info("Performance Metrics:")
        logging.info("=" * 40)
        logging.info(f"Encoder - Accuracy: {encoder_metrics['accuracy']:.4f}, mAP: {encoder_metrics['mean_ap']:.4f}")
        logging.info(f"Temporal - Accuracy: {temporal_metrics['accuracy']:.4f}, mAP: {temporal_metrics['mean_ap']:.4f}")
        logging.info(f"Spatial - Accuracy: {spatial_metrics['accuracy']:.4f}, mAP: {spatial_metrics['mean_ap']:.4f}")


def visualize_trajectories(
    segments: np.ndarray,
    labels: np.ndarray,
    save_path: str = "demo_trajectories.png"
) -> None:
    """
    可视化轨迹数据
    
    Args:
        segments: 轨迹段数据
        labels: 标签
        save_path: 保存路径
    """
    logging.info("Visualizing trajectories...")
    
    plt.figure(figsize=(12, 8))
    
    # 选择前几个目标进行可视化
    unique_labels = np.unique(labels)[:6]  # 最多显示6个目标
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, target_id in enumerate(unique_labels):
        target_mask = labels == target_id
        target_segments = segments[target_mask]
        
        for segment in target_segments:
            x_coords = segment[:, 0]
            y_coords = segment[:, 1]
            
            plt.plot(x_coords, y_coords, 'o-', color=colors[i], 
                    alpha=0.7, linewidth=2, markersize=4,
                    label=f'Target {target_id}' if segment is target_segments[0] else "")
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Demo Trajectory Segments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Trajectory visualization saved to {save_path}")


def save_demo_config(save_path: str = "demo_config.yaml") -> None:
    """保存演示配置"""
    config = {
        'model': {
            'input_dim': 4,
            'encoder_hidden_dim': 128,
            'encoder_output_dim': 64,
            'projection_hidden_dim': 128,
            'projection_output_dim': 32,
            'temperature': 0.07,
            'encoder_layers': 2,
            'projection_layers': 2
        },
        'data': {
            'sequence_length': 10,
            'train_path': 'demo_train_data.npy',
            'val_path': 'demo_val_data.npy'
        },
        'training': {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        },
        'loss': {
            'temporal_weight': 0.5,
            'spatial_weight': 0.5
        },
        'evaluation': {
            'k_neighbors': 5,
            'distance_metric': 'cosine'
        },
        'hardware': {
            'use_gpu': True,
            'num_workers': 4
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Demo configuration saved to {save_path}")


def main():
    """主演示函数"""
    setup_logging()
    logging.info("Starting TSADCNN Demo")
    logging.info("=" * 50)
    
    # 1. 生成演示数据
    segments, labels = generate_demo_data(num_targets=10, segments_per_target=5)
    logging.info(f"Generated {len(segments)} trajectory segments for {len(np.unique(labels))} targets")
    
    # 2. 可视化轨迹
    visualize_trajectories(segments, labels)
    
    # 3. 创建和训练模型
    model = create_demo_model()
    trained_model = train_demo_model(model, segments, labels, epochs=50)
    
    # 4. 演示轨迹关联
    demonstrate_association(trained_model, segments, labels)
    
    # 5. 评估模型性能
    evaluate_demo_model(trained_model, segments, labels)
    
    # 6. 保存演示数据和配置
    np.save('demo_data.npy', {'segments': segments, 'labels': labels})
    save_demo_config()
    
    # 7. 保存模型
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'input_dim': 4,
            'sequence_length': 10,
            'encoder_hidden_dim': 128,
            'encoder_output_dim': 64,
            'projection_hidden_dim': 128,
            'projection_output_dim': 32,
            'temperature': 0.07
        }
    }, 'demo_model.pth')
    
    logging.info("Demo completed successfully!")
    logging.info("Generated files:")
    logging.info("  - demo_trajectories.png: Trajectory visualization")
    logging.info("  - demo_data.npy: Demo dataset")
    logging.info("  - demo_config.yaml: Demo configuration")
    logging.info("  - demo_model.pth: Trained model")


if __name__ == '__main__':
    main()