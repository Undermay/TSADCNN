import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, Any

from models import TSADCNN
from utils.data_loader import TrackDataset
from utils.augmentation import TrackAugmentation
from utils.metrics import compute_association_metrics
from utils.metrics import compute_scene_precision_and_ap


def setup_logging(log_dir: str) -> None:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> TSADCNN:
    """创建TSADCNN模型"""
    model_config = config['model']
    
    model = TSADCNN(
        input_dim=model_config['input_dim'],
        sequence_length=config['data']['sequence_length'],
        encoder_hidden_dim=model_config['encoder_hidden_dim'],
        encoder_output_dim=model_config['encoder_output_dim'],
        projection_hidden_dim=model_config['projection_hidden_dim'],
        projection_output_dim=model_config['projection_output_dim'],
        temperature=model_config['temperature'],
        encoder_layers=model_config['encoder_layers'],
        projection_layers=model_config['projection_layers']
    )
    
    return model


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """创建数据加载器"""
    data_config = config['data']
    augmentation_config = config['augmentation']
    model_config = config['model']
    
    # 数据增强
    augmentation = TrackAugmentation(
        noise_std=augmentation_config['noise_std'],
        dropout_prob=augmentation_config['dropout_prob'],
        time_shift_range=augmentation_config['time_shift_range']
    )
    
    # 训练数据集（不返回场景信息）
    train_dataset = TrackDataset(
        data_path=data_config['train_path'],
        sequence_length=data_config['sequence_length'],
        augmentation=augmentation,
        is_training=True,
        input_dim=model_config.get('input_dim', 4),
        return_scene_info=False
    )
    
    # 验证数据集（返回场景信息）
    val_dataset = TrackDataset(
        data_path=data_config['val_path'],
        sequence_length=data_config['sequence_length'],
        augmentation=None,
        is_training=False,
        input_dim=model_config.get('input_dim', 4),
        return_scene_info=True
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(
    model: TSADCNN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_temporal_loss = 0.0
    total_spatial_loss = 0.0
    total_cross_loss = 0.0
    num_batches = 0
    
    loss_weights = config['loss']
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (track_segments, labels) in enumerate(pbar):
        track_segments = track_segments.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(track_segments, return_projections=True)
        
        # 计算双对比学习损失
        losses = model.compute_dual_contrastive_loss(
            temporal_proj=outputs['temporal_projection'],
            spatial_proj=outputs['spatial_projection'],
            labels=labels,
            temporal_weight=loss_weights['temporal_weight'],
            spatial_weight=loss_weights['spatial_weight'],
            margin=loss_weights.get('margin', 0.2)
        )
        
        # 反向传播
        total_loss_batch = losses['total_loss']
        total_loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计损失
        total_loss += total_loss_batch.item()
        total_temporal_loss += losses['temporal_loss'].item()
        total_spatial_loss += losses['spatial_loss'].item()
        total_cross_loss += losses['cross_loss'].item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Temporal': f'{losses["temporal_loss"].item():.4f}',
            'Spatial': f'{losses["spatial_loss"].item():.4f}'
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'spatial_loss': total_spatial_loss / num_batches,
        'cross_loss': total_cross_loss / num_batches
    }


def validate_epoch(
    model: TSADCNN,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """验证一个epoch（场景级AP/P@K）"""
    model.eval()
    
    total_loss = 0.0
    total_temporal_loss = 0.0
    total_spatial_loss = 0.0
    total_cross_loss = 0.0
    num_batches = 0
    
    all_embeddings = []
    all_scene_ids = []
    all_local_ids = []
    
    loss_weights = config['loss']
    fixed_k = int(config.get('evaluation', {}).get('fixed_k', 8))
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch in pbar:
            # 支持包含场景信息的批次
            if len(batch) == 4:
                track_segments, labels, scene_ids, local_ids = batch
            else:
                track_segments, labels = batch
                scene_ids = torch.full((labels.size(0),), -1, dtype=torch.long)
                local_ids = torch.full((labels.size(0),), -1, dtype=torch.long)
            
            track_segments = track_segments.to(device)
            labels = labels.to(device)
            scene_ids = scene_ids.to(device)
            local_ids = local_ids.to(device)
            
            # 前向传播
            outputs = model(track_segments, return_projections=True)
            
            # 计算损失
            losses = model.compute_dual_contrastive_loss(
                temporal_proj=outputs['temporal_projection'],
                spatial_proj=outputs['spatial_projection'],
                labels=labels,
                temporal_weight=loss_weights['temporal_weight'],
                spatial_weight=loss_weights['spatial_weight'],
                margin=loss_weights.get('margin', 0.2)
            )
            
            # 统计损失
            total_loss += losses['total_loss'].item()
            total_temporal_loss += losses['temporal_loss'].item()
            total_spatial_loss += losses['spatial_loss'].item()
            total_cross_loss += losses['cross_loss'].item()
            num_batches += 1
            
            # 收集嵌入与场景信息
            all_embeddings.append(outputs['encoded_features'].cpu().numpy())
            all_scene_ids.append(scene_ids.cpu().numpy())
            all_local_ids.append(local_ids.cpu().numpy())
            
            pbar.set_postfix({
                'Val Loss': f'{losses["total_loss"].item():.4f}'
            })
    
    # 计算场景级AP/P@K
    embeddings = np.concatenate(all_embeddings, axis=0)
    scene_ids = np.concatenate(all_scene_ids, axis=0)
    local_ids = np.concatenate(all_local_ids, axis=0)
    scene_metrics = compute_scene_precision_and_ap(embeddings, scene_ids, local_ids, fixed_k=fixed_k)
    
    validation_results = {
        'total_loss': total_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'spatial_loss': total_spatial_loss / num_batches,
        'cross_loss': total_cross_loss / num_batches,
        'ap_scene': scene_metrics['ap_scene'],
        'p_at_k': scene_metrics['p_at_k']
    }
    
    return validation_results


def save_checkpoint(
    model: TSADCNN,
    optimizer: optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config
    }
    
    # 保存最新检查点
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
    
    # 保存最佳检查点
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
        logging.info(f"Saved best checkpoint at epoch {epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train TSADCNN')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config['logging']['log_dir'])
    logging.info("Starting TSADCNN training")
    logging.info(f"Config: {config}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['use_gpu'] else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config)
    logging.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        logging.info(f"Resumed training from epoch {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, config['training']['epochs']):
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # 验证
        val_metrics = validate_epoch(model, val_loader, device, config)
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标（更新为AP/P@K）
        logging.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                    f"Temporal: {train_metrics['temporal_loss']:.4f}, "
                    f"Spatial: {train_metrics['spatial_loss']:.4f}")
        
        logging.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                    f"SceneAP: {val_metrics.get('ap_scene', 0):.4f}, "
                    f"P@K: {val_metrics.get('p_at_k', 0):.4f}")
        
        # 保存检查点（以场景级AP为唯一指标）
        current_metric = val_metrics.get('ap_scene', 0)
        is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
        
        save_checkpoint(
            model, optimizer, epoch, best_metric, config,
            config['logging']['checkpoint_dir'], is_best
        )
    
    logging.info("Training completed!")
    logging.info(f"Best validation AP (scene-level): {best_metric:.4f}")


if __name__ == '__main__':
    main()