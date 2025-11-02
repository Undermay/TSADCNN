#!/usr/bin/env python3
"""
简化的TSADCNN对比学习训练脚本
专注于核心训练功能，减少冗余的容错和日志记录
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Dict, Any
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tsadcnn import TSADCNN, TSADCNNConfig
from utils.contrastive_data_loader import create_contrastive_data_loaders
from utils.metrics_pak import evaluate_contrastive_model_correct


def setup_logging(log_dir: str = "logs"):
    """设置简化的日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_simplified.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_model(config: Dict[str, Any]) -> TSADCNN:
    """创建模型"""
    mcfg = config['model']
    lcfg = config.get('loss', {})

    # 类别权重
    pos_weight, neg_weight = 2.0, 1.0
    if isinstance(lcfg.get('class_weights'), (list, tuple)) and len(lcfg['class_weights']) >= 2:
        pos_weight = float(lcfg['class_weights'][0])
        neg_weight = float(lcfg['class_weights'][1])

    cfg = TSADCNNConfig(
        input_dim=int(mcfg.get('input_dim', 2)),
        sequence_length=int(config.get('data', {}).get('sequence_length', 13)),
        encoder_hidden_dim=int(mcfg.get('encoder_hidden_dim', 128)),
        encoder_output_dim=int(mcfg.get('encoder_output_dim', 128)),
        encoder_layers=int(mcfg.get('encoder_layers', 2)),
        dropout=float(mcfg.get('dropout', 0.1)),
        projection_hidden_dim=int(mcfg.get('projection_hidden_dim', 128)),
        projection_output_dim=int(mcfg.get('projection_output_dim', 64)),
        projection_layers=int(mcfg.get('projection_layers', 2)),
        use_dual_projection=bool(mcfg.get('use_dual_projection', True)),
        margin=float(lcfg.get('margin', 1.0)),
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        lambda_symmetric=float(lcfg.get('lambda_sym', 0.1)),
        share_backbone=True,
    )
    return TSADCNN(cfg)


def compute_accuracy(z_old: torch.Tensor, z_new: torch.Tensor, labels: torch.Tensor, 
                    similarity_threshold: float, margin: float) -> Dict[str, float]:
    """计算准确率指标"""
    with torch.no_grad():
        # 相似度准确率
        similarities = torch.sum(z_old * z_new, dim=1)
        predictions = (similarities > similarity_threshold).float()
        accuracy_sim = (predictions == labels).float().mean()

        # 距离准确率
        distances = torch.linalg.vector_norm(z_old - z_new, ord=2, dim=1)
        predictions_dist = (distances < margin).float()
        accuracy_dist = (predictions_dist == labels.float()).float().mean()
        
        return {
            'accuracy_sim': accuracy_sim.item(),
            'accuracy_dist': accuracy_dist.item()
        }


def train_epoch(model: TSADCNN, train_loader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device, loss_config: Dict[str, Any], 
                similarity_threshold: float) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_accuracy_sim = 0.0
    total_accuracy_dist = 0.0
    num_batches = 0
    
    for batch_data in train_loader:
        # 解包数据
        old_segments, new_segments, labels = batch_data
        old_segments = old_segments.to(device)
        new_segments = new_segments.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        z_old, z_new, losses = model(old_segments, new_segments, labels)
        total_loss_batch = losses['total']
        
        # 反向传播
        total_loss_batch.backward()
        
        # 梯度裁剪
        if 'gradient_clip_norm' in loss_config:
            torch.nn.utils.clip_grad_norm_(model.parameters(), loss_config['gradient_clip_norm'])
        
        optimizer.step()
        
        # 计算准确率
        accuracies = compute_accuracy(z_old, z_new, labels, similarity_threshold, loss_config['margin'])
        
        # 累积指标
        total_loss += total_loss_batch.item()
        total_contrastive_loss += losses['contrastive'].item()
        total_accuracy_sim += accuracies['accuracy_sim']
        total_accuracy_dist += accuracies['accuracy_dist']
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'accuracy_sim': total_accuracy_sim / num_batches,
        'accuracy_dist': total_accuracy_dist / num_batches
    }


def validate_epoch(model: TSADCNN, val_loader: DataLoader, device: torch.device,
                  loss_config: Dict[str, Any], similarity_threshold: float) -> Dict[str, float]:
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_accuracy_sim = 0.0
    total_accuracy_dist = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            # 解包数据
            old_segments, new_segments, labels = batch_data
            old_segments = old_segments.to(device)
            new_segments = new_segments.to(device)
            labels = labels.to(device)
            
            # 前向传播
            z_old, z_new, losses = model(old_segments, new_segments, labels)
            
            # 计算准确率
            accuracies = compute_accuracy(z_old, z_new, labels, similarity_threshold, loss_config['margin'])
            
            # 累积指标
            total_loss += losses['total'].item()
            total_contrastive_loss += losses['contrastive'].item()
            total_accuracy_sim += accuracies['accuracy_sim']
            total_accuracy_dist += accuracies['accuracy_dist']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'accuracy_sim': total_accuracy_sim / num_batches,
        'accuracy_dist': total_accuracy_dist / num_batches
    }


def save_checkpoint(model: TSADCNN, optimizer: optim.Optimizer, scheduler, epoch: int,
                   metrics: Dict[str, float], checkpoint_dir: str, is_best: bool = False):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint at epoch {epoch}")


def load_checkpoint(checkpoint_path: str, model: TSADCNN, optimizer: optim.Optimizer, 
                   scheduler, device: torch.device) -> int:
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


def run_evaluation(model: TSADCNN, test_loader: DataLoader, device: torch.device,
                  k_values: list, similarity_threshold: float) -> Dict[str, Any]:
    """运行详细评估"""
    logging.info("Running detailed evaluation...")
    eval_results = evaluate_contrastive_model_correct(
        model=model,
        test_loader=test_loader,
        device=device,
        k_values=k_values,
        similarity_threshold=similarity_threshold
    )
    
    # 记录评估结果
    for metric_name, metric_value in eval_results['association_metrics'].items():
        logging.info(f"{metric_name}: {metric_value:.4f}")
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description='简化的TSADCNN对比学习训练')
    parser.add_argument('--config', type=str, default='config_improved_v4.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖配置中的训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='覆盖配置中的学习率')
    parser.add_argument('--eval_only', action='store_true', help='仅进行评估')
    parser.add_argument('--checkpoint', type=str, default=None, help='评估使用的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖参数
    if args.epochs is not None:
        config['training']['epochs'] = int(args.epochs)
    if args.lr is not None:
        config['training']['learning_rate'] = float(args.lr)
    
    # 设置目录
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', 'checkpoints')
    
    # 设置日志
    setup_logging(log_dir)
    logging.info("Starting Simplified Contrastive TSADCNN training")
    logging.info(f"Config loaded from {args.config}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 自动对齐相似度阈值
    if config.get('evaluation', {}).get('auto_threshold_from_margin', False):
        m = float(config['loss']['margin'])
        auto_thr = 1.0 - (m * m) / 2.0
        auto_thr = max(min(auto_thr, 0.999), 0.001)
        config['evaluation']['similarity_threshold'] = auto_thr
        logging.info(f"Auto-aligned similarity_threshold={auto_thr:.3f} from margin={m}")
    
    # 创建数据加载器
    logging.info("Creating data loaders...")
    dl_cfg = config.get('data', {})
    train_loader, test_loader = create_contrastive_data_loaders(
        train_path=config['data_paths']['train_data'],
        test_path=config['data_paths']['test_data'],
        batch_size=dl_cfg['batch_size'],
        num_workers=dl_cfg['num_workers'],
        group_by_scene=bool(dl_cfg.get('group_by_scene', True))
    )
    
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    # 创建模型
    logging.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 仅评估模式
    if args.eval_only:
        logging.info("Eval-only mode enabled")
        ckpt_path = args.checkpoint or os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        if os.path.exists(ckpt_path):
            logging.info(f"Loading checkpoint: {ckpt_path}")
            load_checkpoint(ckpt_path, model, None, None, device)
        
        k_values = [k for k in config['evaluation']['k_values'] if k > 1]
        run_evaluation(model, test_loader, device, k_values, config['evaluation']['similarity_threshold'])
        return
    
    # 创建优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('scheduler', {}).get('T_max', config['training']['epochs'])
    )
    
    # 恢复训练
    start_epoch = 0
    best_accuracy = 0.0
    
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device) + 1
        logging.info(f"Resumed training from epoch {start_epoch}")
    
    # 训练循环
    logging.info("Starting training loop...")
    similarity_threshold = config['evaluation']['similarity_threshold']
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()
        
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config['loss'], similarity_threshold
        )
        
        # 验证
        val_metrics = validate_epoch(
            model, test_loader, device, config['loss'], similarity_threshold
        )
        
        # 学习率调度
        scheduler.step()
        
        # 保存检查点
        is_best = val_metrics['accuracy_sim'] > best_accuracy
        if is_best:
            best_accuracy = val_metrics['accuracy_sim']
        
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, is_best)
        
        # 记录epoch结果
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        logging.info(f"Train Acc(sim): {train_metrics['accuracy_sim']:.4f}, Val Acc(sim): {val_metrics['accuracy_sim']:.4f}")
        logging.info(f"Train Acc(dist): {train_metrics['accuracy_dist']:.4f}, Val Acc(dist): {val_metrics['accuracy_dist']:.4f}")
        
        # 定期详细评估
        if (epoch + 1) % 10 == 0:
            k_values = [k for k in config['evaluation']['k_values'] if k > 1]
            run_evaluation(model, test_loader, device, k_values, similarity_threshold)
    
    # 最终评估
    logging.info("Training completed. Running final evaluation...")
    k_values = [k for k in config['evaluation']['k_values'] if k > 1]
    run_evaluation(model, test_loader, device, k_values, similarity_threshold)
    
    logging.info("Training finished successfully!")


if __name__ == '__main__':
    main()