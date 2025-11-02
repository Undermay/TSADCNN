#!/usr/bin/env python3
"""
稳定的TSADCNN对比学习训练脚本
避免因小错误而中断训练过程
"""

import os
import sys
import argparse
import logging
import yaml
import traceback
from typing import Dict, Any, Tuple
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tsadcnn import TSADCNN, TSADCNNConfig
from utils.contrastive_data_loader import create_contrastive_data_loaders, ContrastiveTrajectoryDataset
from utils.metrics_pak import evaluate_contrastive_model_correct
 


def setup_logging(log_dir: str = "logs"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_stable.log")
    
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
    """创建模型（论文同构 TSADCNN）"""
    mcfg = config['model']
    lcfg = config.get('loss', {})

    # 类别权重（按约定 class_weights = [pos_weight, neg_weight]）
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


def train_epoch_stable(
    model: TSADCNN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_config: Dict[str, Any],
    epoch: int,
    similarity_threshold: float
) -> Dict[str, float]:
    """稳定的训练epoch函数"""
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_accuracy = 0.0
    total_accuracy_dist = 0.0
    num_batches = 0
    failed_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        try:
            # 解包数据
            if len(batch_data) == 3:
                old_segments, new_segments, labels = batch_data
            else:
                logging.warning(f"Unexpected batch data length: {len(batch_data)}")
                continue
                
            # 移动到设备
            old_segments = old_segments.to(device)
            new_segments = new_segments.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 计算损失（使用完整特征维度，与配置 input_dim 对齐）
            z_old, z_new, losses = model(old_segments, new_segments, labels)
            total_loss_batch = losses['total']
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            if 'gradient_clip_norm' in loss_config:
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_config['gradient_clip_norm'])
            
            optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                similarities = torch.sum(z_old * z_new, dim=1)
                predictions = (similarities > similarity_threshold).float()
                accuracy = (predictions == labels).float().mean()

                # 距离口径准确率：使用当前loss margin（基于嵌入欧氏距离）
                distances = torch.linalg.vector_norm(z_old - z_new, ord=2, dim=1)
                predictions_dist = (distances < loss_config['margin']).float()
                accuracy_dist = (predictions_dist == labels.float()).float().mean()
            
            # 累积指标
            total_loss += total_loss_batch.item()
            total_contrastive_loss += losses['contrastive'].item()
            total_accuracy += accuracy.item()
            total_accuracy_dist += accuracy_dist.item()
            num_batches += 1
            
            # 定期打印进度
            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Total Loss: {total_loss_batch.item():.4f}, "
                           f"Contrastive Loss: {losses['contrastive'].item():.4f}, "
                           f"Symmetric Loss: {losses['symmetric'].item():.4f}, "
                           f"Acc(sim): {accuracy.item():.4f}, Acc(dist): {accuracy_dist.item():.4f}")
            
            # 清理GPU内存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            failed_batches += 1
            logging.warning(f"Batch {batch_idx} 处理失败: {str(e)}")
            if failed_batches > len(train_loader) * 0.1:  # 如果失败批次超过10%，则报告错误
                logging.error(f"过多批次失败 ({failed_batches}/{batch_idx+1})")
            continue
    
    if num_batches == 0:
        logging.error("没有成功处理的批次！")
        return {'loss': float('inf'), 'contrastive_loss': float('inf'), 'accuracy': 0.0}
    
    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'accuracy_dist': total_accuracy_dist / num_batches
    }


def validate_epoch_stable(
    model: TSADCNN,
    val_loader: DataLoader,
    device: torch.device,
    loss_config: Dict[str, Any],
    similarity_threshold: float
) -> Dict[str, float]:
    """稳定的验证epoch函数"""
    model.eval()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_accuracy = 0.0
    total_accuracy_dist = 0.0
    num_batches = 0
    failed_batches = 0
    all_sims = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            try:
                # 解包数据
                if len(batch_data) == 3:
                    old_segments, new_segments, labels = batch_data
                else:
                    continue
                    
                # 移动到设备
                old_segments = old_segments.to(device)
                new_segments = new_segments.to(device)
                labels = labels.to(device)
                
                # 计算损失（使用完整特征维度，与配置 input_dim 对齐）
                z_old, z_new, losses = model(old_segments, new_segments, labels)

                # 计算准确率
                similarities = torch.sum(z_old * z_new, dim=1)
                predictions = (similarities > similarity_threshold).float()
                accuracy = (predictions == labels).float().mean()
                # 距离口径（嵌入欧氏距离）
                distances = torch.linalg.vector_norm(z_old - z_new, ord=2, dim=1)
                predictions_dist = (distances < loss_config['margin']).float()
                accuracy_dist = (predictions_dist == labels.float()).float().mean()
                
                # 累积指标
                total_loss += losses['total'].item()
                total_contrastive_loss += losses['contrastive'].item()
                total_accuracy += accuracy.item()
                total_accuracy_dist += accuracy_dist.item()
                num_batches += 1
                # 聚合用于后续网格搜索与AP/AUC
                all_sims.append(similarities.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
            except Exception as e:
                failed_batches += 1
                logging.warning(f"Validation batch {batch_idx} 处理失败: {str(e)}")
                continue
    
    if num_batches == 0:
        logging.error("验证阶段没有成功处理的批次！")
        return {'loss': float('inf'), 'contrastive_loss': float('inf'), 'accuracy': 0.0}
    
    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'accuracy_dist': total_accuracy_dist / num_batches,
        'all_sims': torch.cat(all_sims) if len(all_sims) > 0 else torch.tensor([]),
        'all_labels': torch.cat(all_labels) if len(all_labels) > 0 else torch.tensor([])
    }


def save_checkpoint(
    model: TSADCNN,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False
):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint at epoch {epoch}")
    
    logging.info(f"Saved checkpoint at epoch {epoch}")


def main():
    parser = argparse.ArgumentParser(description='稳定的TSADCNN对比学习训练')
    parser.add_argument('--config', type=str, default='config_improved_v4.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置中的训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖配置中的学习率')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志目录（覆盖配置）')
    parser.add_argument('--tb_dir', type=str, default=None,
                        help='TensorBoard目录（覆盖配置）')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='检查点目录（覆盖配置）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='相似度判定阈值（覆盖配置）')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅进行评估（跳过训练）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='评估使用的检查点路径（覆盖默认 best_checkpoint.pth）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)

    # 解析并覆盖训练相关参数
    if args.epochs is not None:
        config['training']['epochs'] = int(args.epochs)
    if args.lr is not None:
        config['training']['learning_rate'] = float(args.lr)
    if args.threshold is not None:
        config.setdefault('evaluation', {})
        config['evaluation']['similarity_threshold'] = float(args.threshold)

    # 解析并确定目录
    log_dir = args.log_dir or config.get('logging', {}).get('log_dir', 'logs')
    tb_dir = args.tb_dir or config.get('logging', {}).get('tensorboard_dir', 'runs/contrastive_tsadcnn_stable')
    checkpoint_dir = args.ckpt_dir or config.get('logging', {}).get('checkpoint_dir', 'checkpoints')

    # 设置日志
    setup_logging(log_dir)
    logging.info("Starting Stable Contrastive TSADCNN training")

    try:
        logging.info(f"Config loaded from {args.config}")
        logging.info(f"Resolved dirs -> log: {log_dir}, tb: {tb_dir}, ckpt: {checkpoint_dir}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # 如果需要，根据margin自动对齐相似度阈值（余弦阈值 ≈ 1 - m^2/2，假设嵌入L2归一化）
        try:
            if config.get('evaluation', {}).get('auto_threshold_from_margin', False):
                m = float(config['loss']['margin'])
                auto_thr = 1.0 - (m * m) / 2.0
                auto_thr = max(min(auto_thr, 0.999), 0.001)
                config['evaluation']['similarity_threshold'] = auto_thr
                logging.info(f"Auto-aligned similarity_threshold={auto_thr:.3f} from margin={m}")
        except Exception as e:
            logging.warning(f"Auto-threshold alignment failed: {str(e)}")

        # 创建数据加载器（归一化仅在数据加载器内部执行一次）
        logging.info("Creating data loaders...")
        dl_cfg = config.get('data', {})
        train_loader, test_loader = create_contrastive_data_loaders(
            train_path=config['data_paths']['train_data'],
            test_path=config['data_paths']['test_data'],
            batch_size=dl_cfg['batch_size'],
            num_workers=dl_cfg['num_workers'],
            group_by_scene=bool(dl_cfg.get('group_by_scene', True))
        )
        if config['data'].get('group_by_scene', True):
            logging.info("Training with scene-grouped batches (group_by_scene=True).")
        else:
            logging.info("Training with mixed batching (group_by_scene=False).")
        
        logging.info(f"Training samples: {len(train_loader.dataset)}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        
        # 创建模型
        logging.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # 评估模式（仅评估，不训练）
        if args.eval_only:
            logging.info("Eval-only mode enabled. Skipping training and running evaluation...")
            # 加载检查点（若提供）
            ckpt_path = args.checkpoint or os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            try:
                if os.path.exists(ckpt_path):
                    logging.info(f"Loading checkpoint for evaluation: {ckpt_path}")
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logging.warning(f"Checkpoint not found at {ckpt_path}; evaluating current model weights.")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint for evaluation: {str(e)}")

            # 运行详细评估
            try:
                logging.info("Performing evaluation (evaluate_contrastive_model_correct)...")
                k_values = [k for k in config['evaluation']['k_values'] if k > 1]
                eval_results = evaluate_contrastive_model_correct(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    k_values=k_values,
                    similarity_threshold=config['evaluation']['similarity_threshold']
                )
                for metric_name, metric_value in eval_results['association_metrics'].items():
                    logging.info(f"[eval-only] {metric_name}: {metric_value:.4f}")
            except Exception as e:
                logging.error(f"Eval-only detailed evaluation failed: {str(e)}")
            logging.info("Eval-only completed.")
            return
        
        # 创建优化器
        logging.info("Creating optimizer...")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        logging.info("Creating scheduler...")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('scheduler', {}).get('T_max', config['training']['epochs'])
        )
        
        # TensorBoard
        writer = SummaryWriter(tb_dir)
        
        # 训练循环
        best_accuracy = 0.0
        start_epoch = 0
        consecutive_failures = 0
        max_consecutive_failures = 5

        # 不再创建场景级评估加载器，训练仅保留论文口径评估
        
        # 恢复训练（如果指定）
        if args.resume and os.path.exists(args.resume):
            logging.info(f"Resuming from checkpoint: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint.get('scheduler_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_accuracy = checkpoint['metrics'].get('accuracy', 0.0)
                logging.info(f"Resumed training from epoch {start_epoch}")
            except Exception as e:
                logging.error(f"Failed to resume from checkpoint: {str(e)}")
                logging.info("Starting fresh training...")
        
        logging.info("Starting training loop...")
        for epoch in range(start_epoch, config['training']['epochs']):
            epoch_start_time = time.time()
            
            try:
                logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
                
                # 训练
                logging.info("Starting training epoch...")
                train_metrics = train_epoch_stable(
                    model, train_loader, optimizer, device, config['loss'], epoch,
                    config['evaluation']['similarity_threshold']
                )
                
                # 检查训练是否成功
                if train_metrics['loss'] == float('inf'):
                    consecutive_failures += 1
                    logging.error(f"Training epoch failed! Consecutive failures: {consecutive_failures}")
                    if consecutive_failures >= max_consecutive_failures:
                        logging.error("Too many consecutive failures, stopping training")
                        break
                    continue
                else:
                    consecutive_failures = 0
                
                # 验证
                logging.info("Starting validation epoch...")
                val_metrics = validate_epoch_stable(
                    model, test_loader, device, config['loss'],
                    config['evaluation']['similarity_threshold']
                )
                
                # 学习率调度
                scheduler.step()
                
                # 记录指标
                writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
                writer.add_scalar('AccuracyDist/Train', train_metrics.get('accuracy_dist', 0.0), epoch)
                writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
                writer.add_scalar('AccuracyDist/Val', val_metrics.get('accuracy_dist', 0.0), epoch)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
                
                # 打印结果
                epoch_time = time.time() - epoch_start_time
                logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
                logging.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                logging.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                
                # 保存检查点
                is_best = val_metrics['accuracy'] > best_accuracy
                if is_best:
                    best_accuracy = val_metrics['accuracy']
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}},
                    config, checkpoint_dir, is_best
                )
                
                # 每10个epoch进行详细评估
                if (epoch + 1) % 10 == 0:
                    try:
                        logging.info("Performing detailed evaluation...")
                        
                        # 过滤掉K=1（数据集中没有单目标场景）
                        k_values = [k for k in config['evaluation']['k_values'] if k > 1]
                        
                        eval_results = evaluate_contrastive_model_correct(
                            model=model,
                            test_loader=test_loader,
                            device=device,
                            k_values=k_values,
                            similarity_threshold=config['evaluation']['similarity_threshold']
                        )
                        
                        # 记录评估指标
                        for metric_name, metric_value in eval_results['association_metrics'].items():
                            writer.add_scalar(f'Evaluation/{metric_name}', metric_value, epoch)
                        
                        # 阈值网格搜索（基于验证相似度）
                        grid = config.get('evaluation', {}).get('threshold_grid', [])
                        if grid and val_metrics.get('all_sims') is not None and val_metrics.get('all_labels') is not None:
                            sims_np = val_metrics['all_sims'].cpu().numpy()
                            labels_np = val_metrics['all_labels'].cpu().numpy()
                            best_thr, best_acc, best_f1 = None, -1.0, -1.0
                            for thr in grid:
                                y_pred = (sims_np >= thr).astype(int)
                                acc = (y_pred == labels_np).mean()
                                tp = ((y_pred == 1) & (labels_np == 1)).sum()
                                fp = ((y_pred == 1) & (labels_np == 0)).sum()
                                fn = ((y_pred == 0) & (labels_np == 1)).sum()
                                precision = tp / (tp + fp + 1e-8)
                                recall = tp / (tp + fn + 1e-8)
                                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                                if acc > best_acc:
                                    best_thr, best_acc, best_f1 = thr, acc, f1
                            logging.info(f"[val] GridSearch best_thr={best_thr} acc={best_acc:.4f} f1={best_f1:.4f}")
                            writer.add_scalar('Evaluation/BestThresholdAcc', best_acc, epoch)
                            writer.add_scalar('Evaluation/BestThreshold', float(best_thr), epoch)

                        # 计算AP/AUC（每10轮）
                        try:
                            from sklearn.metrics import average_precision_score, roc_auc_score
                            if val_metrics.get('all_sims') is not None and val_metrics.get('all_labels') is not None and val_metrics['all_sims'].numel() > 0:
                                ap = average_precision_score(val_metrics['all_labels'].cpu().numpy(), val_metrics['all_sims'].cpu().numpy())
                                # AUC需至少两类
                                labels_np = val_metrics['all_labels'].cpu().numpy()
                                if labels_np.min() != labels_np.max():
                                    auc = roc_auc_score(labels_np, val_metrics['all_sims'].cpu().numpy())
                                else:
                                    auc = float('nan')
                                logging.info(f"[val] AP={ap:.4f} AUC={auc:.4f}")
                                writer.add_scalar('Evaluation/AP', ap, epoch)
                                if not np.isnan(auc):
                                    writer.add_scalar('Evaluation/AUC', auc, epoch)
                        except Exception as e:
                            logging.warning(f"AP/AUC 计算失败: {str(e)}")
                            
                        # 移除场景级匈牙利评估，仅保留正确的对比评估
                            
                    except Exception as e:
                        logging.warning(f"详细评估失败: {str(e)}")
                        # 继续训练，不因评估失败而停止
                        
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"Epoch {epoch} 训练失败: {str(e)}")
                logging.error(f"错误详情: {traceback.format_exc()}")
                logging.error(f"Consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logging.error("Too many consecutive failures, stopping training")
                    break
                else:
                    logging.info("Continuing to next epoch...")
                    continue
        
        writer.close()
        logging.info("Training completed successfully!")

        # 训练完成后进行一次最终详细评估（不影响训练结果）
        try:
            logging.info("Running final detailed evaluation after training...")
            k_values = [k for k in config['evaluation']['k_values'] if k > 1]
            final_results = evaluate_contrastive_model_correct(
                model=model,
                test_loader=test_loader,
                device=device,
                k_values=k_values,
                similarity_threshold=config['evaluation']['similarity_threshold']
            )
            for metric_name, metric_value in final_results['association_metrics'].items():
                logging.info(f"[final] {metric_name}: {metric_value:.4f}")
            logging.info("Final evaluation completed.")
        except Exception as e:
            logging.warning(f"Final evaluation failed: {str(e)}")

        # 已移除最终场景级匈牙利评估，仅保留论文口径评估输出
        
    except Exception as e:
        logging.error(f"主程序发生严重错误: {str(e)}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        raise e


if __name__ == '__main__':
    main()