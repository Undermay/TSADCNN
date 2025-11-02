#!/usr/bin/env python3
"""
快速评估：使用已有检查点与测试集，运行 evaluate_contrastive_model_correct
用于验证评估阶段修复是否有效。
"""

import argparse
import yaml
import torch
import logging
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from train_stable import create_model
from utils.metrics_pak import evaluate_contrastive_model_correct
from utils.contrastive_data_loader import create_contrastive_data_loaders


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='Quick evaluation for TSADCNN (contrastive)')
    parser.add_argument('--config', type=str, default='config_improved_v4.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v4/best_checkpoint.pth', help='模型检查点路径')
    parser.add_argument('--threshold', type=float, default=None, help='覆盖相似度阈值')
    args = parser.parse_args()

    setup_logging()
    logging.info('Starting quick evaluation...')

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 构建模型并加载检查点
    model = create_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()
    logging.info(f'Loaded checkpoint: {args.checkpoint}')

    # 数据加载器（仅测试集，归一化仅在数据加载器内部执行一次）
    dl_cfg = config.get('data', {})
    _, test_loader = create_contrastive_data_loaders(
        train_path=config['data_paths']['train_data'],
        test_path=config['data_paths']['test_data'],
        batch_size=dl_cfg['batch_size'],
        num_workers=dl_cfg['num_workers'],
        group_by_scene=bool(dl_cfg.get('group_by_scene', True))
    )

    # 评估
    k_values = [k for k in config['evaluation']['k_values'] if k > 1]
    threshold = args.threshold if args.threshold is not None else config['evaluation']['similarity_threshold']
    logging.info(f'Running evaluation with k_values={k_values}, threshold={threshold}')

    results = evaluate_contrastive_model_correct(
        model=model,
        test_loader=test_loader,
        device=device,
        k_values=k_values,
        similarity_threshold=threshold
    )

    logging.info('Evaluation results (association_metrics):')
    for k, v in results.get('association_metrics', {}).items():
        logging.info(f'  {k}: {v:.4f}')

    logging.info('Quick evaluation completed.')


if __name__ == '__main__':
    main()