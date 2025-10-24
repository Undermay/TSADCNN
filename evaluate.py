import torch
import yaml
import argparse
import os
import logging
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from models import TSADCNN
from utils.data_loader import TrackDataset
from utils.metrics import evaluate_association_performance, compute_association_metrics


def setup_logging() -> None:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> TSADCNN:
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点路径
        config: 配置字典
        device: 设备
        
    Returns:
        model: 加载的模型
    """
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
    
    # 加载检查点（PyTorch 2.6 及以上需显式关闭 weights_only）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        logging.info(f"Model trained for {checkpoint['epoch']} epochs")
    else:
        logging.info("Model checkpoint loaded (epoch information not available)")
    
    return model


def create_test_loader(config: Dict[str, Any]) -> DataLoader:
    """创建测试数据加载器"""
    data_config = config['data']
    
    test_dataset = TrackDataset(
        data_path=data_config.get('test_path', data_config['val_path']),
        sequence_length=data_config['sequence_length'],
        augmentation=None,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )
    
    return test_loader


def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    评估：严格按论文场景级AP与P@K（K=8）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集（评估阶段返回场景信息）
    test_data_path = config['dataset']['test_data_path']
    sequence_length = config['dataset']['sequence_length']
    input_dim = config['dataset']['input_dim']

    test_dataset = TrackDataset(
        data_path=test_data_path,
        sequence_length=sequence_length,
        augmentation=None,
        is_training=False,
        input_dim=input_dim,
        return_scene_info=True
    )

    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'], shuffle=False)

    # 模型
    model_cfg = config['model']
    model = TSADCNN(
        input_dim=input_dim,
        encoder_output_dim=model_cfg['encoder_output_dim'],
        projection_output_dim=model_cfg['projection_output_dim']
    ).to(device)

    if 'checkpoint_path' in config['evaluation'] and os.path.exists(config['evaluation']['checkpoint_path']):
        state = torch.load(config['evaluation']['checkpoint_path'], map_location=device)
        model.load_state_dict(state['model_state_dict'])

    model.eval()

    all_embeddings = []
    all_scene_ids = []
    all_local_ids = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating (scene-level)')
        for batch in pbar:
            # 解包（包含场景信息）
            if len(batch) == 4:
                segments, labels, scene_ids, local_ids = batch
            else:
                # 兼容旧版（无场景信息），填充为-1
                segments, labels = batch
                scene_ids = torch.full((segments.size(0),), -1, dtype=torch.long)
                local_ids = torch.full((segments.size(0),), -1, dtype=torch.long)

            segments = segments.to(device)
            embeds = model.encode(segments)  # [B, D]

            all_embeddings.append(embeds.cpu().numpy())
            all_scene_ids.append(scene_ids.cpu().numpy())
            all_local_ids.append(local_ids.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    scene_ids = np.concatenate(all_scene_ids, axis=0)
    local_ids = np.concatenate(all_local_ids, axis=0)

    # 场景级AP与P@K
    fixed_k = int(config['evaluation'].get('fixed_k', 8))
    results = compute_scene_precision_and_ap(embeddings, scene_ids, local_ids, fixed_k=fixed_k)

    print(f"Scene AP: {results['ap_scene']:.4f} | P@{fixed_k}: {results['p_at_k']:.4f}")
    return results


def evaluate_model(
    model: TSADCNN,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model: TSADCNN模型
        test_loader: 测试数据加载器
        device: 设备
        config: 配置
        
    Returns:
        results: 评估结果
    """
    logging.info("Starting model evaluation...")
    
    # 收集所有嵌入和标签
    all_embeddings = []
    all_temporal_proj = []
    all_spatial_proj = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for track_segments, labels in test_loader:
            track_segments = track_segments.to(device)
            
            # 前向传播
            outputs = model(track_segments, return_projections=True)
            
            all_embeddings.append(outputs['encoded_features'].cpu())
            all_temporal_proj.append(outputs['temporal_projection'].cpu())
            all_spatial_proj.append(outputs['spatial_projection'].cpu())
            all_labels.append(labels)
    
    # 合并结果
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_temporal_proj = torch.cat(all_temporal_proj, dim=0)
    all_spatial_proj = torch.cat(all_spatial_proj, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    logging.info(f"Collected {len(all_embeddings)} test samples")
    
    # 计算各种评估指标
    results = {}
    
    # 1. 基于编码特征的指标
    encoder_metrics = compute_association_metrics(
        embeddings=all_embeddings,
        labels=all_labels,
        k=config['evaluation']['k_neighbors'],
        distance_metric=config['evaluation']['distance_metric']
    )
    
    for key, value in encoder_metrics.items():
        results[f'encoder_{key}'] = value
    
    # 2. 基于时间投影的指标
    temporal_metrics = compute_association_metrics(
        embeddings=all_temporal_proj,
        labels=all_labels,
        k=config['evaluation']['k_neighbors'],
        distance_metric=config['evaluation']['distance_metric']
    )
    
    for key, value in temporal_metrics.items():
        results[f'temporal_{key}'] = value
    
    # 3. 基于空间投影的指标
    spatial_metrics = compute_association_metrics(
        embeddings=all_spatial_proj,
        labels=all_labels,
        k=config['evaluation']['k_neighbors'],
        distance_metric=config['evaluation']['distance_metric']
    )
    
    for key, value in spatial_metrics.items():
        results[f'spatial_{key}'] = value
    
    # 4. 不同k值的性能（可选：由配置控制）
    if config.get('evaluation', {}).get('enable_recall_curve', True):
        k_values = [1, 3, 5, 10, 20]
        performance_results = evaluate_association_performance(
            model, test_loader, device, k_values
        )
        results.update(performance_results)
    
    return results, {
        'embeddings': all_embeddings,
        'temporal_proj': all_temporal_proj,
        'spatial_proj': all_spatial_proj,
        'labels': all_labels
    }


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    title: str = "Embeddings Visualization"
) -> None:
    """
    可视化嵌入空间
    
    Args:
        embeddings: 嵌入特征
        labels: 标签
        save_dir: 保存目录
        title: 图标题
    """
    os.makedirs(save_dir, exist_ok=True)
    
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()
    
    # 限制样本数量以提高可视化速度
    max_samples = 1000
    if len(embeddings_np) > max_samples:
        indices = np.random.choice(len(embeddings_np), max_samples, replace=False)
        embeddings_np = embeddings_np[indices]
        labels_np = labels_np[indices]
    
    # t-SNE可视化
    logging.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels_np, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"{title} - t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_tsne.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA可视化
    logging.info("Computing PCA...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=labels_np, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"{title} - PCA")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.savefig(os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_pca.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_metrics(results: Dict[str, float], save_dir: str) -> None:
    """
    绘制性能指标图表
    
    Args:
        results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取不同类型的指标
    encoder_metrics = {k.replace('encoder_', ''): v for k, v in results.items() if k.startswith('encoder_')}
    temporal_metrics = {k.replace('temporal_', ''): v for k, v in results.items() if k.startswith('temporal_')}
    spatial_metrics = {k.replace('spatial_', ''): v for k, v in results.items() if k.startswith('spatial_')}
    
    # 绘制对比图
    metrics_names = ['accuracy', 'mean_ap', 'silhouette_score', 'distance_ratio']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_names):
        if metric in encoder_metrics:
            values = [
                encoder_metrics.get(metric, 0),
                temporal_metrics.get(metric, 0),
                spatial_metrics.get(metric, 0)
            ]
            labels = ['Encoder', 'Temporal', 'Spatial']
            
            axes[i].bar(labels, values, color=['blue', 'green', 'red'], alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制Recall@K曲线
    recall_keys = [k for k in results.keys() if 'recall_at_' in k]
    if recall_keys:
        k_values = []
        recall_values = []
        
        for key in sorted(recall_keys):
            k = int(key.split('_')[-1])
            k_values.append(k)
            recall_values.append(results[key])
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, recall_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('K')
        plt.ylabel('Recall@K')
        plt.title('Recall@K Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'recall_at_k.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def test_association_examples(
    model: TSADCNN,
    test_loader: DataLoader,
    device: torch.device,
    num_examples: int = 5
) -> None:
    """
    测试轨迹关联示例
    
    Args:
        model: TSADCNN模型
        test_loader: 测试数据加载器
        device: 设备
        num_examples: 示例数量
    """
    logging.info("Testing association examples...")
    
    model.eval()
    
    # 获取一批测试数据
    for track_segments, labels in test_loader:
        track_segments = track_segments.to(device)
        
        # 选择查询样本
        query_indices = torch.randperm(len(track_segments))[:num_examples]
        query_segments = track_segments[query_indices]
        query_labels = labels[query_indices]
        
        # 进行关联
        indices, distances = model.associate_tracks(
            query_segments=query_segments,
            candidate_segments=track_segments,
            k=5,
            distance_metric='cosine'
        )
        
        # 打印结果
        for i in range(num_examples):
            query_label = query_labels[i].item()
            neighbor_indices = indices[i].cpu()  # 移动到CPU
            neighbor_distances = distances[i].cpu()  # 移动到CPU
            neighbor_labels = labels[neighbor_indices]
            
            logging.info(f"Query {i+1} (Label: {query_label}):")
            logging.info(f"  Top 5 neighbors:")
            
            for j, (idx, dist, label) in enumerate(zip(neighbor_indices, neighbor_distances, neighbor_labels)):
                correct = "✓" if label.item() == query_label else "✗"
                logging.info(f"    {j+1}. Index: {idx.item()}, Distance: {dist.item():.4f}, "
                           f"Label: {label.item()} {correct}")
            
            # 计算准确率
            correct_matches = (neighbor_labels == query_label).sum().item()
            accuracy = correct_matches / len(neighbor_labels)
            logging.info(f"  Accuracy: {accuracy:.2%}\n")
        
        break  # 只测试一批


def main():
    parser = argparse.ArgumentParser(description='Evaluate TSADCNN')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logging.info("Starting TSADCNN evaluation")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['use_gpu'] else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, config, device)
    
    # 创建测试数据加载器
    test_loader = create_test_loader(config)
    logging.info(f"Created test loader with {len(test_loader)} batches")
    
    # 评估模型
    results, data = evaluate_model(model, test_loader, device, config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'evaluation_results.yaml')
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # 打印主要结果
    logging.info("Evaluation Results:")
    logging.info("=" * 50)
    
    key_metrics = [
        'encoder_accuracy', 'temporal_accuracy', 'spatial_accuracy',
        'encoder_mean_ap', 'temporal_mean_ap', 'spatial_mean_ap',
        # 可选打印 Recall@K，仅在启用时存在
        'recall_at_5', 'encoder_distance_ratio'
    ]
    
    for metric in key_metrics:
        if metric in results:
            logging.info(f"{metric}: {results[metric]:.4f}")
    
    # 可视化（如果请求）
    if args.visualize:
        logging.info("Generating visualizations...")
        
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        
        # 可视化不同类型的嵌入
        visualize_embeddings(
            data['embeddings'], data['labels'], viz_dir, "Encoder Embeddings"
        )
        visualize_embeddings(
            data['temporal_proj'], data['labels'], viz_dir, "Temporal Projections"
        )
        visualize_embeddings(
            data['spatial_proj'], data['labels'], viz_dir, "Spatial Projections"
        )
        
        # 绘制性能指标
        plot_performance_metrics(results, viz_dir)
    
    # 测试关联示例
    test_association_examples(model, test_loader, device)
    
    logging.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()