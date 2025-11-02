#!/usr/bin/env python3
"""
正确的P@K指标计算模块
按照论文定义实现场景级P@K计算：
- K代表场景中目标的数量
- P@K = 在有K个目标的场景中，正确关联目标数 / K
- 不使用"前K个预测"的错误定义
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import pairwise_distances
import logging


def _pak_for_single_scene(scene_embeddings: np.ndarray, local_ids: List[int]) -> Tuple[int, int]:
    """计算单个场景的K与正确关联目标数，使用向量化最近质心分类。"""
    if len(local_ids) == 0:
        return 0, 0
    unique_targets = np.array(sorted(set(local_ids)))
    K = int(len(unique_targets))
    if K == 0:
        return 0, 0
    # 计算每个目标的质心
    centroids = []
    for tid in unique_targets:
        mask = (np.array(local_ids) == tid)
        centroids.append(scene_embeddings[mask].mean(axis=0))
    centroids = np.stack(centroids, axis=0)  # [K, D]
    # 距离矩阵 [n_samples, K]
    diff = scene_embeddings[:, None, :] - centroids[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    pred = unique_targets[np.argmin(dists, axis=1)]  # 映射回目标ID
    # 统计每个目标是否所有样本均被正确分类
    correct_targets = 0
    for tid in unique_targets:
        mask = (np.array(local_ids) == tid)
        if np.all(pred[mask] == tid):
            correct_targets += 1
    return K, int(correct_targets)


def compute_scene_level_pak_correct(
    embeddings: np.ndarray,
    scene_ids: np.ndarray,
    local_target_ids: np.ndarray,
    k_values: Optional[List[int]] = None,
    parallel: bool = False,
    num_workers: int = 4
) -> Dict[str, float]:
    """
    按照论文定义计算正确的场景级P@K指标
    
    Args:
        embeddings: 所有样本的嵌入向量 [N, D]
        scene_ids: 每个样本的场景ID [N]
        local_target_ids: 场景内目标ID（0到K-1） [N]
        k_values: 要计算的K值列表，如果为None则计算所有出现的K值
        
    Returns:
        Dict包含各个K值对应的P@K结果
    """
    assert embeddings.ndim == 2, "embeddings必须是2维数组 [N, D]"
    N = embeddings.shape[0]
    assert len(scene_ids) == N and len(local_target_ids) == N, "scene_ids和local_target_ids长度必须与embeddings一致"
    
    # 按场景分组
    scenes = {}
    for idx in range(N):
        scene_id = int(scene_ids[idx])
        local_id = int(local_target_ids[idx])
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append((idx, local_id))
    
    # 统计每个K值对应的场景数和正确关联数
    k_stats = {}  # k -> {'total_scenes': int, 'correct_targets': int, 'total_targets': int}

    def process_one(scene_items: List[Tuple[int, int]]):
        indices = [it[0] for it in scene_items]
        local_ids = [it[1] for it in scene_items]
        scene_embeddings = embeddings[indices]
        return _pak_for_single_scene(scene_embeddings, local_ids)

    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
            results = list(ex.map(process_one, scenes.values()))
    else:
        results = [process_one(items) for items in scenes.values()]

    for K, correct_targets in results:
        if K == 0:
            continue
        if K not in k_stats:
            k_stats[K] = {'total_scenes': 0, 'correct_targets': 0, 'total_targets': 0}
        k_stats[K]['total_scenes'] += 1
        k_stats[K]['total_targets'] += K
        k_stats[K]['correct_targets'] += correct_targets
    
    # 计算P@K结果
    results = {}
    
    # 如果指定了k_values，只计算这些K值
    if k_values is not None:
        target_ks = k_values
    else:
        target_ks = sorted(k_stats.keys())
    
    for k in target_ks:
        if k in k_stats and k_stats[k]['total_targets'] > 0:
            pak = k_stats[k]['correct_targets'] / k_stats[k]['total_targets']
            results[f'P@{k}'] = pak
            logging.info(f"P@{k}: {pak:.4f} (正确关联目标: {k_stats[k]['correct_targets']}, 总目标: {k_stats[k]['total_targets']}, 场景数: {k_stats[k]['total_scenes']})")
        else:
            results[f'P@{k}'] = 0.0
            logging.warning(f"P@{k}: 0.0000 (没有K={k}的场景)")
    
    # 计算总体AP（所有场景的平均）
    total_correct = sum(stats['correct_targets'] for stats in k_stats.values())
    total_targets = sum(stats['total_targets'] for stats in k_stats.values())
    ap = total_correct / total_targets if total_targets > 0 else 0.0
    results['AP'] = ap
    
    return results


def compute_contrastive_association_metrics_correct(
    model,
    test_loader,
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
    parallel: bool = False,
    num_workers: int = 4
) -> Dict[str, float]:
    """
    使用正确的P@K定义计算对比学习模型的关联指标
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        k_values: 要计算的K值列表
        
    Returns:
        包含P@K和AP指标的字典
    """
    model.eval()
    
    all_embeddings = []
    all_scene_ids = []
    all_local_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 仅支持三元或四元批次：
            # - (old_trajectories, new_trajectories, labels)
            # - (segments, labels, scene_ids, local_ids)
            if len(batch) == 3:
                # 对比学习格式：old_trajectories, new_trajectories, labels
                old_trajectories, new_trajectories, labels = batch
                batch_size = old_trajectories.size(0)
                
                # 对于对比学习模型，需要分别处理old和new轨迹
                old_trajectories = old_trajectories.to(device)
                new_trajectories = new_trajectories.to(device)
                
                # 获取嵌入向量
                if hasattr(model, 'encode_trajectory'):
                    # 对比学习模型，分别编码；encode_trajectory 返回 (z, corr)
                    old_out = model.encode_trajectory(old_trajectories)
                    new_out = model.encode_trajectory(new_trajectories)
                    old_embeddings = old_out[0] if isinstance(old_out, (tuple, list)) else old_out
                    new_embeddings = new_out[0] if isinstance(new_out, (tuple, list)) else new_out
                    embeddings = torch.cat([old_embeddings, new_embeddings], dim=0)
                else:
                    # 标准模型，合并后编码
                    segments = torch.cat([old_trajectories, new_trajectories], dim=0)
                    embeddings = model(segments)
                
                # 获取真实的场景信息
                # 需要从数据加载器获取场景信息
                batch_start_idx = batch_idx * test_loader.batch_size
                real_scene_ids = []
                real_local_ids = []
                
                for i in range(batch_size):
                    data_idx = batch_start_idx + i
                    if data_idx < len(test_loader.dataset):
                        scene_info = test_loader.dataset.get_scene_info(data_idx)
                        scene_id = scene_info['scene_id']
                        # 对于对比学习，old和new轨迹来自同一场景但可能是不同目标
                        old_target_flag = scene_info.get('old_target_flag', 0)
                        new_target_flag = scene_info.get('new_target_flag', 1)
                        
                        real_scene_ids.extend([scene_id, scene_id])
                        real_local_ids.extend([old_target_flag, new_target_flag])
                    else:
                        # 如果索引超出范围，使用默认值
                        real_scene_ids.extend([0, 0])
                        real_local_ids.extend([0, 1])
                
                scene_ids = np.array(real_scene_ids)
                local_ids = np.array(real_local_ids)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_scene_ids.append(scene_ids)
                all_local_ids.append(local_ids)
                continue  # 跳过后面的通用处理
                
            elif len(batch) == 4:
                # 包含场景信息的格式
                segments, labels, scene_ids, local_ids = batch
                
            else:
                raise ValueError(f"不支持的批次格式，长度为{len(batch)}；仅支持(old,new,label)或(segments,label,scene_id,local_id)")
            
            segments = segments.to(device)
            
            # 获取嵌入向量
            if hasattr(model, 'encode'):
                embeddings = model.encode(segments)
            elif hasattr(model, 'encode_trajectory'):
                out = model.encode_trajectory(segments)
                embeddings = out[0] if isinstance(out, (tuple, list)) else out
            else:
                embeddings = model(segments)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_scene_ids.append(scene_ids.cpu().numpy())
            all_local_ids.append(local_ids.cpu().numpy())
    
    # 合并所有批次的结果
    embeddings = np.concatenate(all_embeddings, axis=0)
    scene_ids = np.concatenate(all_scene_ids, axis=0)
    local_ids = np.concatenate(all_local_ids, axis=0)
    
    # 计算P@K指标
    results = compute_scene_level_pak_correct(
        embeddings, scene_ids, local_ids, k_values, parallel=parallel, num_workers=num_workers
    )
    
    return results


def evaluate_contrastive_model_correct(
    model,
    test_loader,
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
    similarity_threshold: float = 0.5,
    parallel: bool = False,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    使用正确的P@K定义全面评估对比学习模型
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        k_values: K值列表
        similarity_threshold: 相似度阈值（用于准确率计算）
        
    Returns:
        完整的评估结果
    """
    logging.info("开始使用正确的P@K定义评估对比学习模型...")
    
    # 计算正确的P@K和AP指标
    association_metrics = compute_contrastive_association_metrics_correct(
        model, test_loader, device, k_values, parallel=parallel, num_workers=num_workers
    )
    
    # 计算轨迹关联准确率
    accuracy_metrics = compute_trajectory_association_accuracy_correct(
        model, test_loader, device, similarity_threshold
    )
    
    # 合并结果
    evaluation_results = {
        'association_metrics': association_metrics,
        'accuracy_metrics': accuracy_metrics,
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    # 打印主要指标
    logging.info("=== 正确P@K评估结果 ===")
    logging.info(f"AP: {association_metrics.get('AP', 0.0):.4f}")
    for k in k_values:
        p_at_k = association_metrics.get(f'P@{k}', 0.0)
        logging.info(f"P@{k}: {p_at_k:.4f}")
    
    logging.info(f"Accuracy: {accuracy_metrics['accuracy']:.4f}")
    
    return evaluation_results


def compute_trajectory_association_accuracy_correct(
    model,
    test_loader,
    device: torch.device,
    similarity_threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算轨迹关联准确率
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        similarity_threshold: 相似度阈值
        
    Returns:
        包含准确率指标的字典
    """
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 仅支持含标签的对比学习批次：(old_segments, new_segments, labels)
            if len(batch) == 3:
                # 正确支持含标签的对比学习批次
                old_segments, new_segments, labels = batch
                batch_size = old_segments.size(0)
                
                old_segments = old_segments.to(device)
                new_segments = new_segments.to(device)
                labels = labels.to(device).float()
                
                # 获取嵌入向量与相似度
                if hasattr(model, 'encode_trajectory'):
                    old_out = model.encode_trajectory(old_segments)
                    new_out = model.encode_trajectory(new_segments)
                    old_embeddings = old_out[0] if isinstance(old_out, (tuple, list)) else old_out
                    new_embeddings = new_out[0] if isinstance(new_out, (tuple, list)) else new_out
                    old_norm = torch.nn.functional.normalize(old_embeddings, p=2, dim=1)
                    new_norm = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
                    similarities = torch.sum(old_norm * new_norm, dim=1)
                else:
                    outputs = model(old_segments, new_segments, return_similarity=True)
                    similarities = outputs['similarity']
                
                predictions = (similarities > similarity_threshold).float()
                
                correct_predictions += torch.sum(predictions == labels).item()
                total_predictions += batch_size
                
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                raise ValueError(f"不支持的批次格式，长度为{len(batch)}；仅支持(old,new,label)格式")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_samples': total_predictions,
        'correct_predictions': correct_predictions
    }


def compute_scene_precision_and_ap_correct(
    embeddings: np.ndarray,
    scene_ids: np.ndarray,
    local_target_ids: np.ndarray,
    k_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    按照论文定义计算场景级精度P@K和总体AP的正确实现
    这是utils/metrics.py中compute_scene_precision_and_ap函数的正确版本
    
    Args:
        embeddings: 所有样本的嵌入向量 [N, D]
        scene_ids: 每个样本的场景ID [N]
        local_target_ids: 场景内目标ID（0到K-1） [N]
        k_values: 要计算的K值列表
        
    Returns:
        包含ap_scene和各K值P@K的字典
    """
    results = compute_scene_level_pak_correct(embeddings, scene_ids, local_target_ids, k_values)
    
    # 转换格式以兼容现有代码
    formatted_results = {
        'ap_scene': results['AP']
    }
    
    # 添加各个K值的P@K
    for key, value in results.items():
        if key.startswith('P@'):
            k = key.split('@')[1]
            formatted_results[f'p_at_k_{k}'] = value
    
    return formatted_results


# 为了向后兼容，提供与原始函数相同的接口
def compute_scene_level_pak_legacy_compatible(
    scene_data: Dict[int, List[Dict[str, Any]]],
    k_values: List[int]
) -> Dict[str, float]:
    """
    为了与现有代码兼容而提供的包装函数
    将scene_data格式转换为新的计算函数所需的格式
    
    Args:
        scene_data: 场景数据字典，格式为 {scene_id: [{'embedding': np.ndarray, 'target_id': int}, ...]}
        k_values: 要计算的K值列表
        
    Returns:
        P@K结果字典
    """
    # 转换数据格式
    all_embeddings = []
    all_scene_ids = []
    all_local_ids = []
    
    for scene_id, items in scene_data.items():
        for item in items:
            all_embeddings.append(item['embedding'])
            all_scene_ids.append(scene_id)
            all_local_ids.append(item['target_id'])
    
    if len(all_embeddings) == 0:
        return {f'P@{k}': 0.0 for k in k_values}
    
    embeddings = np.stack(all_embeddings)
    scene_ids = np.array(all_scene_ids)
    local_ids = np.array(all_local_ids)
    
    return compute_scene_level_pak_correct(embeddings, scene_ids, local_ids, k_values)