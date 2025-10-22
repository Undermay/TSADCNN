import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import logging


def compute_association_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    distance_metric: str = 'cosine'
) -> Dict[str, float]:
    """
    计算轨迹关联的评估指标
    
    Args:
        embeddings: 嵌入特征 [num_samples, embedding_dim]
        labels: 真实标签 [num_samples]
        k: 最近邻数量
        distance_metric: 距离度量方式
        
    Returns:
        metrics: 评估指标字典
    """
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 计算最近邻准确率
    knn_accuracy = compute_knn_accuracy(embeddings_np, labels_np, k, distance_metric)
    
    # 计算平均精度
    mean_ap = compute_mean_average_precision(embeddings_np, labels_np, distance_metric)
    
    # 计算Recall@K
    recall_at_k = compute_recall_at_k(embeddings_np, labels_np, k, distance_metric)
    
    # 计算聚类指标
    clustering_metrics = compute_clustering_metrics(embeddings_np, labels_np)
    
    # 计算距离分布指标
    distance_metrics = compute_distance_metrics(embeddings_np, labels_np, distance_metric)
    
    metrics = {
        'accuracy': knn_accuracy,
        'mean_ap': mean_ap,
        f'recall_at_{k}': recall_at_k,
        **clustering_metrics,
        **distance_metrics
    }
    
    return metrics


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    distance_metric: str = 'cosine'
) -> float:
    """
    计算K最近邻准确率
    
    Args:
        embeddings: 嵌入特征
        labels: 真实标签
        k: 最近邻数量
        distance_metric: 距离度量方式
        
    Returns:
        accuracy: K最近邻准确率
    """
    if distance_metric == 'cosine':
        # 归一化嵌入用于余弦距离
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        nbrs.fit(embeddings_norm)
        _, indices = nbrs.kneighbors(embeddings_norm)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=distance_metric)
        nbrs.fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)
    
    correct = 0
    total = 0
    
    for i in range(len(labels)):
        # 排除自己
        neighbor_indices = indices[i][1:]
        neighbor_labels = labels[neighbor_indices]
        
        # 检查是否有相同标签的邻居
        if labels[i] in neighbor_labels:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def compute_mean_average_precision(
    embeddings: np.ndarray,
    labels: np.ndarray,
    distance_metric: str = 'cosine'
) -> float:
    """
    计算平均精度均值 (mAP)
    
    Args:
        embeddings: 嵌入特征
        labels: 真实标签
        distance_metric: 距离度量方式
        
    Returns:
        mean_ap: 平均精度均值
    """
    if distance_metric == 'cosine':
        # 计算余弦相似度矩阵
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        distance_matrix = 1 - similarity_matrix
    else:
        # 计算欧几里得距离矩阵
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(embeddings, embeddings, metric=distance_metric)
    
    aps = []
    
    for i in range(len(labels)):
        # 获取排序后的距离索引（排除自己）
        distances = distance_matrix[i]
        sorted_indices = np.argsort(distances)[1:]  # 排除自己
        
        # 计算精度
        query_label = labels[i]
        relevant_mask = (labels[sorted_indices] == query_label)
        
        if np.sum(relevant_mask) == 0:
            continue
        
        # 计算平均精度
        precisions = []
        num_relevant = 0
        
        for j, is_relevant in enumerate(relevant_mask):
            if is_relevant:
                num_relevant += 1
                precision = num_relevant / (j + 1)
                precisions.append(precision)
        
        if precisions:
            ap = np.mean(precisions)
            aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    distance_metric: str = 'cosine'
) -> float:
    """
    计算Recall@K
    
    Args:
        embeddings: 嵌入特征
        labels: 真实标签
        k: 最近邻数量
        distance_metric: 距离度量方式
        
    Returns:
        recall_at_k: Recall@K值
    """
    if distance_metric == 'cosine':
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        nbrs.fit(embeddings_norm)
        _, indices = nbrs.kneighbors(embeddings_norm)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=distance_metric)
        nbrs.fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)
    
    recalls = []
    
    for i in range(len(labels)):
        query_label = labels[i]
        
        # 获取相同标签的所有样本（排除自己）
        same_label_mask = (labels == query_label)
        same_label_mask[i] = False
        num_same_label = np.sum(same_label_mask)
        
        if num_same_label == 0:
            continue
        
        # 获取前k个邻居（排除自己）
        neighbor_indices = indices[i][1:k+1]
        neighbor_labels = labels[neighbor_indices]
        
        # 计算召回率
        num_retrieved_relevant = np.sum(neighbor_labels == query_label)
        recall = num_retrieved_relevant / num_same_label
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    计算聚类相关指标
    
    Args:
        embeddings: 嵌入特征
        labels: 真实标签
        
    Returns:
        metrics: 聚类指标字典
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    try:
        # 计算轮廓系数
        silhouette = silhouette_score(embeddings, labels)
        
        # 使用K-means聚类并计算ARI
        num_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(embeddings)
        ari = adjusted_rand_score(labels, predicted_labels)
        
        return {
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari
        }
    
    except Exception as e:
        logging.warning(f"Error computing clustering metrics: {e}")
        return {
            'silhouette_score': 0.0,
            'adjusted_rand_index': 0.0
        }


def compute_distance_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    distance_metric: str = 'cosine'
) -> Dict[str, float]:
    """
    计算距离分布相关指标
    
    Args:
        embeddings: 嵌入特征
        labels: 真实标签
        distance_metric: 距离度量方式
        
    Returns:
        metrics: 距离指标字典
    """
    if distance_metric == 'cosine':
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        distance_matrix = 1 - similarity_matrix
    else:
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(embeddings, embeddings, metric=distance_metric)
    
    # 计算类内和类间距离
    intra_class_distances = []
    inter_class_distances = []
    
    unique_labels = np.unique(labels)
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            distance = distance_matrix[i, j]
            
            if labels[i] == labels[j]:
                intra_class_distances.append(distance)
            else:
                inter_class_distances.append(distance)
    
    # 计算统计指标
    if intra_class_distances and inter_class_distances:
        mean_intra_distance = np.mean(intra_class_distances)
        mean_inter_distance = np.mean(inter_class_distances)
        
        # 类间类内距离比
        distance_ratio = mean_inter_distance / (mean_intra_distance + 1e-8)
        
        return {
            'mean_intra_class_distance': mean_intra_distance,
            'mean_inter_class_distance': mean_inter_distance,
            'distance_ratio': distance_ratio
        }
    else:
        return {
            'mean_intra_class_distance': 0.0,
            'mean_inter_class_distance': 0.0,
            'distance_ratio': 0.0
        }


def compute_contrastive_loss_components(
    temporal_proj: torch.Tensor,
    spatial_proj: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> Dict[str, float]:
    """
    计算对比学习损失的各个组件
    
    Args:
        temporal_proj: 时间投影
        spatial_proj: 空间投影
        labels: 标签
        temperature: 温度参数
        
    Returns:
        loss_components: 损失组件字典
    """
    batch_size = temporal_proj.shape[0]
    
    # 计算相似度矩阵
    temporal_sim = torch.matmul(temporal_proj, temporal_proj.T) / temperature
    spatial_sim = torch.matmul(spatial_proj, spatial_proj.T) / temperature
    cross_sim = torch.matmul(temporal_proj, spatial_proj.T) / temperature
    
    # 创建正样本掩码
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T).float()
    positive_mask = positive_mask - torch.eye(batch_size, device=temporal_proj.device)
    
    # 计算各种相似度统计
    temporal_pos_sim = torch.sum(temporal_sim * positive_mask) / torch.sum(positive_mask)
    temporal_neg_sim = torch.sum(temporal_sim * (1 - positive_mask - torch.eye(batch_size, device=temporal_proj.device))) / torch.sum(1 - positive_mask - torch.eye(batch_size, device=temporal_proj.device))
    
    spatial_pos_sim = torch.sum(spatial_sim * positive_mask) / torch.sum(positive_mask)
    spatial_neg_sim = torch.sum(spatial_sim * (1 - positive_mask - torch.eye(batch_size, device=spatial_proj.device))) / torch.sum(1 - positive_mask - torch.eye(batch_size, device=spatial_proj.device))
    
    cross_pos_sim = torch.mean(torch.diag(cross_sim))
    
    return {
        'temporal_positive_similarity': temporal_pos_sim.item(),
        'temporal_negative_similarity': temporal_neg_sim.item(),
        'spatial_positive_similarity': spatial_pos_sim.item(),
        'spatial_negative_similarity': spatial_neg_sim.item(),
        'cross_positive_similarity': cross_pos_sim.item(),
        'temporal_separation': (temporal_pos_sim - temporal_neg_sim).item(),
        'spatial_separation': (spatial_pos_sim - spatial_neg_sim).item()
    }


def evaluate_association_performance(
    model,
    test_loader,
    device: torch.device,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    评估轨迹关联性能
    
    Args:
        model: TSADCNN模型
        test_loader: 测试数据加载器
        device: 设备
        k_values: 不同的k值列表
        
    Returns:
        results: 评估结果字典
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for track_segments, labels in test_loader:
            track_segments = track_segments.to(device)
            
            # 获取嵌入
            embeddings = model.get_track_embeddings(track_segments)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    # 合并所有结果
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算各种指标
    results = {}
    
    for k in k_values:
        metrics = compute_association_metrics(
            embeddings=all_embeddings,
            labels=all_labels,
            k=k,
            distance_metric='cosine'
        )
        
        for metric_name, value in metrics.items():
            if f'recall_at_{k}' in metric_name:
                results[f'recall_at_{k}'] = value
            elif metric_name == 'accuracy':
                results[f'accuracy_at_{k}'] = value
    
    # 计算整体指标
    overall_metrics = compute_association_metrics(
        embeddings=all_embeddings,
        labels=all_labels,
        k=5,
        distance_metric='cosine'
    )
    
    results.update({
        'mean_ap': overall_metrics['mean_ap'],
        'silhouette_score': overall_metrics['silhouette_score'],
        'distance_ratio': overall_metrics['distance_ratio']
    })
    
    return results