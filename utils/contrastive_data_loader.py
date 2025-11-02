import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import csv
from typing import Tuple, List, Dict, Any
import logging
from .normalization import TrajectoryNormalizer


class ContrastiveTrajectoryDataset(Dataset):
    """
    对比学习轨迹数据集
    
    用于加载old-new轨迹对数据，支持对比学习训练
    
    数据格式：
    - 每个样本包含old_trajectory和new_trajectory
    - 标签：1表示同一目标，0表示不同目标
    
    归一化策略：
    - normalization_mode='segment'：逐段MinMax归一化（论文口径，默认）
    - normalization_mode='scene'：场景内拟合MinMax并应用到该场景的样本
    - normalization_mode='global'：在全数据上拟合一次MinMax并应用到所有样本
    """
    
    def __init__(
        self,
        data_path: str,
        is_training: bool = True,
        normalize: bool = True,
        use_minmax_normalization: bool = True,
        normalization_mode: str = 'segment'  # 只支持segment模式
    ):
        self.data_path = data_path
        self.is_training = is_training
        self.normalize = normalize
        self.use_minmax_normalization = use_minmax_normalization
        
        # 统一归一化模式：仅保留'segment'（段级，论文口径）
        if normalization_mode != 'segment':
            logging.warning(f"不支持的归一化模式 '{normalization_mode}'，强制使用 'segment' 模式")
        self.normalization_mode = 'segment'
        
        # 数据存储
        self.trajectory_pairs = []
        self.normalizer = None  # 用于segment模式：单一全局标准化器
        self.normalization_mode_meta = None  # 记录数据集内的归一化模式（若存在）

        # 统计信息
        self.pos_count = 0
        self.neg_count = 0

        # 加载数据
        self.trajectory_pairs = self._load_data()
        logging.info(f"加载了 {len(self.trajectory_pairs)} 个轨迹对")
        logging.info(f"正样本: {self.pos_count}, 负样本: {self.neg_count}")

        # 归一化处理
        if self.normalize and self.use_minmax_normalization:
            if not self._check_if_prenormalized():
                self._apply_segment_minmax_normalization()
            else:
                logging.info("数据已预归一化，跳过归一化步骤")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        加载轨迹对数据
        
        Returns:
            trajectory_pairs: 轨迹对列表，每个元素包含old_trajectory, new_trajectory, label等信息
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")
        
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == '.csv':
            return self._load_data_from_csv(self.data_path)
        
        # 默认：加载numpy数据
        data = np.load(self.data_path, allow_pickle=True).item()
        
        trajectory_pairs = []
        
        # 仅支持新格式：顶层字典，必须包含场景信息
        if not isinstance(data, dict):
            raise ValueError(
                "Dataset must be a dict saved by save_dataset(...). Old list-style format is no longer supported."
            )

        required_keys = ['old_trajectories', 'new_trajectories', 'labels', 'scene_ids']
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise KeyError(f"Dataset missing required keys: {missing}")

        old_trajectories = data['old_trajectories']
        new_trajectories = data['new_trajectories']
        labels = data['labels']
        scene_ids = data['scene_ids']
        old_flags = data.get('old_target_flags', np.zeros(len(labels)))
        new_flags = data.get('new_target_flags', np.zeros(len(labels)))
        motion_modes = data.get('motion_modes', ['unknown'] * len(labels))
        
        # 检测数据集是否已离线归一化
        meta = data.get('meta', {}) if isinstance(data.get('meta', {}), dict) else {}
        self.pre_normalized = bool(meta.get('normalized', False)) or bool(data.get('normalized', False))
        self.normalization_mode_meta = meta.get('normalization_mode', None)

        for i in range(len(labels)):
            old_traj = np.array(old_trajectories[i], dtype=np.float32)
            new_traj = np.array(new_trajectories[i], dtype=np.float32)
            label = int(labels[i])
            
            # 统计正负样本
            if label == 1:
                self.pos_count += 1
            else:
                self.neg_count += 1
            
            # 数据归一化（仅在不使用MinMax标准化、且未预归一化时进行）
            if self.normalize and not self.use_minmax_normalization and not self.pre_normalized:
                old_traj = self._normalize_trajectory(old_traj)
                new_traj = self._normalize_trajectory(new_traj)
            
            pair_data = {
                'old_trajectory': old_traj,
                'new_trajectory': new_traj,
                'label': label,
                'scene_id': int(scene_ids[i]),
                'old_target_flag': int(old_flags[i]),
                'new_target_flag': int(new_flags[i]),
                'motion_mode': motion_modes[i] if isinstance(motion_modes[i], str) else motion_modes[i].decode('utf-8')
            }
            
            trajectory_pairs.append(pair_data)
        
        return trajectory_pairs

    def _check_if_prenormalized(self) -> bool:
        """检查数据是否已预归一化"""
        return getattr(self, 'pre_normalized', False)

    def _load_data_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """从CSV加载轨迹对数据（raw-only列，自动适配特征维度）。"""
        trajectory_pairs: List[Dict[str, Any]] = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            # 通过列名推断步长
            step_candidates = []
            for name in fields:
                if name.startswith('old_raw_x_'):
                    try:
                        step_candidates.append(int(name.split('_')[-1]))
                    except Exception:
                        pass
            steps = (max(step_candidates) + 1) if step_candidates else int(13)
            # 通过列名推断特征集合，支持 ax/ay
            detected_features = set()
            for name in fields:
                if name.startswith('old_raw_'):
                    parts = name.split('_')
                    if len(parts) >= 4:
                        feat = parts[2]
                        # 确保是形如 old_raw_<feat>_<t>
                        try:
                            int(parts[-1])
                            detected_features.add(feat)
                        except Exception:
                            pass
            canonical_order = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
            feature_names = [f for f in canonical_order if f in detected_features] or ['x','y','vx','vy']
            for row in reader:
                # 解析原始轨迹
                old_raw = np.zeros((steps, len(feature_names)), dtype=np.float32)
                new_raw = np.zeros((steps, len(feature_names)), dtype=np.float32)
                for t in range(steps):
                    for j, fn in enumerate(feature_names):
                        old_raw[t, j] = float(row.get(f'old_raw_{fn}_{t}', 0.0))
                        new_raw[t, j] = float(row.get(f'new_raw_{fn}_{t}', 0.0))
                
                # 统计正负样本
                label = int(float(row.get('label', 0)))
                if label == 1:
                    self.pos_count += 1
                else:
                    self.neg_count += 1
                
                # 初始轨迹即原始轨迹，归一化在后续步骤完成
                pair_data = {
                    'old_trajectory': old_raw.copy(),
                    'new_trajectory': new_raw.copy(),
                    'label': label,
                    'scene_id': int(float(row.get('scene_id', -1))),
                    'old_target_flag': int(float(row.get('old_target_flag', 0))),
                    'new_target_flag': int(float(row.get('new_target_flag', 0))),
                    'motion_mode': row.get('motion_mode', 'unknown')
                }
                trajectory_pairs.append(pair_data)
        return trajectory_pairs

    def _fit_normalizer(self):
        """拟合段级MinMax标准化器（已弃用，直接使用段级归一化）"""
        logging.info("使用段级归一化，无需拟合全局标准化器")

    def _apply_segment_minmax_normalization(self):
        """逐段应用MinMax归一化（论文口径：old/new各自独立归一化）"""
        for pair in self.trajectory_pairs:
            pair['old_trajectory'] = self._minmax_per_segment(pair['old_trajectory'])
            pair['new_trajectory'] = self._minmax_per_segment(pair['new_trajectory'])
        logging.info("Applied per-segment MinMax normalization to trajectory pairs")

    def _minmax_per_segment(self, trajectory: np.ndarray) -> np.ndarray:
        """对单个轨迹段进行MinMax归一化（按时间维聚合每个特征）"""
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 2:
            # [seq_len, feature_dim] 之外的情况按transform兼容处理
            normalizer = TrajectoryNormalizer(feature_range=(0.0, 1.0))
            return normalizer.fit_transform(traj)
        # 计算每个特征的min/max（基于该段的所有时间点）
        feat_min = traj.min(axis=0, keepdims=True)
        feat_max = traj.max(axis=0, keepdims=True)
        feat_range = feat_max - feat_min
        feat_range[feat_range == 0] = 1.0
        return (traj - feat_min) / feat_range
    
    def _normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        轨迹归一化
        
        Args:
            trajectory: 轨迹数据 [sequence_length, feature_dim]
            
        Returns:
            normalized_trajectory: 归一化后的轨迹
        """
        if self.use_minmax_normalization:
            # 在MinMax模式下，仅使用段级归一化
            return self._minmax_per_segment(trajectory)
        elif self.normalize:
            # 使用原有的位置归一化方法
            normalized = trajectory.copy()
            if len(normalized) > 0:
                # 位置归一化（x, y坐标）
                start_pos = normalized[0, :2]
                normalized[:, :2] -= start_pos
                
                # 速度归一化（可选，这里保持原始速度）
                # 如果需要速度归一化，可以除以最大速度或使用其他方法
                
            return normalized
        else:
            return trajectory
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.trajectory_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            old_trajectory: old轨迹张量 [sequence_length, feature_dim]
            new_trajectory: new轨迹张量 [sequence_length, feature_dim]
            label: 标签张量 (1表示同一目标，0表示不同目标)
        """
        pair = self.trajectory_pairs[idx]
        
        old_trajectory = torch.tensor(pair['old_trajectory'], dtype=torch.float32)
        new_trajectory = torch.tensor(pair['new_trajectory'], dtype=torch.float32)
        label = torch.tensor(pair['label'], dtype=torch.long)
        
        return old_trajectory, new_trajectory, label
    
    def get_scene_info(self, idx: int) -> Dict[str, Any]:
        """
        获取场景信息
        
        Args:
            idx: 样本索引
            
        Returns:
            scene_info: 包含scene_id, target_flags等信息的字典
        """
        pair = self.trajectory_pairs[idx]
        return {
            'scene_id': pair['scene_id'],
            'old_target_flag': pair['old_target_flag'],
            'new_target_flag': pair['new_target_flag'],
            'motion_mode': pair['motion_mode']
        }
    
    def get_pairs_by_scene(self) -> Dict[int, List[int]]:
        """
        按场景分组获取样本索引
        
        Returns:
            scene_pairs: 场景ID -> 样本索引列表的映射
        """
        scene_pairs = {}
        for idx, pair in enumerate(self.trajectory_pairs):
            scene_id = pair['scene_id']
            if scene_id not in scene_pairs:
                scene_pairs[scene_id] = []
            scene_pairs[scene_id].append(idx)
        
        return scene_pairs


class SceneGroupedBatchSampler(Sampler):
    """
    基于场景分组的批采样器：确保每个batch只来自同一scene_id。
    - 支持按场景打乱顺序（每个epoch），以及场景内索引打乱
    - 对每个场景的样本按batch_size分块；当drop_last=True时，丢弃不足一个完整batch的尾块
    """
    def __init__(
        self,
        dataset: ContrastiveTrajectoryDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle_scenes: bool = True,
        shuffle_within_scene: bool = True,
    ) -> None:
        if not hasattr(dataset, 'get_pairs_by_scene'):
            raise ValueError("Dataset must implement get_pairs_by_scene() for SceneGroupedBatchSampler")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle_scenes = bool(shuffle_scenes)
        self.shuffle_within_scene = bool(shuffle_within_scene)

    def __iter__(self):
        import random
        scene_to_indices = self.dataset.get_pairs_by_scene()
        scenes = list(scene_to_indices.keys())
        if self.shuffle_scenes:
            random.shuffle(scenes)

        for sid in scenes:
            indices = list(scene_to_indices[sid])
            if self.shuffle_within_scene:
                random.shuffle(indices)

            # 分块生成批次
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        # 估计总批次数（用于进度显示），当drop_last=True时按整除计算
        scene_to_indices = self.dataset.get_pairs_by_scene()
        total = 0
        for indices in scene_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                # 向上取整
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def create_contrastive_data_loaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    use_minmax_normalization: bool = True,
    normalization_mode: str = 'segment',
    group_by_scene: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建对比学习数据加载器
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        normalize: 是否归一化
        use_minmax_normalization: 是否使用0-1标准化
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 创建训练数据集
    train_dataset = ContrastiveTrajectoryDataset(
        data_path=train_path,
        is_training=True,
        normalize=normalize,
        use_minmax_normalization=use_minmax_normalization,
        normalization_mode=normalization_mode
    )
    
    # 创建测试数据集
    test_dataset = ContrastiveTrajectoryDataset(
        data_path=test_path,
        is_training=False,
        normalize=normalize,
        use_minmax_normalization=use_minmax_normalization,
        normalization_mode=normalization_mode
    )
    
    # 创建训练数据加载器：默认按scene分批，避免跨场景混合
    if group_by_scene:
        batch_sampler = SceneGroupedBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle_scenes=True,
            shuffle_within_scene=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_contrastive_batch
        )
        logging.info("Training DataLoader uses SceneGroupedBatchSampler (group_by_scene=True)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # 确保批次大小一致
            collate_fn=collate_contrastive_batch
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_contrastive_batch
    )
    
    return train_loader, test_loader


def collate_contrastive_batch(batch):
    """
    自定义批次整理函数，用于对比学习
    
    Args:
        batch: 批次数据列表
        
    Returns:
        old_trajectories: 批次old轨迹张量 [batch_size, sequence_length, feature_dim]
        new_trajectories: 批次new轨迹张量 [batch_size, sequence_length, feature_dim]
        labels: 批次标签张量 [batch_size]
    """
    old_trajectories = []
    new_trajectories = []
    labels = []
    
    for old_traj, new_traj, label in batch:
        old_trajectories.append(old_traj)
        new_trajectories.append(new_traj)
        labels.append(label)
    
    old_trajectories = torch.stack(old_trajectories)
    new_trajectories = torch.stack(new_trajectories)
    labels = torch.stack(labels)
    
    return old_trajectories, new_trajectories, labels