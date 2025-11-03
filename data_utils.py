import os
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

 


def _create_tensor_pair(old_traj, new_traj, label):
    """创建张量对"""
    return (
        torch.tensor(old_traj, dtype=torch.float32),
        torch.tensor(new_traj, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long)
    )


class TrajectoryDataset(Dataset):
    """
    轨迹数据集
    
    用于加载old-new轨迹对数据，支持对比学习训练
    
    数据格式：
    - 每个样本包含old_trajectory和new_trajectory
    - 标签：1表示同一目标，0表示不同目标
    """
    
    def __init__(
        self,
        data_path: str,
        is_training: bool = True
    ):
        self.data_path = data_path
        self.is_training = is_training
        
        # 数据存储
        self.trajectory_pairs = []

        # 统计信息
        self.pos_count = 0
        self.neg_count = 0

        # 加载数据
        self.trajectory_pairs = self._load_data()
        print(f"加载了 {len(self.trajectory_pairs)} 个轨迹对")
        print(f"正样本: {self.pos_count}, 负样本: {self.neg_count}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        加载轨迹对数据
        
        Returns:
            trajectory_pairs: 轨迹对列表，每个元素包含old_trajectory, new_trajectory, label等信息
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")
        
        # 仅支持由 save_dataset(...) 生成的 .npy 字典数据
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

        for i in range(len(labels)):
            old_traj = np.array(old_trajectories[i], dtype=np.float32)
            new_traj = np.array(new_trajectories[i], dtype=np.float32)
            label = int(labels[i])
            
            # 统计正负样本
            if label == 1:
                self.pos_count += 1
            else:
                self.neg_count += 1
            
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
        return _create_tensor_pair(
            pair['old_trajectory'], 
            pair['new_trajectory'], 
            pair['label']
        )

def collate_batch(batch):
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

def create_data_loaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    创建对比学习数据加载器
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 创建数据集
    train_dataset = TrajectoryDataset(
        data_path=train_path,
        is_training=True
    )
    
    test_dataset = TrajectoryDataset(
        data_path=test_path,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 确保批次大小一致
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )
    
    return train_loader, test_loader