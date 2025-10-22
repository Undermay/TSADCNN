import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from typing import Optional, Tuple, List, Dict, Any
import logging


class TrackDataset(Dataset):
    """
    轨迹段数据集
    
    用于加载和处理轨迹段数据，支持数据增强
    
    数据格式：
    - 每个轨迹段包含多个时间步的位置和速度信息
    - 输入维度：[x, y, vx, vy] 或 [x, y, vx, vy, ax, ay]
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 10,
        augmentation: Optional['TrackAugmentation'] = None,
        is_training: bool = True,
        input_dim: int = 4
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            sequence_length: 轨迹段长度
            augmentation: 数据增强器
            is_training: 是否为训练模式
            input_dim: 输入维度
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.augmentation = augmentation
        self.is_training = is_training
        self.input_dim = input_dim
        
        # 加载数据
        self.track_segments, self.labels = self._load_data()
        
        logging.info(f"Loaded {len(self.track_segments)} track segments from {data_path}")
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        加载轨迹数据
        
        Returns:
            track_segments: 轨迹段列表
            labels: 标签列表
        """
        if not os.path.exists(self.data_path):
            # 如果数据文件不存在，生成模拟数据（默认使用论文仿真模型）
            logging.warning(f"Data file {self.data_path} not found. Generating synthetic data.")
            return generate_motion_dataset(
                sequence_length=self.sequence_length,
                input_dim=self.input_dim,
                segments_per_mode=1000,  # 测试集默认规模
                segments_per_target=5
            )
        
        # 根据文件扩展名选择加载方式
        if self.data_path.endswith('.json'):
            return self._load_json_data()
        elif self.data_path.endswith('.npy'):
            return self._load_numpy_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
    
    def _load_json_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """从JSON文件加载数据"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        track_segments = []
        labels = []
        
        for item in data:
            segment = np.array(item['segment'], dtype=np.float32)
            label = item['label']
            
            # 确保序列长度一致
            if len(segment) >= self.sequence_length:
                segment = segment[:self.sequence_length]
            else:
                # 填充到指定长度
                padding = np.zeros((self.sequence_length - len(segment), self.input_dim))
                segment = np.vstack([segment, padding])
            
            track_segments.append(segment)
            labels.append(label)
        
        return track_segments, labels
    
    def _load_numpy_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """从NumPy文件加载数据"""
        data = np.load(self.data_path, allow_pickle=True).item()
        
        track_segments = data['segments']
        labels = data['labels']
        # modes 字段可选，不强制使用
        
        # 处理序列长度
        processed_segments = []
        for segment in track_segments:
            if len(segment) >= self.sequence_length:
                segment = segment[:self.sequence_length]
            else:
                padding = np.zeros((self.sequence_length - len(segment), self.input_dim))
                segment = np.vstack([segment, padding])
            
            processed_segments.append(segment.astype(np.float32))
        
        return processed_segments, labels
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.track_segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            segment: 轨迹段张量 [sequence_length, input_dim]
            label: 标签张量
        """
        segment = self.track_segments[idx].copy()
        label = self.labels[idx]
        
        # 数据增强
        if self.augmentation is not None and self.is_training:
            segment = self.augmentation(segment)
        
        # 转换为张量
        segment_tensor = torch.from_numpy(segment).float()
        label_tensor = torch.tensor(label).long()
        return segment_tensor, label_tensor


# ------------------ 论文仿真数据生成器 ------------------

def generate_motion_dataset(
    sequence_length: int,
    input_dim: int,
    segments_per_mode: int,
    segments_per_target: int = 5,
    noise_std_pos: float = 0.3,
    noise_std_vel: float = 0.1
) -> Tuple[List[np.ndarray], List[int]]:
    """
    按论文要求生成多运动模型数据集：
    - 匀速（CV）
    - 匀加速（CA）
    - 小转弯（0-60°）
    - 中转弯（60-120°）
    - 大转弯（120-180°）
    
    每种运动模式生成指定数量的轨迹段，并为每个目标生成多个段，以保证对比学习正样本。
    
    Args:
        sequence_length: 序列长度
        input_dim: 输入维度（4或6）
        segments_per_mode: 每种运动模式的轨迹段数量
        segments_per_target: 每个目标的轨迹段数量（用于正样本）
        noise_std_pos: 位置噪声标准差
        noise_std_vel: 速度噪声标准差
    
    Returns:
        track_segments: 轨迹段列表
        labels: 目标ID标签列表（跨模式唯一）
    """
    modes = [
        ('CV', 0.0, 0.0),             # 匀速
        ('CA', 0.0, 1.0),             # 匀加速（使用加速度幅度因子）
        ('TURN_SMALL', 60.0, 0.0),    # 小转弯：总转角0-60°
        ('TURN_MEDIUM', 120.0, 0.0),  # 中转弯：总转角60-120°
        ('TURN_LARGE', 180.0, 0.0)    # 大转弯：总转角120-180°
    ]
    
    track_segments: List[np.ndarray] = []
    labels: List[int] = []
    
    global_target_id = 0
    
    for mode_idx, (mode_name, max_turn_deg, accel_factor) in enumerate(modes):
        targets_count = segments_per_mode // segments_per_target
        for target_local_id in range(targets_count):
            # 为同一目标设定基础参数，保证同目标段相似
            rng = np.random.default_rng(seed=(mode_idx * 100000 + target_local_id))
            base_x = rng.uniform(-100, 100)
            base_y = rng.uniform(-100, 100)
            speed = rng.uniform(2.0, 10.0)
            heading = rng.uniform(-np.pi, np.pi)
            base_vx = speed * np.cos(heading)
            base_vy = speed * np.sin(heading)
            
            # 加速度（用于CA）
            ax = rng.uniform(-0.5, 0.5) * accel_factor
            ay = rng.uniform(-0.5, 0.5) * accel_factor
            
            # 转弯（用于TURN_*）
            if mode_name.startswith('TURN'):
                if mode_name == 'TURN_SMALL':
                    total_turn_deg = rng.uniform(0.0, 60.0)
                elif mode_name == 'TURN_MEDIUM':
                    total_turn_deg = rng.uniform(60.0, 120.0)
                else:
                    total_turn_deg = rng.uniform(120.0, 180.0)
                total_turn_rad = np.deg2rad(total_turn_deg)
                turn_rate = total_turn_rad / max(1, sequence_length)  # 每步转角
            else:
                turn_rate = 0.0
            
            for seg_id in range(segments_per_target):
                # 对每个段添加轻微扰动，模拟碎片化起点/速度差异
                x, y = base_x + rng.normal(0, 2.0), base_y + rng.normal(0, 2.0)
                vx, vy = base_vx + rng.normal(0, 0.3), base_vy + rng.normal(0, 0.3)
                
                segment = []
                current_heading = np.arctan2(vy, vx)
                
                for t in range(sequence_length):
                    # 噪声
                    nx = rng.normal(0, noise_std_pos)
                    ny = rng.normal(0, noise_std_pos)
                    nvx = rng.normal(0, noise_std_vel)
                    nvy = rng.normal(0, noise_std_vel)
                    
                    # 运动模型更新
                    if mode_name == 'CV':
                        # 匀速：保持速度，叠加小噪声
                        pass
                    elif mode_name == 'CA':
                        # 匀加速：速度受常值加速度影响
                        vx += ax
                        vy += ay
                    else:
                        # 转弯：速度方向逐步旋转，速率近似保持
                        current_heading += turn_rate
                        speed_now = np.hypot(vx, vy)
                        vx = speed_now * np.cos(current_heading)
                        vy = speed_now * np.sin(current_heading)
                    
                    # 应用噪声
                    vx += nvx
                    vy += nvy
                    
                    # 位置更新
                    x += vx + nx
                    y += vy + ny
                    
                    # 速度限制（防止爆炸）
                    vx = np.clip(vx, -15, 15)
                    vy = np.clip(vy, -15, 15)
                    
                    if input_dim == 4:
                        point = [x, y, vx, vy]
                    elif input_dim == 6:
                        # 简单加速度估计（基于噪声），或直接使用ax, ay
                        point = [x, y, vx, vy, ax, ay]
                    else:
                        raise ValueError(f"Unsupported input_dim: {input_dim}")
                    
                    segment.append(point)
                
                track_segments.append(np.array(segment, dtype=np.float32))
                labels.append(global_target_id)
                
            global_target_id += 1
    
    return track_segments, labels



def create_train_val_split(
    data_path: str,
    train_ratio: float = 0.8,
    save_dir: str = None
) -> Tuple[str, str]:
    """
    创建训练/验证数据分割
    
    Args:
        data_path: 原始数据路径
        train_ratio: 训练集比例
        save_dir: 保存目录
        
    Returns:
        train_path: 训练集路径
        val_path: 验证集路径
    """
    # 加载原始数据
    dataset = TrackDataset(data_path, sequence_length=10)
    
    # 分割数据
    num_samples = len(dataset)
    num_train = int(num_samples * train_ratio)
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # 创建训练和验证数据
    train_segments = [dataset.track_segments[i] for i in train_indices]
    train_labels = [dataset.labels[i] for i in train_indices]
    
    val_segments = [dataset.track_segments[i] for i in val_indices]
    val_labels = [dataset.labels[i] for i in val_indices]
    
    # 保存数据
    if save_dir is None:
        save_dir = os.path.dirname(data_path)
    
    train_path = os.path.join(save_dir, 'train_data.npy')
    val_path = os.path.join(save_dir, 'val_data.npy')
    
    np.save(train_path, {'segments': train_segments, 'labels': train_labels})
    np.save(val_path, {'segments': val_segments, 'labels': val_labels})
    
    logging.info(f"Created train/val split: {len(train_segments)}/{len(val_segments)} samples")
    
    return train_path, val_path