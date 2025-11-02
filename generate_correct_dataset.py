#!/usr/bin/env python3
"""
按照用户文档要求重新实现正确的数据生成逻辑
- 32步完整轨迹 → 13步old + 13步new轨迹对
- 对比学习：同目标标签1，不同目标标签0
- 场景K值：{1, 5, 10, 20}
- 运动模式：CV, CA, CT-small, CT-medium, CT-large
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Any
import yaml
from dataclasses import dataclass
import csv

@dataclass
class TrajectoryPair:
    """轨迹对数据结构"""
    old_trajectory: np.ndarray  # [13, 6] - 前13步
    new_trajectory: np.ndarray  # [13, 6] - 后13步
    old_raw: np.ndarray         # 未归一化old片段 [13, 6]
    new_raw: np.ndarray         # 未归一化新片段 [13, 6]
    label: int  # 1=同目标, 0=不同目标
    scene_id: int
    old_target_flag: int  # old轨迹的目标标识
    new_target_flag: int  # new轨迹的目标标识
    motion_mode: str

class MotionGenerator:
    """运动模式生成器"""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.modes = ['CV', 'CA', 'CT_SMALL', 'CT_MEDIUM', 'CT_LARGE']
    
    def generate_trajectory(self, mode: str, steps: int = 32, 
                          start_pos: np.ndarray = None, 
                          start_vel: np.ndarray = None) -> np.ndarray:
        """
        生成指定运动模式的轨迹
        
        Args:
            mode: 运动模式
            steps: 轨迹步数
            start_pos: 起始位置 [x, y]
            start_vel: 起始速度 [vx, vy]
            
        Returns:
            trajectory: [steps, 6] - [x, y, vx, vy, ax, ay]
        """
        if start_pos is None:
            start_pos = np.random.uniform(-50, 50, 2)
        if start_vel is None:
            speed = np.random.uniform(5, 15)  # m/s
            angle = np.random.uniform(0, 2*np.pi)
            start_vel = speed * np.array([np.cos(angle), np.sin(angle)])
        
        trajectory = np.zeros((steps, 6))
        trajectory[0] = np.concatenate([start_pos, start_vel, np.zeros(2)])
        
        for i in range(1, steps):
            pos = trajectory[i-1, :2]
            vel = trajectory[i-1, 2:4]
            acc = np.zeros(2)
            
            if mode == 'CV':
                # 恒速直线运动
                new_pos = pos + vel * self.dt
                new_vel = vel
                acc = np.zeros(2)
                
            elif mode == 'CA':
                # 恒加速度直线运动
                acc_mag = np.random.uniform(1, 3)  # m/s²
                acc_dir = vel / np.linalg.norm(vel)
                acc = acc_mag * acc_dir
                
                new_pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
                new_vel = vel + acc * self.dt
                
            elif mode == 'CT_SMALL':
                # 小角度匀速转弯 (0-60°/s)
                omega = np.random.uniform(0, np.pi/3)  # rad/s
                new_vel = self._rotate_velocity(vel, omega * self.dt)
                new_pos = pos + new_vel * self.dt
                acc = (new_vel - vel) / self.dt
                
            elif mode == 'CT_MEDIUM':
                # 中角度匀速转弯 (60-120°/s)
                omega = np.random.uniform(np.pi/3, 2*np.pi/3)  # rad/s
                new_vel = self._rotate_velocity(vel, omega * self.dt)
                new_pos = pos + new_vel * self.dt
                acc = (new_vel - vel) / self.dt
                
            elif mode == 'CT_LARGE':
                # 大角度匀速转弯 (120-180°/s)
                omega = np.random.uniform(2*np.pi/3, np.pi)  # rad/s
                new_vel = self._rotate_velocity(vel, omega * self.dt)
                new_pos = pos + new_vel * self.dt
                acc = (new_vel - vel) / self.dt
            
            trajectory[i] = np.concatenate([new_pos, new_vel, acc])
        
        return trajectory
    
    def _rotate_velocity(self, vel: np.ndarray, angle: float) -> np.ndarray:
        """旋转速度向量"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotation_matrix @ vel

class SceneGenerator:
    """场景生成器"""
    
    def __init__(self, motion_generator: MotionGenerator):
        self.motion_gen = motion_generator
        self.k_range = (2, 20)  # 每个场景的目标数量范围：2到20个
        
    def generate_scene(self, scene_id: int, k: int) -> Tuple[List[np.ndarray], List[str]]:
        """
        生成一个场景的所有轨迹，每个轨迹随机选择运动模式
        
        Args:
            scene_id: 场景ID
            k: 场景中的目标数量
            
        Returns:
            trajectories: 场景中所有轨迹的列表
            motion_modes: 每个轨迹对应的运动模式列表
        """
        trajectories = []
        motion_modes = []
        
        # 定义场景区域
        scene_center = np.random.uniform(-100, 100, 2)
        scene_radius = np.random.uniform(20, 50)
        
        for target_id in range(k):
            # 为每个目标随机选择运动模式
            motion_mode = np.random.choice(self.motion_gen.modes)
            motion_modes.append(motion_mode)
            
            # 在场景区域内随机选择起始位置
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, scene_radius)
            start_pos = scene_center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            # 生成轨迹
            trajectory = self.motion_gen.generate_trajectory(
                mode=motion_mode,
                steps=32,
                start_pos=start_pos
            )
            trajectories.append(trajectory)
            
        return trajectories, motion_modes

class TrajectoryPairGenerator:
    """轨迹对生成器"""
    
    def __init__(self):
        self.motion_gen = MotionGenerator()
        self.scene_gen = SceneGenerator(self.motion_gen)
    
    def create_trajectory_pairs(self, trajectories: List[np.ndarray], 
                              scene_id: int, motion_modes: List[str]) -> List[TrajectoryPair]:
        """
        从场景轨迹创建old-new轨迹对
        
        Args:
            trajectories: 场景中的所有轨迹
            scene_id: 场景ID
            motion_modes: 每个轨迹对应的运动模式列表
            
        Returns:
            pairs: 轨迹对列表（包含正样本和负样本）
        """
        pairs = []
        
        # 为每条轨迹创建old和new片段
        old_segments = []
        new_segments = []
        target_flags = []
        trajectory_motion_modes = []
        
        for target_id, (trajectory, motion_mode) in enumerate(zip(trajectories, motion_modes)):
            # 32步轨迹 → 13步old + 13步new (中间6步作为间隔)
            old_raw = trajectory[:13]  # 前13步（原始米单位）
            new_raw = trajectory[19:]  # 后13步（原始米单位，跳过中间6步）
            
            # 保留原始米单位片段，归一化在数据加载器阶段执行
            old_segment = old_raw.copy()
            new_segment = new_raw.copy()
            
            old_segments.append((old_segment, old_raw))
            new_segments.append((new_segment, new_raw))
            target_flags.append(target_id)
            trajectory_motion_modes.append(motion_mode)
        
        # 生成正样本对（同目标）
        for i, ((old_seg, old_raw), (new_seg, new_raw)) in enumerate(zip(old_segments, new_segments)):
            pair = TrajectoryPair(
                old_trajectory=old_seg,
                new_trajectory=new_seg,
                old_raw=old_raw,
                new_raw=new_raw,
                label=1,  # 同目标
                scene_id=scene_id,
                old_target_flag=target_flags[i],
                new_target_flag=target_flags[i],
                motion_mode=trajectory_motion_modes[i]
            )
            pairs.append(pair)
        
        # 生成负样本对（不同目标）
        # 只有当场景中有多个目标时才生成负样本
        if len(old_segments) > 1:
            num_positive = len(old_segments)
            for _ in range(num_positive):
                # 随机选择两个不同目标的片段
                old_idx = np.random.randint(0, len(old_segments))
                new_idx = np.random.randint(0, len(new_segments))
                
                # 确保选择的是不同目标
                max_attempts = 10  # 防止无限循环
                attempts = 0
                while target_flags[old_idx] == target_flags[new_idx] and attempts < max_attempts:
                    new_idx = np.random.randint(0, len(new_segments))
                    attempts += 1
                
                # 如果找到了不同目标的配对，才添加负样本
                if target_flags[old_idx] != target_flags[new_idx]:
                    # 负样本的运动模式使用old轨迹的运动模式
                    pair = TrajectoryPair(
                        old_trajectory=old_segments[old_idx][0],
                        new_trajectory=new_segments[new_idx][0],
                        old_raw=old_segments[old_idx][1],
                        new_raw=new_segments[new_idx][1],
                        label=0,  # 不同目标
                        scene_id=scene_id,
                        old_target_flag=target_flags[old_idx],
                        new_target_flag=target_flags[new_idx],
                        motion_mode=trajectory_motion_modes[old_idx]
                    )
                    pairs.append(pair)
        
        return pairs
    
    def generate_dataset(self, num_trajectories_per_mode: int, 
                        dataset_type: str = 'train',
                        test_k_values: List[int] | None = None) -> List[TrajectoryPair]:
        """
        生成完整数据集
        
        Args:
            num_trajectories_per_mode: 每种运动模式的轨迹数量
            dataset_type: 数据集类型 ('train' 或 'test')
            
        Returns:
            all_pairs: 所有轨迹对的列表
        """
        all_pairs = []
        scene_id = 0
        
        modes = self.motion_gen.modes
        k_range = self.scene_gen.k_range

        if test_k_values is None:
            # 按要求：测试集包含5种场景K=2、5、10、15、20
            test_k_values = [2, 5, 10, 15, 20]
        
        print(f"生成{dataset_type}数据集:")
        print(f"  运动模式: {modes}")
        print(f"  每种模式轨迹数: {num_trajectories_per_mode}")
        if dataset_type == 'test':
            print(f"  场景K值集合(测试集): {test_k_values}")
        else:
            print(f"  场景K值范围(训练集): {k_range}")
        
        for mode in modes:
            trajectories_generated = 0
            
            while trajectories_generated < num_trajectories_per_mode:
                # 训练集使用范围采样，测试集使用指定集合采样
                if dataset_type == 'test':
                    k = int(np.random.choice(test_k_values))
                else:
                    k = np.random.randint(k_range[0], k_range[1] + 1)
                
                # 生成场景轨迹（每个轨迹的运动模式随机选择）
                scene_trajectories, motion_modes = self.scene_gen.generate_scene(
                    scene_id=scene_id,
                    k=k
                )
                
                # 创建轨迹对
                scene_pairs = self.create_trajectory_pairs(
                    trajectories=scene_trajectories,
                    scene_id=scene_id,
                    motion_modes=motion_modes
                )
                
                all_pairs.extend(scene_pairs)
                trajectories_generated += k
                scene_id += 1
                
                if trajectories_generated % 100 == 0:
                    print(f"  {mode}: 已生成 {trajectories_generated}/{num_trajectories_per_mode} 条轨迹")
        
        print(f"数据集生成完成: 总共 {len(all_pairs)} 个轨迹对")
        return all_pairs

def save_dataset(pairs: List[TrajectoryPair], save_path: str):
    """保存数据集：支持 .npy 或 .csv（raw-only）。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    ext = os.path.splitext(save_path)[1].lower()
    if ext == '.csv':
        export_dataset_to_csv(pairs, save_path)
        return

    # 转换为numpy数组格式
    old_trajectories = np.array([pair.old_trajectory for pair in pairs])
    new_trajectories = np.array([pair.new_trajectory for pair in pairs])
    # 原始米单位片段（若存在）
    try:
        old_raw_trajectories = np.array([pair.old_raw for pair in pairs])
        new_raw_trajectories = np.array([pair.new_raw for pair in pairs])
    except AttributeError:
        old_raw_trajectories = None
        new_raw_trajectories = None
    labels = np.array([pair.label for pair in pairs])
    scene_ids = np.array([pair.scene_id for pair in pairs])
    old_flags = np.array([pair.old_target_flag for pair in pairs])
    new_flags = np.array([pair.new_target_flag for pair in pairs])
    motion_modes = np.array([pair.motion_mode for pair in pairs])

    # 统计信息
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    unique_scenes = len(np.unique(scene_ids))

    dataset = {
        'old_trajectories': old_trajectories,
        'new_trajectories': new_trajectories,
        # 可视化用：原始坐标（米）
        **({ 'old_raw_trajectories': old_raw_trajectories } if old_raw_trajectories is not None else {}),
        **({ 'new_raw_trajectories': new_raw_trajectories } if new_raw_trajectories is not None else {}),
        'labels': labels,
        'scene_ids': scene_ids,
        'old_target_flags': old_flags,
        'new_target_flags': new_flags,
        'motion_modes': motion_modes,
        'meta': {
            'num_pairs': len(pairs),
            'num_positive_pairs': num_positive,
            'num_negative_pairs': num_negative,
            'num_scenes': unique_scenes,
            'trajectory_length': 13,  # old和new片段的长度
            'feature_dim': 6,  # [x, y, vx, vy, ax, ay]
            'k_range': (2, 20),  # 场景大小范围
            'motion_modes': ['CV', 'CA', 'CT_SMALL', 'CT_MEDIUM', 'CT_LARGE']
        }
    }

    np.save(save_path, dataset)
    print(f"数据集已保存: {save_path}")
    print(f"  轨迹对数量: {len(pairs)}")
    print(f"  正样本: {num_positive}, 负样本: {num_negative}")
    print(f"  场景数量: {unique_scenes}")

def export_dataset_to_csv(pairs: List[TrajectoryPair], save_csv_path: str) -> None:
    """导出纯原始米单位CSV，仅包含 old_raw/new_raw 与元数据。"""
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    steps = 13
    feature_names = ['x','y','vx','vy','ax','ay']

    # 表头（不含任何归一化列）
    header = ['pair_index','scene_id','old_target_flag','new_target_flag','label','motion_mode']
    for prefix in ['old_raw','new_raw']:
        for t in range(steps):
            for fn in feature_names:
                header.append(f"{prefix}_{fn}_{t}")

    with open(save_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, pair in enumerate(pairs):
            row = [idx, pair.scene_id, pair.old_target_flag, pair.new_target_flag, pair.label, pair.motion_mode]
            for t in range(steps):
                row.extend(pair.old_raw[t].tolist())
            for t in range(steps):
                row.extend(pair.new_raw[t].tolist())
            writer.writerow(row)
    print(f"CSV已保存: {save_csv_path}")

def generate_complete_dataset():
    """生成完整的训练和测试数据集"""
    
    print("=== 按照用户文档要求生成正确的数据集 ===")
    
    generator = TrajectoryPairGenerator()
    
    # 生成训练集：每种模式2000条轨迹
    print("\n1. 生成训练集...")
    train_pairs = generator.generate_dataset(
        num_trajectories_per_mode=2000,
        dataset_type='train'
    )
    
    # 生成测试集：每种模式200条轨迹
    print("\n2. 生成测试集...")
    test_pairs = generator.generate_dataset(
        num_trajectories_per_mode=200,
        dataset_type='test'
    )
    
    # 保存数据集（直接保存为CSV，避免后续转换）
    os.makedirs('data/csv', exist_ok=True)
    save_dataset(train_pairs, 'data/csv/train_correct.csv')
    save_dataset(test_pairs, 'data/csv/test_correct.csv')
    
    # 生成配置文件
    config = {
        'dataset_info': {
            'description': '按照用户文档要求生成的正确数据集',
            'data_format': 'old-new轨迹对 + 对比学习标签',
            'trajectory_length': 32,  # 完整轨迹长度
            'segment_length': 13,     # old/new片段长度
            'gap_length': 6,          # 中间间隔长度
            'motion_modes': ['CV', 'CA', 'CT_SMALL', 'CT_MEDIUM', 'CT_LARGE'],
            'k_values': [5, 10, 20],
            'train_trajectories_per_mode': 2000,
            'test_trajectories_per_mode': 200
        },
        'files': {
            'train': 'train_correct.csv',
            'test': 'test_correct.csv'
        },
        'usage': {
            'training': '使用对比学习训练TSADCNN模型',
            'evaluation': '使用P@K和AP指标评估轨迹关联性能'
        }
    }
    
    with open('data/correct_dataset_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n=== 数据集生成完成 ===")
    print(f"配置文件: data/correct_dataset_config.yaml")

if __name__ == "__main__":
    generate_complete_dataset()