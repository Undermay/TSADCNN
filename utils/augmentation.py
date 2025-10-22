import numpy as np
from typing import Optional, Tuple
import random


class TrackAugmentation:
    """
    轨迹数据增强器
    
    提供多种轨迹数据增强方法：
    1. 高斯噪声
    2. 时间步丢弃
    3. 时间偏移
    4. 速度扰动
    5. 坐标变换
    """
    
    def __init__(
        self,
        noise_std: float = 0.1,
        dropout_prob: float = 0.1,
        time_shift_range: int = 2,
        velocity_noise_std: float = 0.05,
        coordinate_noise_std: float = 0.2,
        augmentation_prob: float = 0.8
    ):
        """
        初始化数据增强器
        
        Args:
            noise_std: 高斯噪声标准差
            dropout_prob: 时间步丢弃概率
            time_shift_range: 时间偏移范围
            velocity_noise_std: 速度噪声标准差
            coordinate_noise_std: 坐标噪声标准差
            augmentation_prob: 应用增强的概率
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.time_shift_range = time_shift_range
        self.velocity_noise_std = velocity_noise_std
        self.coordinate_noise_std = coordinate_noise_std
        self.augmentation_prob = augmentation_prob
    
    def __call__(self, track_segment: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            track_segment: 轨迹段 [sequence_length, input_dim]
            
        Returns:
            augmented_segment: 增强后的轨迹段
        """
        if random.random() > self.augmentation_prob:
            return track_segment
        
        augmented_segment = track_segment.copy()
        
        # 随机选择增强方法
        augmentation_methods = [
            self._add_gaussian_noise,
            self._apply_time_dropout,
            self._apply_time_shift,
            self._add_velocity_noise,
            self._add_coordinate_noise
        ]
        
        # 随机选择1-3种增强方法
        num_augmentations = random.randint(1, 3)
        selected_methods = random.sample(augmentation_methods, num_augmentations)
        
        for method in selected_methods:
            augmented_segment = method(augmented_segment)
        
        return augmented_segment
    
    def _add_gaussian_noise(self, segment: np.ndarray) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            segment: 轨迹段
            
        Returns:
            noisy_segment: 添加噪声后的轨迹段
        """
        noise = np.random.normal(0, self.noise_std, segment.shape)
        return segment + noise
    
    def _apply_time_dropout(self, segment: np.ndarray) -> np.ndarray:
        """
        应用时间步丢弃
        
        Args:
            segment: 轨迹段
            
        Returns:
            dropout_segment: 丢弃部分时间步后的轨迹段
        """
        sequence_length, input_dim = segment.shape
        dropout_segment = segment.copy()
        
        # 随机选择要丢弃的时间步
        num_dropout = int(sequence_length * self.dropout_prob)
        if num_dropout > 0:
            dropout_indices = np.random.choice(
                sequence_length, 
                size=num_dropout, 
                replace=False
            )
            
            # 用前一个时间步的值填充（如果是第一个时间步，用后一个填充）
            for idx in dropout_indices:
                if idx > 0:
                    dropout_segment[idx] = dropout_segment[idx - 1]
                elif idx < sequence_length - 1:
                    dropout_segment[idx] = dropout_segment[idx + 1]
        
        return dropout_segment
    
    def _apply_time_shift(self, segment: np.ndarray) -> np.ndarray:
        """
        应用时间偏移
        
        Args:
            segment: 轨迹段
            
        Returns:
            shifted_segment: 时间偏移后的轨迹段
        """
        sequence_length, input_dim = segment.shape
        
        # 随机选择偏移量
        shift = random.randint(-self.time_shift_range, self.time_shift_range)
        
        if shift == 0:
            return segment
        
        shifted_segment = np.zeros_like(segment)
        
        if shift > 0:
            # 向右偏移
            shifted_segment[shift:] = segment[:-shift]
            # 用第一个值填充前面的空位
            shifted_segment[:shift] = segment[0]
        else:
            # 向左偏移
            shifted_segment[:shift] = segment[-shift:]
            # 用最后一个值填充后面的空位
            shifted_segment[shift:] = segment[-1]
        
        return shifted_segment
    
    def _add_velocity_noise(self, segment: np.ndarray) -> np.ndarray:
        """
        添加速度噪声
        
        Args:
            segment: 轨迹段
            
        Returns:
            noisy_segment: 添加速度噪声后的轨迹段
        """
        noisy_segment = segment.copy()
        
        # 假设速度在第2、3列（vx, vy）
        if segment.shape[1] >= 4:
            velocity_noise = np.random.normal(
                0, self.velocity_noise_std, 
                (segment.shape[0], 2)
            )
            noisy_segment[:, 2:4] += velocity_noise
        
        return noisy_segment
    
    def _add_coordinate_noise(self, segment: np.ndarray) -> np.ndarray:
        """
        添加坐标噪声
        
        Args:
            segment: 轨迹段
            
        Returns:
            noisy_segment: 添加坐标噪声后的轨迹段
        """
        noisy_segment = segment.copy()
        
        # 假设坐标在第0、1列（x, y）
        coordinate_noise = np.random.normal(
            0, self.coordinate_noise_std, 
            (segment.shape[0], 2)
        )
        noisy_segment[:, :2] += coordinate_noise
        
        return noisy_segment
    
    def _apply_rotation(self, segment: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        应用旋转变换
        
        Args:
            segment: 轨迹段
            angle: 旋转角度（弧度），如果为None则随机生成
            
        Returns:
            rotated_segment: 旋转后的轨迹段
        """
        if angle is None:
            angle = random.uniform(-np.pi/6, np.pi/6)  # ±30度
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        rotated_segment = segment.copy()
        
        # 旋转位置坐标
        positions = segment[:, :2]  # x, y
        rotated_positions = positions @ rotation_matrix.T
        rotated_segment[:, :2] = rotated_positions
        
        # 旋转速度向量（如果存在）
        if segment.shape[1] >= 4:
            velocities = segment[:, 2:4]  # vx, vy
            rotated_velocities = velocities @ rotation_matrix.T
            rotated_segment[:, 2:4] = rotated_velocities
        
        return rotated_segment
    
    def _apply_scaling(self, segment: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
        """
        应用缩放变换
        
        Args:
            segment: 轨迹段
            scale_factor: 缩放因子，如果为None则随机生成
            
        Returns:
            scaled_segment: 缩放后的轨迹段
        """
        if scale_factor is None:
            scale_factor = random.uniform(0.8, 1.2)
        
        scaled_segment = segment.copy()
        
        # 缩放位置坐标
        scaled_segment[:, :2] *= scale_factor
        
        # 缩放速度（如果存在）
        if segment.shape[1] >= 4:
            scaled_segment[:, 2:4] *= scale_factor
        
        return scaled_segment
    
    def _apply_translation(self, segment: np.ndarray, offset: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        应用平移变换
        
        Args:
            segment: 轨迹段
            offset: 平移偏移量 (dx, dy)，如果为None则随机生成
            
        Returns:
            translated_segment: 平移后的轨迹段
        """
        if offset is None:
            dx = random.uniform(-5, 5)
            dy = random.uniform(-5, 5)
            offset = (dx, dy)
        
        translated_segment = segment.copy()
        translated_segment[:, 0] += offset[0]  # x
        translated_segment[:, 1] += offset[1]  # y
        
        return translated_segment
    
    def apply_geometric_augmentation(self, segment: np.ndarray) -> np.ndarray:
        """
        应用几何变换增强
        
        Args:
            segment: 轨迹段
            
        Returns:
            augmented_segment: 几何变换后的轨迹段
        """
        augmented_segment = segment.copy()
        
        # 随机选择几何变换
        transformations = [
            lambda x: self._apply_rotation(x),
            lambda x: self._apply_scaling(x),
            lambda x: self._apply_translation(x)
        ]
        
        # 随机选择1-2种变换
        num_transforms = random.randint(1, 2)
        selected_transforms = random.sample(transformations, num_transforms)
        
        for transform in selected_transforms:
            augmented_segment = transform(augmented_segment)
        
        return augmented_segment
    
    def apply_temporal_augmentation(self, segment: np.ndarray) -> np.ndarray:
        """
        应用时间相关的增强
        
        Args:
            segment: 轨迹段
            
        Returns:
            augmented_segment: 时间增强后的轨迹段
        """
        augmented_segment = segment.copy()
        
        # 应用时间相关的增强
        augmented_segment = self._apply_time_dropout(augmented_segment)
        augmented_segment = self._apply_time_shift(augmented_segment)
        
        return augmented_segment