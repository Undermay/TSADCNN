#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrajectoryNormalizer:
    """
    轨迹数据0-1标准化处理器
    
    支持对轨迹数据进行Min-Max标准化，将数据缩放到[0,1]范围内
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        """
        初始化标准化器
        
        Args:
            feature_range: 标准化后的数据范围，默认为(0, 1)
        """
        self.feature_range = feature_range
        self.min_vals = None
        self.max_vals = None
        self.fitted = False
        
    def fit(self, trajectories: Union[np.ndarray, torch.Tensor]) -> 'TrajectoryNormalizer':
        """
        拟合标准化参数
        
        Args:
            trajectories: 轨迹数据，形状为 [N, sequence_length, feature_dim] 或 [sequence_length, feature_dim]
            
        Returns:
            self: 返回自身以支持链式调用
        """
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.numpy()
            
        # 确保数据是3维的
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, ...]
        elif trajectories.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {trajectories.ndim}D")
            
        # 重塑数据为 [N*sequence_length, feature_dim]
        reshaped = trajectories.reshape(-1, trajectories.shape[-1])
        
        # 计算每个特征维度的最小值和最大值
        self.min_vals = np.min(reshaped, axis=0)
        self.max_vals = np.max(reshaped, axis=0)
        
        # 避免除零错误
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1.0
        
        self.fitted = True
        logger.info(f"Normalizer fitted with min_vals: {self.min_vals}, max_vals: {self.max_vals}")
        
        return self
    
    def transform(self, trajectories: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        应用标准化变换
        
        Args:
            trajectories: 待标准化的轨迹数据
            
        Returns:
            normalized_trajectories: 标准化后的轨迹数据
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        is_tensor = isinstance(trajectories, torch.Tensor)
        device = trajectories.device if is_tensor else None
        
        if is_tensor:
            trajectories = trajectories.cpu().numpy()
            
        original_shape = trajectories.shape
        
        # 确保数据是3维的
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, ...]
        elif trajectories.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {trajectories.ndim}D")
            
        # 重塑数据
        reshaped = trajectories.reshape(-1, trajectories.shape[-1])
        
        # 应用Min-Max标准化: (x - min) / (max - min) * (max_range - min_range) + min_range
        min_range, max_range = self.feature_range
        normalized = (reshaped - self.min_vals) / self.range_vals
        normalized = normalized * (max_range - min_range) + min_range
        
        # 恢复原始形状
        normalized = normalized.reshape(trajectories.shape)
        
        # 如果原始输入是2维的，去掉添加的维度
        if len(original_shape) == 2:
            normalized = normalized[0]
            
        # 转换回tensor（如果需要）
        if is_tensor:
            normalized = torch.from_numpy(normalized).to(device)
            
        return normalized
    
    def fit_transform(self, trajectories: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        拟合并应用标准化变换
        
        Args:
            trajectories: 轨迹数据
            
        Returns:
            normalized_trajectories: 标准化后的轨迹数据
        """
        return self.fit(trajectories).transform(trajectories)
    
    def inverse_transform(self, normalized_trajectories: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        逆变换，将标准化后的数据恢复到原始范围
        
        Args:
            normalized_trajectories: 标准化后的轨迹数据
            
        Returns:
            original_trajectories: 恢复到原始范围的轨迹数据
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
            
        is_tensor = isinstance(normalized_trajectories, torch.Tensor)
        device = normalized_trajectories.device if is_tensor else None
        
        if is_tensor:
            normalized_trajectories = normalized_trajectories.cpu().numpy()
            
        original_shape = normalized_trajectories.shape
        
        # 确保数据是3维的
        if normalized_trajectories.ndim == 2:
            normalized_trajectories = normalized_trajectories[np.newaxis, ...]
        elif normalized_trajectories.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {normalized_trajectories.ndim}D")
            
        # 重塑数据
        reshaped = normalized_trajectories.reshape(-1, normalized_trajectories.shape[-1])
        
        # 逆变换: x_original = (x_normalized - min_range) / (max_range - min_range) * (max - min) + min
        min_range, max_range = self.feature_range
        original = (reshaped - min_range) / (max_range - min_range)
        original = original * self.range_vals + self.min_vals
        
        # 恢复原始形状
        original = original.reshape(normalized_trajectories.shape)
        
        # 如果原始输入是2维的，去掉添加的维度
        if len(original_shape) == 2:
            original = original[0]
            
        # 转换回tensor（如果需要）
        if is_tensor:
            original = torch.from_numpy(original).to(device)
            
        return original
    
    def get_params(self) -> dict:
        """
        获取标准化参数
        
        Returns:
            params: 包含标准化参数的字典
        """
        if not self.fitted:
            return {}
            
        return {
            'min_vals': self.min_vals.tolist() if self.min_vals is not None else None,
            'max_vals': self.max_vals.tolist() if self.max_vals is not None else None,
            'range_vals': self.range_vals.tolist() if self.range_vals is not None else None,
            'feature_range': self.feature_range,
            'fitted': self.fitted
        }
    
    def set_params(self, params: dict) -> 'TrajectoryNormalizer':
        """
        设置标准化参数
        
        Args:
            params: 包含标准化参数的字典
            
        Returns:
            self: 返回自身以支持链式调用
        """
        self.min_vals = np.array(params['min_vals']) if params['min_vals'] is not None else None
        self.max_vals = np.array(params['max_vals']) if params['max_vals'] is not None else None
        self.range_vals = np.array(params['range_vals']) if params['range_vals'] is not None else None
        self.feature_range = params['feature_range']
        self.fitted = params['fitted']
        
        return self


def normalize_trajectory_batch(
    trajectories: Union[np.ndarray, torch.Tensor],
    feature_range: Tuple[float, float] = (0.0, 1.0),
    fit_data: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Tuple[Union[np.ndarray, torch.Tensor], TrajectoryNormalizer]:
    """
    批量标准化轨迹数据的便捷函数
    
    Args:
        trajectories: 待标准化的轨迹数据
        feature_range: 标准化后的数据范围
        fit_data: 用于拟合标准化参数的数据，如果为None则使用trajectories
        
    Returns:
        normalized_trajectories: 标准化后的轨迹数据
        normalizer: 拟合好的标准化器
    """
    normalizer = TrajectoryNormalizer(feature_range=feature_range)
    
    if fit_data is not None:
        normalizer.fit(fit_data)
        normalized = normalizer.transform(trajectories)
    else:
        normalized = normalizer.fit_transform(trajectories)
    
    return normalized, normalizer


def test_normalizer():
    """测试标准化器功能"""
    print("测试轨迹标准化器...")
    
    # 创建测试数据
    np.random.seed(42)
    trajectories = np.random.randn(100, 10, 4) * 10 + 5  # [N, seq_len, feature_dim]
    
    print(f"原始数据范围: min={trajectories.min():.3f}, max={trajectories.max():.3f}")
    
    # 测试标准化
    normalizer = TrajectoryNormalizer()
    normalized = normalizer.fit_transform(trajectories)
    
    print(f"标准化后范围: min={normalized.min():.3f}, max={normalized.max():.3f}")
    
    # 测试逆变换
    recovered = normalizer.inverse_transform(normalized)
    
    print(f"逆变换后范围: min={recovered.min():.3f}, max={recovered.max():.3f}")
    print(f"逆变换误差: {np.mean(np.abs(trajectories - recovered)):.6f}")
    
    # 测试torch tensor
    trajectories_tensor = torch.from_numpy(trajectories).float()
    normalized_tensor = normalizer.transform(trajectories_tensor)
    
    print(f"Tensor标准化后范围: min={normalized_tensor.min():.3f}, max={normalized_tensor.max():.3f}")
    
    print("标准化器测试完成！")


if __name__ == "__main__":
    test_normalizer()