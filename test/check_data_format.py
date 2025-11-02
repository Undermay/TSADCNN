#!/usr/bin/env python3
"""
检查数据格式和维度
"""

import numpy as np
import torch

def check_data_format():
    """检查数据的实际格式"""
    print("🔍 检查数据格式...")
    
    # 加载训练数据
    try:
        data = np.load("data/train_correct.npy", allow_pickle=True).item()
        print(f"✅ 数据加载成功")
        
        # 检查数据结构
        print(f"📊 数据结构:")
        for key in data.keys():
            print(f"  - {key}: {type(data[key])}")
            if isinstance(data[key], (list, np.ndarray)):
                print(f"    长度: {len(data[key])}")
                if len(data[key]) > 0:
                    sample = data[key][0]
                    if isinstance(sample, np.ndarray):
                        print(f"    样本形状: {sample.shape}")
                    else:
                        print(f"    样本类型: {type(sample)}")
        
        # 检查轨迹维度
        old_trajectories = data['old_trajectories']
        new_trajectories = data['new_trajectories']
        
        print(f"\n📏 轨迹维度分析:")
        print(f"  - old轨迹数量: {len(old_trajectories)}")
        print(f"  - new轨迹数量: {len(new_trajectories)}")
        
        if len(old_trajectories) > 0:
            sample_old = np.array(old_trajectories[0])
            sample_new = np.array(new_trajectories[0])
            
            print(f"  - old轨迹形状: {sample_old.shape}")
            print(f"  - new轨迹形状: {sample_new.shape}")
            
            print(f"\n📋 样本数据预览:")
            print(f"  - old轨迹前3个时间步:")
            print(sample_old[:3])
            print(f"  - new轨迹前3个时间步:")
            print(sample_new[:3])
            
            # 检查特征维度
            feature_dim = sample_old.shape[1] if len(sample_old.shape) > 1 else 1
            print(f"\n🎯 特征维度: {feature_dim}")
            
            if feature_dim == 4:
                print("  特征可能是: [x, y, vx, vy]")
            elif feature_dim == 6:
                print("  特征可能是: [x, y, vx, vy, ax, ay]")
            else:
                print(f"  未知的特征维度: {feature_dim}")
        
        # 检查标签分布
        labels = data['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n🏷️ 标签分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  - 标签 {label}: {count} 个样本")
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")

if __name__ == "__main__":
    check_data_format()