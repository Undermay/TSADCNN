#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计分析脚本
分析训练集和测试集的场景分布、轨迹数量等统计信息
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os

def analyze_dataset_statistics():
    """分析数据集统计信息"""
    
    # 数据文件路径
    train_path = "data/csv/train_correct.csv"
    test_path = "data/csv/test_correct.csv"
    
    print("=== 数据集统计分析 ===\n")
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        print(f"❌ 训练集文件不存在: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"❌ 测试集文件不存在: {test_path}")
        return
    
    # 加载数据集
    print("📊 加载数据集...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ 训练集加载完成: {len(train_df)} 条记录")
    print(f"✅ 测试集加载完成: {len(test_df)} 条记录\n")
    
    # 分析训练集
    print("🔍 训练集统计信息:")
    analyze_single_dataset(train_df, "训练集")
    
    print("\n" + "="*50 + "\n")
    
    # 分析测试集
    print("🔍 测试集统计信息:")
    analyze_single_dataset(test_df, "测试集")
    
    return train_df, test_df

def analyze_single_dataset(df, dataset_name):
    """分析单个数据集的统计信息"""
    
    # 基本统计
    total_pairs = len(df)
    positive_pairs = len(df[df['label'] == 1])
    negative_pairs = len(df[df['label'] == 0])
    
    print(f"📈 {dataset_name}基本统计:")
    print(f"   总轨迹对数: {total_pairs}")
    print(f"   正样本对数: {positive_pairs} ({positive_pairs/total_pairs*100:.1f}%)")
    print(f"   负样本对数: {negative_pairs} ({negative_pairs/total_pairs*100:.1f}%)")
    
    # 场景统计
    if 'scene_id' in df.columns:
        unique_scenes = df['scene_id'].nunique()
        scene_counts = df['scene_id'].value_counts().sort_index()
        
        print(f"\n🎬 场景统计:")
        print(f"   总场景数: {unique_scenes}")
        print(f"   每场景轨迹对数范围: {scene_counts.min()} - {scene_counts.max()}")
        print(f"   平均每场景轨迹对数: {scene_counts.mean():.1f}")
        
        # 显示前10个场景的轨迹对数
        print(f"\n   前10个场景的轨迹对分布:")
        for scene_id, count in scene_counts.head(10).items():
            print(f"     场景 {scene_id}: {count} 对")
    
    # 运动模式统计
    if 'motion_mode' in df.columns:
        motion_counts = df['motion_mode'].value_counts()
        
        print(f"\n🚀 运动模式统计:")
        for mode, count in motion_counts.items():
            print(f"   {mode}: {count} 对 ({count/total_pairs*100:.1f}%)")
    
    # 目标标志统计
    if 'old_target_flag' in df.columns and 'new_target_flag' in df.columns:
        old_targets = df['old_target_flag'].nunique()
        new_targets = df['new_target_flag'].nunique()
        
        print(f"\n🎯 目标统计:")
        print(f"   Old轨迹唯一目标数: {old_targets}")
        print(f"   New轨迹唯一目标数: {new_targets}")
        
        # 统计每个场景中的目标数量
        if 'scene_id' in df.columns:
            scene_target_stats = []
            for scene_id in df['scene_id'].unique():
                scene_data = df[df['scene_id'] == scene_id]
                old_targets_in_scene = scene_data['old_target_flag'].nunique()
                new_targets_in_scene = scene_data['new_target_flag'].nunique()
                scene_target_stats.append({
                    'scene_id': scene_id,
                    'old_targets': old_targets_in_scene,
                    'new_targets': new_targets_in_scene,
                    'total_pairs': len(scene_data)
                })
            
            scene_target_df = pd.DataFrame(scene_target_stats)
            print(f"\n   场景目标统计:")
            print(f"     平均每场景Old目标数: {scene_target_df['old_targets'].mean():.1f}")
            print(f"     平均每场景New目标数: {scene_target_df['new_targets'].mean():.1f}")
            print(f"     目标数范围 (Old): {scene_target_df['old_targets'].min()} - {scene_target_df['old_targets'].max()}")
            print(f"     目标数范围 (New): {scene_target_df['new_targets'].min()} - {scene_target_df['new_targets'].max()}")

def get_trajectory_data(df, row_idx):
    """从数据框中提取轨迹数据"""
    row = df.iloc[row_idx]
    
    # 提取old轨迹
    old_traj = []
    new_traj = []
    
    for t in range(13):  # 13个时间步
        old_point = [
            row[f'old_raw_x_{t}'],
            row[f'old_raw_y_{t}']
        ]
        new_point = [
            row[f'new_raw_x_{t}'],
            row[f'new_raw_y_{t}']
        ]
        old_traj.append(old_point)
        new_traj.append(new_point)
    
    return np.array(old_traj), np.array(new_traj)

def get_scene_trajectories(df, scene_id, max_pairs=None):
    """获取指定场景的所有轨迹对"""
    scene_data = df[df['scene_id'] == scene_id]
    
    if max_pairs:
        scene_data = scene_data.head(max_pairs)
    
    trajectories = []
    for idx in range(len(scene_data)):
        row = scene_data.iloc[idx]
        old_traj, new_traj = get_trajectory_data(pd.DataFrame([row]), 0)
        
        trajectories.append({
            'old_trajectory': old_traj,
            'new_trajectory': new_traj,
            'label': row['label'],
            'old_target_flag': row['old_target_flag'],
            'new_target_flag': row['new_target_flag'],
            'motion_mode': row.get('motion_mode', 'unknown')
        })
    
    return trajectories

if __name__ == "__main__":
    # 运行统计分析
    train_df, test_df = analyze_dataset_statistics()
    
    print("\n" + "="*60)
    print("📋 数据集统计分析完成!")
    print("="*60)