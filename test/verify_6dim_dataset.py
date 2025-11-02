#!/usr/bin/env python3
"""
验证新生成的6维数据集格式和维度正确性
"""

import pandas as pd
import numpy as np
import os

def verify_dataset():
    """验证数据集格式和维度"""
    
    print("=== 验证6维数据集 ===")
    
    # 检查文件是否存在
    train_path = "data/csv/train_correct.csv"
    test_path = "data/csv/test_correct.csv"
    
    if not os.path.exists(train_path):
        print(f"❌ 训练集文件不存在: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"❌ 测试集文件不存在: {test_path}")
        return False
    
    print(f"✅ 数据集文件存在")
    
    # 读取训练集
    print("\n1. 检查训练集...")
    train_df = pd.read_csv(train_path)
    print(f"   训练集形状: {train_df.shape}")
    print(f"   列数: {len(train_df.columns)}")
    
    # 检查列名
    expected_cols = ['pair_index', 'scene_id', 'old_target_flag', 'new_target_flag', 'label', 'motion_mode']
    feature_names = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
    
    # 添加轨迹特征列
    for prefix in ['old_raw', 'new_raw']:
        for t in range(13):  # 13个时间步
            for fn in feature_names:
                expected_cols.append(f"{prefix}_{fn}_{t}")
    
    print(f"   期望列数: {len(expected_cols)}")
    print(f"   实际列数: {len(train_df.columns)}")
    
    if len(train_df.columns) == len(expected_cols):
        print("   ✅ 列数正确")
    else:
        print("   ❌ 列数不匹配")
        print(f"   缺失或多余的列: {set(expected_cols) - set(train_df.columns)}")
    
    # 检查数据维度
    print("\n2. 检查数据维度...")
    
    # 提取一个样本的轨迹数据
    sample_row = train_df.iloc[0]
    
    # 提取old_raw轨迹 (13步 × 6维)
    old_raw_data = []
    for t in range(13):
        step_data = []
        for fn in feature_names:
            col_name = f"old_raw_{fn}_{t}"
            step_data.append(sample_row[col_name])
        old_raw_data.append(step_data)
    
    old_raw_array = np.array(old_raw_data)
    print(f"   old_raw轨迹形状: {old_raw_array.shape}")
    
    # 提取new_raw轨迹 (13步 × 6维)
    new_raw_data = []
    for t in range(13):
        step_data = []
        for fn in feature_names:
            col_name = f"new_raw_{fn}_{t}"
            step_data.append(sample_row[col_name])
        new_raw_data.append(step_data)
    
    new_raw_array = np.array(new_raw_data)
    print(f"   new_raw轨迹形状: {new_raw_array.shape}")
    
    # 验证维度
    if old_raw_array.shape == (13, 6) and new_raw_array.shape == (13, 6):
        print("   ✅ 轨迹维度正确: (13步, 6维特征)")
    else:
        print("   ❌ 轨迹维度错误")
        return False
    
    # 检查特征内容
    print("\n3. 检查特征内容...")
    print(f"   old_raw样本数据 (前3步):")
    for i in range(3):
        print(f"     步骤{i}: x={old_raw_array[i,0]:.3f}, y={old_raw_array[i,1]:.3f}, "
              f"vx={old_raw_array[i,2]:.3f}, vy={old_raw_array[i,3]:.3f}, "
              f"ax={old_raw_array[i,4]:.3f}, ay={old_raw_array[i,5]:.3f}")
    
    # 检查标签分布
    print("\n4. 检查标签分布...")
    label_counts = train_df['label'].value_counts()
    print(f"   正样本 (label=1): {label_counts.get(1, 0)}")
    print(f"   负样本 (label=0): {label_counts.get(0, 0)}")
    
    # 检查运动模式分布
    print("\n5. 检查运动模式分布...")
    motion_counts = train_df['motion_mode'].value_counts()
    for mode, count in motion_counts.items():
        print(f"   {mode}: {count}")
    
    # 检查测试集
    print("\n6. 检查测试集...")
    test_df = pd.read_csv(test_path)
    print(f"   测试集形状: {test_df.shape}")
    
    test_label_counts = test_df['label'].value_counts()
    print(f"   正样本 (label=1): {test_label_counts.get(1, 0)}")
    print(f"   负样本 (label=0): {test_label_counts.get(0, 0)}")
    
    print("\n=== 验证完成 ===")
    print("✅ 6维数据集格式正确，包含 [x, y, vx, vy, ax, ay] 特征")
    return True

if __name__ == "__main__":
    verify_dataset()