#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹可视化脚本
生成四张图片：
1. 完整训练集空域轨迹
2. 完整测试集空域轨迹  
3. 训练集随机4个场景轨迹
4. 训练集另外随机4个场景轨迹
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import seaborn as sns
import random
import os
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置颜色方案
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',
    '#A3E4D7', '#F9E79F', '#D5A6BD', '#AED6F1', '#A9DFBF'
]

def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试设置中文字体
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        print("✅ 中文字体设置成功")
    except Exception as e:
        print(f"⚠️ 中文字体设置失败: {e}")

def load_dataset(file_path):
    """加载数据集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ 加载数据集: {file_path}, 共 {len(df)} 条记录")
    return df

def extract_trajectory(row):
    """从数据行中提取轨迹坐标"""
    old_traj = []
    new_traj = []
    
    for t in range(13):  # 13个时间步
        old_x = row[f'old_raw_x_{t}']
        old_y = row[f'old_raw_y_{t}']
        new_x = row[f'new_raw_x_{t}']
        new_y = row[f'new_raw_y_{t}']
        
        old_traj.append([old_x, old_y])
        new_traj.append([new_x, new_y])
    
    return np.array(old_traj), np.array(new_traj)

def plot_trajectory(ax, trajectory, color, linestyle='-', alpha=0.7, linewidth=1.5, label=None):
    """绘制单条轨迹"""
    if len(trajectory) < 2:
        return
    
    # 绘制轨迹线
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
           color=color, linestyle=linestyle, alpha=alpha, 
           linewidth=linewidth, label=label)
    
    # 标记起点（圆圈）
    ax.scatter(trajectory[0, 0], trajectory[0, 1], 
              color=color, marker='o', s=50, alpha=0.8, 
              edgecolors='white', linewidth=1, zorder=5)
    
    # 标记终点（三角形）
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
              color=color, marker='^', s=60, alpha=0.8, 
              edgecolors='white', linewidth=1, zorder=5)

def plot_full_dataset_trajectories(df, title, save_path, max_trajectories=1000):
    """绘制完整数据集的轨迹"""
    print(f"🎨 开始绘制: {title}")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # 随机采样轨迹以避免过度拥挤
    if len(df) > max_trajectories:
        sample_df = df.sample(n=max_trajectories, random_state=42)
        print(f"   随机采样 {max_trajectories} 条轨迹进行可视化")
    else:
        sample_df = df
    
    # 按目标分组绘制
    target_colors = {}
    color_idx = 0
    
    for idx, row in sample_df.iterrows():
        old_traj, new_traj = extract_trajectory(row)
        
        # 为每个目标分配颜色
        old_target = row['old_target_flag']
        new_target = row['new_target_flag']
        
        if old_target not in target_colors:
            target_colors[old_target] = COLORS[color_idx % len(COLORS)]
            color_idx += 1
        if new_target not in target_colors:
            target_colors[new_target] = COLORS[color_idx % len(COLORS)]
            color_idx += 1
        
        # 绘制old轨迹（实线）
        plot_trajectory(ax, old_traj, target_colors[old_target], 
                       linestyle='-', alpha=0.6, linewidth=1.2)
        
        # 绘制new轨迹（虚线）
        plot_trajectory(ax, new_traj, target_colors[new_target], 
                       linestyle='--', alpha=0.6, linewidth=1.2)
    
    # 设置图形属性
    ax.set_xlabel('X坐标 (米)', fontsize=14)
    ax.set_ylabel('Y坐标 (米)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='-', label='Old轨迹 (实线)'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='New轨迹 (虚线)'),
        plt.scatter([], [], color='gray', marker='o', s=50, label='起点'),
        plt.scatter([], [], color='gray', marker='^', s=60, label='终点')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # 添加统计信息
    stats_text = f"总轨迹对: {len(df)}\n"
    stats_text += f"显示轨迹对: {len(sample_df)}\n"
    stats_text += f"场景数: {df['scene_id'].nunique()}\n"
    stats_text += f"目标数: {df['old_target_flag'].nunique()}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存图片: {save_path}")

def plot_scene_trajectories(df, scene_ids, title, save_path):
    """绘制指定场景的轨迹"""
    print(f"🎨 开始绘制: {title}")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, scene_id in enumerate(scene_ids):
        ax = axes[i]
        scene_data = df[df['scene_id'] == scene_id]
        
        if len(scene_data) == 0:
            ax.text(0.5, 0.5, f'场景 {scene_id}\n无数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue
        
        # 为每个目标分配颜色
        target_colors = {}
        color_idx = 0
        
        # 统计场景信息
        positive_pairs = len(scene_data[scene_data['label'] == 1])
        negative_pairs = len(scene_data[scene_data['label'] == 0])
        motion_modes = scene_data['motion_mode'].value_counts()
        
        for idx, row in scene_data.iterrows():
            old_traj, new_traj = extract_trajectory(row)
            
            # 为每个目标分配颜色
            old_target = row['old_target_flag']
            new_target = row['new_target_flag']
            
            if old_target not in target_colors:
                target_colors[old_target] = COLORS[color_idx % len(COLORS)]
                color_idx += 1
            if new_target not in target_colors:
                target_colors[new_target] = COLORS[color_idx % len(COLORS)]
                color_idx += 1
            
            # 绘制轨迹
            plot_trajectory(ax, old_traj, target_colors[old_target], 
                           linestyle='-', alpha=0.8, linewidth=2)
            plot_trajectory(ax, new_traj, target_colors[new_target], 
                           linestyle='--', alpha=0.8, linewidth=2)
        
        # 设置子图属性
        ax.set_xlabel('X坐标 (米)', fontsize=12)
        ax.set_ylabel('Y坐标 (米)', fontsize=12)
        ax.set_title(f'场景 {scene_id}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # 添加场景统计信息
        stats_text = f"轨迹对: {len(scene_data)}\n"
        stats_text += f"正样本: {positive_pairs}\n"
        stats_text += f"负样本: {negative_pairs}\n"
        stats_text += f"目标数: {len(target_colors)}\n"
        if len(motion_modes) > 0:
            top_mode = motion_modes.index[0]
            stats_text += f"主要模式: {top_mode}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 添加总图例
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='-', label='Old轨迹 (实线)'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='New轨迹 (虚线)'),
        plt.scatter([], [], color='gray', marker='o', s=50, label='起点'),
        plt.scatter([], [], color='gray', marker='^', s=60, label='终点')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存图片: {save_path}")

def main():
    """主函数"""
    print("🚀 开始轨迹可视化...")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建输出目录
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据集
        train_df = load_dataset("data/csv/train_correct.csv")
        test_df = load_dataset("data/csv/test_correct.csv")
        
        print("\n📊 数据集统计信息:")
        print(f"训练集: {len(train_df)} 轨迹对, {train_df['scene_id'].nunique()} 场景")
        print(f"测试集: {len(test_df)} 轨迹对, {test_df['scene_id'].nunique()} 场景")
        
        # 1. 绘制完整训练集轨迹
        try:
            print("\n🎨 步骤1: 绘制训练集轨迹...")
            plot_full_dataset_trajectories(
                train_df, 
                "训练集空域轨迹分布\n(随机采样1000条轨迹对)", 
                f"{output_dir}/01_train_full_trajectories.png",
                max_trajectories=1000
            )
        except Exception as e:
            print(f"❌ 训练集轨迹绘制失败: {e}")
        
        # 2. 绘制完整测试集轨迹
        try:
            print("\n🎨 步骤2: 绘制测试集轨迹...")
            plot_full_dataset_trajectories(
                test_df, 
                "测试集空域轨迹分布\n(全部轨迹对)", 
                f"{output_dir}/02_test_full_trajectories.png",
                max_trajectories=2000
            )
        except Exception as e:
            print(f"❌ 测试集轨迹绘制失败: {e}")
        
        # 3. 随机选择训练集场景进行详细展示
        try:
            print("\n🎨 步骤3: 准备场景数据...")
            train_scenes = train_df['scene_id'].unique()
            random.seed(42)  # 固定随机种子以便复现
            
            # 选择有足够轨迹对的场景
            scene_counts = train_df['scene_id'].value_counts()
            good_scenes = scene_counts[scene_counts >= 10].index.tolist()
            print(f"   找到 {len(good_scenes)} 个有足够数据的场景")
            
            if len(good_scenes) >= 8:
                selected_scenes_1 = random.sample(good_scenes, 4)
                remaining_scenes = [s for s in good_scenes if s not in selected_scenes_1]
                selected_scenes_2 = random.sample(remaining_scenes, 4)
            else:
                # 如果场景不够，就用所有可用场景
                selected_scenes_1 = good_scenes[:4]
                selected_scenes_2 = good_scenes[4:8] if len(good_scenes) >= 8 else good_scenes[:4]
            
            print(f"   选择场景组1: {selected_scenes_1}")
            print(f"   选择场景组2: {selected_scenes_2}")
        except Exception as e:
            print(f"❌ 场景选择失败: {e}")
            selected_scenes_1 = [0, 1, 2, 3]
            selected_scenes_2 = [4, 5, 6, 7]
        
        # 4. 绘制第一组场景
        try:
            print("\n🎨 步骤4: 绘制第一组场景...")
            plot_scene_trajectories(
                train_df, 
                selected_scenes_1,
                f"训练集随机场景轨迹详情 (第一组)\n场景ID: {selected_scenes_1}",
                f"{output_dir}/03_train_sample_scenes_group1.png"
            )
        except Exception as e:
            print(f"❌ 第一组场景绘制失败: {e}")
        
        # 5. 绘制第二组场景
        try:
            print("\n🎨 步骤5: 绘制第二组场景...")
            plot_scene_trajectories(
                train_df, 
                selected_scenes_2,
                f"训练集随机场景轨迹详情 (第二组)\n场景ID: {selected_scenes_2}",
                f"{output_dir}/04_train_sample_scenes_group2.png"
            )
        except Exception as e:
            print(f"❌ 第二组场景绘制失败: {e}")
        
        print(f"\n🎉 可视化完成！所有图片保存在 {output_dir}/ 目录下")
        print("\n📋 生成的图片:")
        print("   1. 01_train_full_trajectories.png - 训练集完整轨迹")
        print("   2. 02_test_full_trajectories.png - 测试集完整轨迹")
        print("   3. 03_train_sample_scenes_group1.png - 训练集样本场景(第一组)")
        print("   4. 04_train_sample_scenes_group2.png - 训练集样本场景(第二组)")
        
    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()