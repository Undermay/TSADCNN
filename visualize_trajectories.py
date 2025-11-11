#!/usr/bin/env python3
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _setup_chinese_font():
    # 尝试设置中文字体与负号显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    old_raw = data.get('old_raw_trajectories')
    new_raw = data.get('new_raw_trajectories')
    # 若不存在原始米单位片段，则使用已存的归一化/数值片段
    if old_raw is None or new_raw is None:
        old_raw = data['old_trajectories']
        new_raw = data['new_trajectories']
    dataset = {
        'old': old_raw.astype(np.float32),
        'new': new_raw.astype(np.float32),
        'labels': data['labels'].astype(np.int32),
        'scene_ids': data['scene_ids'].astype(np.int32),
        'old_flags': data['old_target_flags'].astype(np.int32),
        'new_flags': data['new_target_flags'].astype(np.int32),
        'motion_modes': data['motion_modes'],
        'meta': data.get('meta', {}),
    }
    return dataset


def group_indices_by_scene(dataset: Dict[str, np.ndarray]) -> Dict[int, List[int]]:
    scenes = defaultdict(list)
    N = len(dataset['labels'])
    for i in range(N):
        sid = int(dataset['scene_ids'][i])
        scenes[sid].append(i)
    return dict(scenes)


def build_unique_segments_for_scene(dataset: Dict[str, np.ndarray], sid: int, idxs: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """为场景构建唯一 old/new 轨迹映射，避免重复绘制。
    返回：old_map: flag->traj, new_map: flag->traj
    """
    old_map: Dict[int, np.ndarray] = {}
    new_map: Dict[int, np.ndarray] = {}
    for i in idxs:
        of = int(dataset['old_flags'][i])
        nf = int(dataset['new_flags'][i])
        if of not in old_map:
            old_map[of] = dataset['old'][i]
        if nf not in new_map:
            new_map[nf] = dataset['new'][i]
    return old_map, new_map


def compute_scene_stats(dataset: Dict[str, np.ndarray]) -> Dict[str, object]:
    scenes = group_indices_by_scene(dataset)
    # 按场景统计 K（目标数）与总轨迹数（目标总数）
    k_by_scene: Dict[int, int] = {}
    total_targets = 0
    for sid, idxs in scenes.items():
        old_map, new_map = build_unique_segments_for_scene(dataset, sid, idxs)
        K = max(len(old_map), len(new_map))
        k_by_scene[sid] = K
        total_targets += K
    k_counter = Counter(k_by_scene.values())
    stats = {
        'num_scenes': len(scenes),
        'scene_size_distribution': dict(sorted(k_counter.items())),
        'num_targets_total': total_targets,
        'num_pairs': len(dataset['labels']),
    }
    return stats


def _get_palette(n: int) -> List[Tuple[float, float, float, float]]:
    base = plt.get_cmap('tab20')
    colors = [base(i % base.N) for i in range(n)]
    return colors


def _draw_trajectory(ax, traj: np.ndarray, color, linestyle: str, start_marker: str, end_marker: str, alpha: float = 0.9, lw: float = 0.8):
    x = traj[:, 0]
    y = traj[:, 1]
    ax.plot(x, y, linestyle=linestyle, color=color, linewidth=lw, alpha=alpha)
    # 起点与终点符号
    ax.scatter(x[0], y[0], marker=start_marker, s=18, color=color, edgecolors='black', linewidths=0.3, alpha=alpha)
    ax.scatter(x[-1], y[-1], marker=end_marker, s=20, color=color, alpha=alpha)


def plot_full_dataset(dataset: Dict[str, np.ndarray], out_path: str, title: str):
    scenes = group_indices_by_scene(dataset)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    # 收集所有轨迹段，使用 LineCollection 加速与降内存占用
    old_segments_all = []
    old_colors_all = []
    old_starts = []
    old_ends = []

    new_segments_all = []
    new_colors_all = []
    new_starts = []
    new_ends = []

    for sid, idxs in scenes.items():
        old_map, new_map = build_unique_segments_for_scene(dataset, sid, idxs)
        flags_sorted = sorted(set(list(old_map.keys()) + list(new_map.keys())))
        colors = _get_palette(max(len(flags_sorted), 1))
        color_map = {f: colors[i % len(colors)] for i, f in enumerate(flags_sorted)}

        # old 轨迹段
        for f, traj in old_map.items():
            # 构造连续点段 (L-1, 2, 2)
            seg = np.stack([
                traj[:-1, :2],
                traj[1:, :2]
            ], axis=1)
            old_segments_all.append(seg)
            old_colors_all.extend([color_map[f]] * (seg.shape[0]))
            old_starts.append(traj[0, :2])
            old_ends.append(traj[-1, :2])

        # new 轨迹段
        for f, traj in new_map.items():
            seg = np.stack([
                traj[:-1, :2],
                traj[1:, :2]
            ], axis=1)
            new_segments_all.append(seg)
            new_colors_all.extend([color_map[f]] * (seg.shape[0]))
            new_starts.append(traj[0, :2])
            new_ends.append(traj[-1, :2])

    # 合并成单一数组以一次性绘制
    if len(old_segments_all) > 0:
        old_segments_all = np.concatenate(old_segments_all, axis=0)
        lc_old = LineCollection(old_segments_all, colors=old_colors_all, linewidths=0.5, alpha=0.25, linestyles='--')
        ax.add_collection(lc_old)
    if len(new_segments_all) > 0:
        new_segments_all = np.concatenate(new_segments_all, axis=0)
        lc_new = LineCollection(new_segments_all, colors=new_colors_all, linewidths=0.5, alpha=0.25, linestyles='-')
        ax.add_collection(lc_new)

    # 起点与终点一次性散点绘制
    if len(old_starts) > 0:
        old_starts = np.array(old_starts)
        old_ends = np.array(old_ends)
        ax.scatter(old_starts[:, 0], old_starts[:, 1], marker='o', s=8, c='gray', edgecolors='black', linewidths=0.2, alpha=0.4)
        ax.scatter(old_ends[:, 0], old_ends[:, 1], marker='s', s=10, c='gray', alpha=0.4)
    if len(new_starts) > 0:
        new_starts = np.array(new_starts)
        new_ends = np.array(new_ends)
        ax.scatter(new_starts[:, 0], new_starts[:, 1], marker='D', s=8, c='black', alpha=0.4)
        ax.scatter(new_ends[:, 0], new_ends[:, 1], marker='^', s=10, c='black', alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    # 简要图例：虚线=old，实线=new
    line_old, = ax.plot([], [], linestyle='--', color='gray', label='old（虚线）')
    line_new, = ax.plot([], [], linestyle='-', color='black', label='new（实线）')
    ax.legend(handles=[line_old, line_new], loc='upper right')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_random_scenes(dataset: Dict[str, np.ndarray], out_path: str, title: str, num_scenes: int = 4, seed: int = 42):
    scenes = group_indices_by_scene(dataset)
    sids = list(scenes.keys())
    if len(sids) == 0:
        raise RuntimeError('数据集中不存在场景。')
    random.seed(seed)
    random.shuffle(sids)
    pick = sids[:min(num_scenes, len(sids))]

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), dpi=150)
    axes = axes.flatten()

    for ax, sid in zip(axes, pick):
        idxs = scenes[sid]
        old_map, new_map = build_unique_segments_for_scene(dataset, sid, idxs)
        flags_sorted = sorted(set(list(old_map.keys()) + list(new_map.keys())))
        colors = _get_palette(max(len(flags_sorted), 1))
        cmap = {f: colors[i % len(colors)] for i, f in enumerate(flags_sorted)}

        for f, traj in old_map.items():
            _draw_trajectory(ax, traj, color=cmap[f], linestyle='--', start_marker='o', end_marker='s', alpha=0.9, lw=1.0)
        for f, traj in new_map.items():
            _draw_trajectory(ax, traj, color=cmap[f], linestyle='-', start_marker='D', end_marker='^', alpha=0.9, lw=1.0)

        K = max(len(old_map), len(new_map))
        ax.set_title(f'场景 {sid}（K={K}）')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

    # 若不足 4 个场景，隐藏多余子图
    for j in range(len(pick), rows * cols):
        axes[j].axis('off')

    fig.suptitle(title)
    # 增加统一图例
    handles = [
        matplotlib.lines.Line2D([], [], linestyle='--', color='gray', label='old（虚线）'),
        matplotlib.lines.Line2D([], [], linestyle='-', color='black', label='new（实线）'),
        matplotlib.lines.Line2D([], [], marker='o', linestyle='None', color='gray', label='old起点'),
        matplotlib.lines.Line2D([], [], marker='s', linestyle='None', color='gray', label='old终点'),
        matplotlib.lines.Line2D([], [], marker='D', linestyle='None', color='black', label='new起点'),
        matplotlib.lines.Line2D([], [], marker='^', linestyle='None', color='black', label='new终点'),
    ]
    fig.legend(handles=handles, loc='upper right')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def write_stats(train_stats: Dict[str, object], test_stats: Dict[str, object], out_txt: str):
    os.makedirs(os.path.dirname(out_txt) or '.', exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('=== 数据统计 ===\n')
        f.write('[训练集]\n')
        f.write(f"场景数: {train_stats['num_scenes']}\n")
        f.write(f"目标总数(≈唯一轨迹数): {train_stats['num_targets_total']}\n")
        f.write(f"轨迹对(pairs)数量: {train_stats['num_pairs']}\n")
        f.write('场景规模分布(K→场景数):\n')
        for k, cnt in train_stats['scene_size_distribution'].items():
            f.write(f"  K={k}: {cnt}\n")
        f.write('\n[测试集]\n')
        f.write(f"场景数: {test_stats['num_scenes']}\n")
        f.write(f"目标总数(≈唯一轨迹数): {test_stats['num_targets_total']}\n")
        f.write(f"轨迹对(pairs)数量: {test_stats['num_pairs']}\n")
        f.write('场景规模分布(K→场景数):\n')
        for k, cnt in test_stats['scene_size_distribution'].items():
            f.write(f"  K={k}: {cnt}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='轨迹可视化与统计')
    parser.add_argument('--train-path', type=str, default='data/train_trajectories.npy')
    parser.add_argument('--test-path', type=str, default='data/test_trajectories.npy')
    parser.add_argument('--out-dir', type=str, default='viz')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-scenes', type=int, default=4)
    args = parser.parse_args()

    _setup_chinese_font()

    train = load_dataset(args.train_path)
    test = load_dataset(args.test_path)

    # 统计
    train_stats = compute_scene_stats(train)
    test_stats = compute_scene_stats(test)
    write_stats(train_stats, test_stats, os.path.join(args.out_dir, 'stats.txt'))

    # 绘图
    os.makedirs(args.out_dir, exist_ok=True)
    plot_full_dataset(train, os.path.join(args.out_dir, 'train_full.png'), '训练集完整空域轨迹（old虚线/new实线）')
    plot_full_dataset(test, os.path.join(args.out_dir, 'test_full.png'), '测试集完整空域轨迹（old虚线/new实线）')
    plot_random_scenes(train, os.path.join(args.out_dir, 'train_scenes_4.png'), '训练集随机4个场景', num_scenes=args.num_scenes, seed=args.seed)
    plot_random_scenes(test, os.path.join(args.out_dir, 'test_scenes_4.png'), '测试集随机4个场景', num_scenes=args.num_scenes, seed=args.seed)

    print('图片已生成：')
    print(f" - {os.path.join(args.out_dir, 'train_full.png')}")
    print(f" - {os.path.join(args.out_dir, 'test_full.png')}")
    print(f" - {os.path.join(args.out_dir, 'train_scenes_4.png')}")
    print(f" - {os.path.join(args.out_dir, 'test_scenes_4.png')}")
    print(f"统计文件：{os.path.join(args.out_dir, 'stats.txt')}")


if __name__ == '__main__':
    main()