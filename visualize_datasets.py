import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Fix Chinese font rendering for labels
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODE_ORDER = ['CV', 'CA', 'TURN_SMALL', 'TURN_MEDIUM', 'TURN_LARGE']


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True).item()
    segments = np.array(data['segments'], dtype=np.float32)
    labels = np.array(data['labels'], dtype=np.int64)
    meta = data.get('meta', {})
    return segments, labels, meta


def split_by_mode(segments: np.ndarray, labels: np.ndarray,
                  segments_per_mode: int, segments_per_target: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    targets_count = segments_per_mode // segments_per_target
    per_mode_count = targets_count * segments_per_target
    result = {}
    start = 0
    for mode in MODE_ORDER:
        end = start + per_mode_count
        result[mode] = (segments[start:end], labels[start:end])
        start = end
    return result


def plot_mode(ax, mode_name: str, segments: np.ndarray, labels: np.ndarray):
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_labels))))
    color_map = {lid: colors[i % len(colors)] for i, lid in enumerate(unique_labels)}

    for i in range(len(segments)):
        seg = segments[i]
        lid = labels[i]
        c = color_map[lid]
        ax.plot(seg[:, 0], seg[:, 1], '-', color=c, alpha=0.6, linewidth=1)

    ax.set_title(mode_name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.2)
    ax.axis('equal')


def visualize(train_path: str, test_path: str,
              segments_per_target: int, save_path: str):
    train_segments, train_labels, train_meta = load_dataset(train_path)
    test_segments, test_labels, test_meta = load_dataset(test_path)

    train_spm = int(train_meta.get('segments_per_mode', len(train_segments) // 5))
    test_spm = int(test_meta.get('segments_per_mode', len(test_segments) // 5))

    train_split = split_by_mode(train_segments, train_labels, train_spm, segments_per_target)
    test_split = split_by_mode(test_segments, test_labels, test_spm, segments_per_target)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    for idx, mode in enumerate(MODE_ORDER):
        ax = axes[0, idx]
        segs, labs = train_split[mode]
        plot_mode(ax, f"{mode}", segs, labs)

    for idx, mode in enumerate(MODE_ORDER):
        ax = axes[1, idx]
        segs, labs = test_split[mode]
        plot_mode(ax, f"{mode}", segs, labs)

    axes[0, 0].text(0.02, 1.06, '(a) 训练数据集', transform=axes[0, 0].transAxes, fontsize=12)
    axes[1, 0].text(0.02, 1.06, '(b) 验证/测试数据集', transform=axes[1, 0].transAxes, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize datasets by motion models in two rows (train/val-or-test).')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing train/test npy')
    parser.add_argument('--segments_per_target', type=int, default=5)
    parser.add_argument('--data_variant', type=str, default='clean', choices=['clean', 'noisy'])
    parser.add_argument('--save_path', type=str, default='outputs/motion_models.png')
    args = parser.parse_args()

    if args.data_variant == 'clean':
        train_path = os.path.join(args.data_dir, 'train_data.npy')
        # 优先使用 val_data.npy（若存在），否则回退到 test_data.npy
        val_candidate = os.path.join(args.data_dir, 'val_data.npy')
        test_candidate = os.path.join(args.data_dir, 'test_data.npy')
        test_path = val_candidate if os.path.exists(val_candidate) else test_candidate
        default_save = 'outputs/motion_models.png'
    else:
        train_path = os.path.join(args.data_dir, 'train_data_noisy.npy')
        test_path = os.path.join(args.data_dir, 'test_data_noisy.npy')
        default_save = 'outputs/motion_models_noisy.png'

    save_path = args.save_path or default_save
    visualize(train_path, test_path, args.segments_per_target, save_path)


if __name__ == '__main__':
    main()