import os
import numpy as np
import argparse
from utils.data_loader import generate_motion_dataset


def save_dataset_dict(output_dir, filename, segments, labels, scene_ids, local_target_ids, meta=None):
    os.makedirs(output_dir, exist_ok=True)
    data = {
        'segments': segments,
        'labels': labels,
        'scene_ids': scene_ids,
        'local_target_ids': local_target_ids,
        'meta': meta or {}
    }
    np.save(os.path.join(output_dir, filename), data)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic motion datasets")
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=64, help='Sequence length')
    parser.add_argument('--input_dim', type=int, default=4, choices=[4,6], help='Input feature dimension')
    parser.add_argument('--segments_per_mode', type=int, default=1000, help='Segments per motion mode')
    parser.add_argument('--segments_per_target', type=int, default=5, help='Segments per target')
    parser.add_argument('--noise_std_pos', type=float, default=0.3, help='Position noise std')
    parser.add_argument('--noise_std_vel', type=float, default=0.1, help='Velocity noise std')
    parser.add_argument('--k_per_scene', type=int, default=8, help='Targets per scene (K)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    np.random.seed(args.seed)

    # 生成按模式顺序排列的数据（CV, CA, TURN_SMALL, TURN_MEDIUM, TURN_LARGE）
    segments, labels, scene_ids, local_target_ids = generate_motion_dataset(
        sequence_length=args.sequence_length,
        input_dim=args.input_dim,
        segments_per_mode=args.segments_per_mode,
        segments_per_target=args.segments_per_target,
        noise_std_pos=args.noise_std_pos,
        noise_std_vel=args.noise_std_vel,
        targets_per_scene=args.k_per_scene
    )

    # 关键修复：按“每个模式”独立切分训练/验证，确保每个子集内部仍保持按模式连续排列
    num_modes = 5
    spm_full = args.segments_per_mode
    unit = args.segments_per_target * args.k_per_scene  # 一个场景的段数（K个目标 * 每目标段数）
    max_scenes_per_mode = spm_full // unit
    train_scenes_per_mode = int(max_scenes_per_mode * (1 - args.val_ratio))
    # 至少保留1个场景到训练或验证，防止极端参数导致空模式
    train_scenes_per_mode = max(1, min(train_scenes_per_mode, max_scenes_per_mode - 1)) if max_scenes_per_mode >= 2 else max_scenes_per_mode
    train_spm = train_scenes_per_mode * unit
    val_spm = spm_full - train_spm

    train_segments, train_labels, train_scene_ids, train_local_ids = [], [], [], []
    val_segments, val_labels, val_scene_ids, val_local_ids = [], [], [], []

    for m in range(num_modes):
        start = m * spm_full
        mid = start + train_spm
        end = start + spm_full
        # 追加该模式的训练切片
        train_segments.extend(segments[start:mid])
        train_labels.extend(labels[start:mid])
        train_scene_ids.extend(scene_ids[start:mid])
        train_local_ids.extend(local_target_ids[start:mid])
        # 追加该模式的验证切片
        val_segments.extend(segments[mid:end])
        val_labels.extend(labels[mid:end])
        val_scene_ids.extend(scene_ids[mid:end])
        val_local_ids.extend(local_target_ids[mid:end])

    # 为不同子集写入对应的每模式段数，便于可视化按照正确窗口切分
    meta_train = {
        'segments_per_mode': train_spm,
        'segments_per_target': args.segments_per_target,
        'k_per_scene': args.k_per_scene,
        'input_dim': args.input_dim,
        'sequence_length': args.sequence_length
    }
    meta_val = {
        'segments_per_mode': val_spm,
        'segments_per_target': args.segments_per_target,
        'k_per_scene': args.k_per_scene,
        'input_dim': args.input_dim,
        'sequence_length': args.sequence_length
    }

    save_dataset_dict(args.output_dir, 'train_data.npy', train_segments, train_labels, train_scene_ids, train_local_ids, meta_train)
    save_dataset_dict(args.output_dir, 'val_data.npy', val_segments, val_labels, val_scene_ids, val_local_ids, meta_val)

    print(f"Saved datasets to {args.output_dir}. Train per-mode: {train_spm}, Val per-mode: {val_spm}. Total Train: {len(train_segments)}, Val: {len(val_segments)}")


if __name__ == '__main__':
    main()