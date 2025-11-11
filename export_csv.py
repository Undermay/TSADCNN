#!/usr/bin/env python3
import os
import csv
import json
from typing import Dict
import numpy as np


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    # 原始米单位片段（如存在优先使用）
    old_raw = data.get('old_raw_trajectories')
    new_raw = data.get('new_raw_trajectories')
    if old_raw is None or new_raw is None:
        old_raw = data['old_trajectories']
        new_raw = data['new_trajectories']
    return {
        'old': old_raw.astype(np.float32),
        'new': new_raw.astype(np.float32),
        'labels': data['labels'].astype(np.int32),
        'scene_ids': data['scene_ids'].astype(np.int32),
        'old_flags': data.get('old_target_flags', np.zeros(len(data['labels']), dtype=np.int32)).astype(np.int32),
        'new_flags': data.get('new_target_flags', np.zeros(len(data['labels']), dtype=np.int32)).astype(np.int32),
        'motion_modes': np.array(data.get('motion_modes', ['unknown'] * len(data['labels'])), dtype=object),
        # 若存在原始片段字段，则也返回，供严格镜像写出
        'old_raw_optional': data.get('old_raw_trajectories'),
        'new_raw_optional': data.get('new_raw_trajectories'),
        'meta': data.get('meta', {}),
    }


def export_pairs_csv(dataset: Dict[str, np.ndarray], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['pair_index', 'scene_id', 'label', 'old_flag', 'new_flag', 'motion_mode'])
        N = len(dataset['labels'])
        for i in range(N):
            w.writerow([
                i,
                int(dataset['scene_ids'][i]),
                int(dataset['labels'][i]),
                int(dataset['old_flags'][i]),
                int(dataset['new_flags'][i]),
                str(dataset['motion_modes'][i]),
            ])


def export_trajectories_csv(dataset: Dict[str, np.ndarray], out_csv: str, split_name: str):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        # 长表：每行一个时间步
        w.writerow([
            'split', 'pair_index', 'scene_id', 'role', 'flag', 'motion_mode', 't',
            'x', 'y', 'vx', 'vy', 'ax', 'ay'
        ])
        N = len(dataset['labels'])
        for i in range(N):
            scene_id = int(dataset['scene_ids'][i])
            motion = str(dataset['motion_modes'][i])
            # old 片段
            old_flag = int(dataset['old_flags'][i])
            old = dataset['old'][i]  # [L, 6]
            L = old.shape[0]
            for t in range(L):
                x, y, vx, vy, ax, ay = old[t]
                w.writerow([split_name, i, scene_id, 'old', old_flag, motion, t, x, y, vx, vy, ax, ay])

            # new 片段
            new_flag = int(dataset['new_flags'][i])
            new = dataset['new'][i]
            L2 = new.shape[0]
            for t in range(L2):
                x, y, vx, vy, ax, ay = new[t]
                w.writerow([split_name, i, scene_id, 'new', new_flag, motion, t, x, y, vx, vy, ax, ay])


def export_dataset_strict(dataset: Dict[str, np.ndarray], out_csv: str, split_name: str):
    """
    严格镜像导出：每个Numpy样本对为一行，数组字段以JSON字符串写入，保持原始形状与数值。
    字段包含：pair_index, scene_id, label, old_flag, new_flag, motion_mode,
             old_trajectory_json, new_trajectory_json,
             old_raw_json(若存在), new_raw_json(若存在)
    """
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow([
            'split', 'pair_index', 'scene_id', 'label', 'old_flag', 'new_flag', 'motion_mode',
            'old_trajectory_json', 'new_trajectory_json', 'old_raw_json', 'new_raw_json'
        ])
        N = len(dataset['labels'])
        has_raw = (dataset.get('old_raw_optional') is not None) and (dataset.get('new_raw_optional') is not None)
        for i in range(N):
            # 数组转JSON字符串，保留形状 [13,6]
            old_json = json.dumps(dataset['old'][i].tolist(), ensure_ascii=False)
            new_json = json.dumps(dataset['new'][i].tolist(), ensure_ascii=False)
            if has_raw:
                old_raw = json.dumps(dataset['old_raw_optional'][i].tolist(), ensure_ascii=False)
                new_raw = json.dumps(dataset['new_raw_optional'][i].tolist(), ensure_ascii=False)
            else:
                old_raw = ''
                new_raw = ''

            w.writerow([
                split_name,
                i,
                int(dataset['scene_ids'][i]),
                int(dataset['labels'][i]),
                int(dataset['old_flags'][i]),
                int(dataset['new_flags'][i]),
                str(dataset['motion_modes'][i]),
                old_json,
                new_json,
                old_raw,
                new_raw,
            ])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='导出NPY数据集为CSV')
    parser.add_argument('--train-path', type=str, default='data/train_trajectories.npy')
    parser.add_argument('--test-path', type=str, default='data/test_trajectories.npy')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--mode', type=str, default='strict', choices=['strict', 'long'],
                        help='strict: 每个npy导出一个CSV（数组字段JSON镜像）；long: 生成pairs与长表两类CSV')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 训练集
    train = load_dataset(args.train_path)
    # 测试集
    test = load_dataset(args.test_path)

    if args.mode == 'strict':
        export_dataset_strict(train, os.path.join(args.out_dir, 'train_dataset.csv'), split_name='train')
        export_dataset_strict(test, os.path.join(args.out_dir, 'test_dataset.csv'), split_name='test')
        print('CSV 导出完成（严格镜像模式）:')
        print(f" - {os.path.join(args.out_dir, 'train_dataset.csv')}")
        print(f" - {os.path.join(args.out_dir, 'test_dataset.csv')}")
    else:
        export_pairs_csv(train, os.path.join(args.out_dir, 'train_pairs.csv'))
        export_trajectories_csv(train, os.path.join(args.out_dir, 'train_trajectories.csv'), split_name='train')
        export_pairs_csv(test, os.path.join(args.out_dir, 'test_pairs.csv'))
        export_trajectories_csv(test, os.path.join(args.out_dir, 'test_trajectories.csv'), split_name='test')
        print('CSV 导出完成（长表模式）:')
        print(f" - {os.path.join(args.out_dir, 'train_pairs.csv')}")
        print(f" - {os.path.join(args.out_dir, 'train_trajectories.csv')}")
        print(f" - {os.path.join(args.out_dir, 'test_pairs.csv')}")
        print(f" - {os.path.join(args.out_dir, 'test_trajectories.csv')}")


if __name__ == '__main__':
    main()