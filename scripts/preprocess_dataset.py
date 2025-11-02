#!/usr/bin/env python3
"""
离线归一化预处理脚本：将轨迹数据按指定模式（segment/scene/global）进行0-1 MinMax归一化，
并保存为标准格式的 .npy 数据集，避免训练时重复归一化计算。

用法示例：
python scripts/preprocess_dataset.py --input data/train_correct.npy --output data/train_correct_segment.npy
python scripts/preprocess_dataset.py --input data/test_correct.npy  --output data/test_correct_segment.npy

输入格式：
- .npy 字典，至少包含以下键：
  'old_trajectories', 'new_trajectories', 'labels', 'scene_ids'
- 若包含 'old_raw_trajectories' / 'new_raw_trajectories'，则以其为归一化来源并保留原始数据；否则使用 old/new_trajectories 作为来源。

输出格式：
- .npy 字典，覆盖写入归一化后的 'old_trajectories' / 'new_trajectories'
- 保留或新增 'old_raw_trajectories' / 'new_raw_trajectories'
- 在 'meta' 中写入 {'normalized': True, 'normalization_mode': <mode>} 以便加载器跳过运行时归一化
"""

import argparse
import os
import numpy as np
import logging
from typing import Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.normalization import TrajectoryNormalizer


def minmax_per_segment(traj: np.ndarray) -> np.ndarray:
    """对单个轨迹段进行MinMax归一化（按时间维聚合每个特征）。"""
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2:
        normalizer = TrajectoryNormalizer(feature_range=(0.0, 1.0))
        return normalizer.fit_transform(traj)
    feat_min = traj.min(axis=0, keepdims=True)
    feat_max = traj.max(axis=0, keepdims=True)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    return (traj - feat_min) / feat_range


def preprocess_npy(input_path: str, output_path: str) -> None:
    data = np.load(input_path, allow_pickle=True).item()

    required = ['old_trajectories', 'new_trajectories', 'labels', 'scene_ids']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Input dataset missing keys: {missing}")

    labels = data['labels']
    scene_ids = data['scene_ids']
    old_src = data.get('old_raw_trajectories', data['old_trajectories'])
    new_src = data.get('new_raw_trajectories', data['new_trajectories'])

    old_src = np.asarray(old_src, dtype=np.float32)
    new_src = np.asarray(new_src, dtype=np.float32)

    assert old_src.shape == new_src.shape, "old/new shape mismatch"

    # 归一化：统一为段级MinMax
    old_norm = np.stack([minmax_per_segment(seg) for seg in old_src], axis=0)
    new_norm = np.stack([minmax_per_segment(seg) for seg in new_src], axis=0)

    # 组装并保存输出
    out: Dict[str, Any] = dict(data)  # 拷贝字典，以防修改引用
    # 保留原始（若不存在则写入）
    if 'old_raw_trajectories' not in out:
        out['old_raw_trajectories'] = old_src
    if 'new_raw_trajectories' not in out:
        out['new_raw_trajectories'] = new_src
    # 写入归一化结果
    out['old_trajectories'] = old_norm
    out['new_trajectories'] = new_norm
    # 更新meta
    meta = out.get('meta', {})
    if not isinstance(meta, dict):
        meta = {}
    meta['normalized'] = True
    meta['normalization_mode'] = 'segment'
    out['meta'] = meta

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, out)
    logging.info(
        f"Preprocessed dataset saved: {output_path} | mode=segment | pairs={len(labels)}"
    )


def main():
    parser = argparse.ArgumentParser(description="离线归一化预处理脚本（统一段级MinMax）")
    parser.add_argument('--input', required=True, help="输入 .npy 数据集路径")
    parser.add_argument('--output', required=True, help="输出 .npy 保存路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    preprocess_npy(args.input, args.output)


if __name__ == '__main__':
    main()