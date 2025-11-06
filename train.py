import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import create_data_loaders
from model import TSADCNN, contrastive_loss




def eval_linking_top1(model: TSADCNN, dataset, device: torch.device, margin: float = 0.2) -> float:
    """
    Top-1 链接准确率：
    对每个 (scene_id, old_target_flag) 的旧轨迹，
    在该组内所有候选新轨迹里选择欧氏距离最近的一个，
    若其标签为 1（同目标）则计为正确。
    返回“正确组数/总组数”。
    """
    model.eval()
    # 分组：同一场景、同一 old_target 的所有候选
    groups = {}
    pairs = dataset.trajectory_pairs
    for p in pairs:
        key = (int(p['scene_id']), int(p['old_target_flag']))
        groups.setdefault(key, []).append(p)

    correct = 0
    total = 0
    with torch.no_grad():
        for _, plist in groups.items():
            # old 片段（同组共享同一 old）
            old_np = plist[0]['old_trajectory']
            new_np_list = [q['new_trajectory'] for q in plist]
            labels = torch.tensor([int(q['label']) for q in plist], dtype=torch.long, device=device)
            n = len(new_np_list)

            old = torch.tensor(old_np, dtype=torch.float32, device=device).unsqueeze(0).repeat(n, 1, 1)
            new = torch.tensor(np.stack(new_np_list), dtype=torch.float32, device=device)

            z_old, z_new = model(old, new)
            distances = torch.norm(z_old - z_new, dim=1)
            min_idx = torch.argmin(distances)
            if labels[min_idx].item() == 1:
                correct += 1
            total += 1

    return correct / max(total, 1)


def eval_scene_precision_pk(model: TSADCNN, dataset, device: torch.device, method: str = 'auto'):
    """
    计算场景级 P@K：
    - 每个场景包含 K 个 old 与 K 个 new 目标（一对一匹配）
    - 构建场景内的距离矩阵 D[i,j] = ||z_old_i - z_new_j||_2
    - 使用匈牙利算法（若不可用则贪心匹配）得到 K 对预测配对
    - 与真实标注的“正确配对”比较，P@K = 正确配对数 / K
    返回：({K: pk值}, overall_micro)
    """
    model.eval()
    # 收集每个场景的所有样本行
    scenes = {}
    for p in dataset.trajectory_pairs:
        sid = int(p['scene_id'])
        scenes.setdefault(sid, []).append(p)

    by_k = {}
    total_correct_all = 0
    total_targets_all = 0
    with torch.no_grad():
        for sid, plist in scenes.items():
            # 该场景内的唯一 old/new 标识
            old_map = {}
            new_map = {}
            correct_pairs = set()
            for q in plist:
                of = int(q['old_target_flag'])
                nf = int(q['new_target_flag'])
                # 记录首个出现的轨迹片段（去重）
                if of not in old_map:
                    old_map[of] = q['old_trajectory']
                if nf not in new_map:
                    new_map[nf] = q['new_trajectory']
                if int(q['label']) == 1:
                    correct_pairs.add((of, nf))

            old_flags = sorted(list(old_map.keys()))
            new_flags = sorted(list(new_map.keys()))
            K = min(len(old_flags), len(new_flags))
            if K == 0:
                continue

            # 构建张量并编码为嵌入
            old_list = [old_map[f] for f in old_flags[:K]]
            new_list = [new_map[f] for f in new_flags[:K]]
            old = torch.tensor(np.stack(old_list), dtype=torch.float32, device=device)
            new = torch.tensor(np.stack(new_list), dtype=torch.float32, device=device)
            z_old = model.encode(old)
            z_new = model.encode(new)
            # 距离矩阵 [K,K]
            dist_mat = torch.cdist(z_old, z_new, p=2)

            # 选择匹配算法：hungarian/greedy/auto（auto 优先匈牙利）
            use_hungarian = None
            if method == 'hungarian':
                use_hungarian = True
            elif method == 'greedy':
                use_hungarian = False
            else:
                try:
                    from scipy.optimize import linear_sum_assignment  # noqa: F401
                    use_hungarian = True
                except Exception:
                    use_hungarian = False

            row_ind = []
            col_ind = []
            if use_hungarian:
                try:
                    from scipy.optimize import linear_sum_assignment
                    ri, ci = linear_sum_assignment(dist_mat.detach().cpu().numpy())
                    row_ind = list(ri)
                    col_ind = list(ci)
                except Exception:
                    use_hungarian = False  # 回退到贪心

            if not use_hungarian:
                dm = dist_mat.detach().cpu().numpy().copy()
                assigned_rows = set()
                assigned_cols = set()
                for _ in range(K):
                    min_val = float('inf')
                    min_pos = (-1, -1)
                    for i in range(K):
                        if i in assigned_rows:
                            continue
                        for j in range(K):
                            if j in assigned_cols:
                                continue
                            v = dm[i, j]
                            if v < min_val:
                                min_val = v
                                min_pos = (i, j)
                    i, j = min_pos
                    if i == -1 or j == -1:
                        break
                    row_ind.append(i)
                    col_ind.append(j)
                    assigned_rows.add(i)
                    assigned_cols.add(j)

            # 统计正确配对数
            correct_cnt = 0
            for i, j in zip(row_ind, col_ind):
                of = old_flags[i]
                nf = new_flags[j]
                if (of, nf) in correct_pairs:
                    correct_cnt += 1

            by_k.setdefault(K, {'correct': 0, 'total': 0})
            by_k[K]['correct'] += correct_cnt
            by_k[K]['total'] += K
            total_correct_all += correct_cnt
            total_targets_all += K

    pk_values = {k: (v['correct'] / v['total'] if v['total'] > 0 else 0.0) for k, v in by_k.items()}
    overall = (total_correct_all / total_targets_all) if total_targets_all > 0 else 0.0
    return pk_values, overall


def train_epoch(model: TSADCNN, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, margin: float = 0.2):
    model.train()
    running_loss = 0.0
    count = 0
    for old_traj, new_traj, labels in loader:
        old_traj = old_traj.to(device)
        new_traj = new_traj.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        z_old, z_new = model(old_traj, new_traj)
        loss = contrastive_loss(z_old, z_new, labels, margin)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size
    return running_loss / count


def eval_epoch(model: TSADCNN, loader: DataLoader, device: torch.device, margin: float = 0.2):
    model.eval()
    running_loss = 0.0
    count = 0
    with torch.no_grad():
        for old_traj, new_traj, labels in loader:
            old_traj = old_traj.to(device)
            new_traj = new_traj.to(device)
            labels = labels.to(device)
            z_old, z_new = model(old_traj, new_traj)
            loss = contrastive_loss(z_old, z_new, labels, margin)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size
    return running_loss / count


def parse_args():
    parser = argparse.ArgumentParser(description='TSADCNN Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--margin', type=float, default=0.9, help='Contrastive margin')
    parser.add_argument('--train-path', type=str, default='data/train_trajectories.npy', help='Train data path')
    parser.add_argument('--test-path', type=str, default='data/test_trajectories.npy', help='Test data path')
    return parser.parse_args()


def main(args):
    feature_dim = 6
    hidden_dim = 128
    num_layers = 2
    conv_channels = 32
    embed_dim = 32
    dropout = 0.1
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    margin = args.margin

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, test_loader = create_data_loaders(
        train_path=args.train_path,
        test_path=args.test_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    model = TSADCNN(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        conv_channels=conv_channels,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs('logs', exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    # 在文件名加入常用超参数，便于快速区分不同实验
    log_fname = f'train_metrics_{timestamp}_e{epochs}_bs{batch_size}_lr{lr}_m{margin}.txt'
    log_path = os.path.join('logs', log_fname)
    print('Starting minimal training...')
    print(f'Logging to {log_path}')
    lf = open(log_path, 'w', encoding='utf-8')
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, margin)
        val_loss = eval_epoch(model, test_loader, device, margin)
        # Top-1 链接准确率（按场景+旧目标分组，挑最近候选）
        val_top1 = eval_linking_top1(model, test_loader.dataset, device)
        # 训练集 AP（使用匈牙利）
        _, pk_overall_train = eval_scene_precision_pk(model, train_loader.dataset, device, method='hungarian')
        ap_train = pk_overall_train
        # 验证集 P@K 与 AP（匈牙利）
        pk_by_k_val_h, pk_overall_val_h = eval_scene_precision_pk(model, test_loader.dataset, device, method='hungarian')
        ap_val_h = pk_overall_val_h
        # 验证集 AP（贪心）
        _, pk_overall_val_g = eval_scene_precision_pk(model, test_loader.dataset, device, method='greedy')
        ap_val_g = pk_overall_val_g
        pk_parts = ' '.join([f'P@{k} {pk_by_k_val_h[k]:.4f}' for k in sorted(pk_by_k_val_h.keys())])
        line = (
            f'Epoch {epoch}/{epochs} | '
            f'Train Loss {train_loss:.4f} AP {ap_train:.4f} | '
            f'Val Loss {val_loss:.4f} Top1 {val_top1:.4f} AP(H) {ap_val_h:.4f} AP(G) {ap_val_g:.4f} Diff {ap_val_h - ap_val_g:.4f} | '
            f'{pk_parts}'
        )
        print(line)
        lf.write(line + '\n')
    # 在日志末尾追加本次训练的参数配置，便于核对与复现
    lf.write('\nRun Config\n')
    lf.write(f'epochs={epochs}\n')
    lf.write(f'batch_size={batch_size}\n')
    lf.write(f'lr={lr}\n')
    lf.write(f'weight_decay={weight_decay}\n')
    lf.write(f'margin={margin}\n')
    lf.write(f'num_workers={args.num_workers}\n')
    lf.write(f'train_path={args.train_path}\n')
    lf.write(f'test_path={args.test_path}\n')
    lf.write(f'device={device}\n')
    lf.write('model.feature_dim=6\n')
    lf.write('model.hidden_dim=128\n')
    lf.write('model.num_layers=2\n')
    lf.write('model.conv_channels=32\n')
    lf.write('model.embed_dim=32\n')
    lf.write('model.dropout=0.1\n')
    lf.write('--- end ---\n')
    lf.close()
    print(f'Training complete. Metrics saved to {log_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args)