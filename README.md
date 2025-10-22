# TSADCNN: Track Segment Association with Dual Contrast Neural Network

本项目是对论文《Track Segment Association With Dual Contrast Neural Network》的复现实现。

## 项目概述

TSADCNN是一种基于深度度量学习的轨迹段关联方法，通过双对比学习机制解决雷达数据处理中的轨迹中断问题。该方法相比传统的基于假设目标运动模型的统计估计方法，具有更好的适应性和抗噪声能力。

## 核心特性

1. **时空信息提取模块**: 基于深度度量学习提取轨迹信息，特别是对称相关信息并进行升维
2. **双对比学习机制**: 通过构建适当的优化函数，使属于同一目标的轨迹段在高维空间中更接近，不同目标的轨迹段更远离
3. **最近邻关联**: 在高维空间中选择最近邻向量作为关联轨迹

## 项目结构

```
TSADCNN/
├── models/
│   ├── __init__.py
│   ├── tsadcnn.py          # TSADCNN主模型
│   ├── encoder.py          # 时空信息提取编码器
│   └── projection.py       # 投影头
├── data/
│   ├── __init__.py
│   ├── dataset.py          # 数据集处理
│   └── augmentation.py     # 数据增强
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # 评估指标
├── train.py                # 训练脚本
├── eval.py                 # 评估脚本
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型
```bash
python train.py --config config.yaml
```

### 评估模型
```bash
python eval.py --model_path checkpoints/best_model.pth
```

## 损失函数

本项目实现了论文中的三个核心损失函数公式：

### 公式23 - 距离优化对比损失 (Lc)
```
Lc = 1/2 * l * D² + 1/2 * (1-l) * [max(0, m-D)]²
```
- **实现位置**: `models/tsadcnn.py` 中的 `_compute_distance_contrastive_loss` 方法
- **具体计算**: 基于欧几里得距离的对比损失，其中l为标签（同类为1，异类为0），D为特征距离，m为边界参数
- **作用**: 使同类样本距离最小化，异类样本距离在边界m之外最大化

### 公式24 - 对称约束损失 (Ls)  
```
Ls = ΣΣ(aij - aji)²
```
- **实现位置**: `models/tsadcnn.py` 中的 `_compute_symmetric_constraint_loss` 方法
- **具体计算**: 计算相关性矩阵的对称性约束，其中aij表示第i个轨迹点与第j个轨迹点的相关性
- **作用**: 保持时间-空间相关性信息提取的对称性，确保相关性矩阵满足对称约束

### 公式25 - 总损失函数
```
L = 10 * Ls + Lc
```
- **实现位置**: `models/tsadcnn.py` 中的 `compute_dual_contrastive_loss` 方法返回值
- **权重配比**: 严格按照论文设定，对称约束损失权重为10，距离优化损失权重为1
- **注意**: `temporal_weight` 和 `spatial_weight` 参数不再参与损失加权

## 参考文献

Xiong, W., Xu, P., Cui, Y., Xiong, Z., Lv, Y., & Gu, X. (2022). Track Segment Association with Dual Contrast Neural Network. IEEE Transactions on Aerospace and Electronic Systems, 58(1), 247-261.