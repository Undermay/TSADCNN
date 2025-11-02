# TSADCNN 项目上手文档

## 项目概述

TSADCNN (Track Segment Association With Dual Contrast Neural Network) 是一个基于对比学习的轨迹关联深度学习项目。该项目专注于解决多目标跟踪中的轨迹关联问题，通过学习轨迹的时空特征表示来判断不同轨迹段是否属于同一目标。

### 核心特性
- **对比学习框架**：使用old-new轨迹对进行训练，学习区分同目标和不同目标的轨迹
- **时空特征融合**：结合LSTM时间建模和CNN空间建模
- **段级归一化策略**：支持段级MinMax归一化，确保数据质量
- **完整评估体系**：基于P@K和AP指标的轨迹关联性能评估

## 项目架构

### 目录结构
```
TSADCNN/
├── models/                    # 模型定义
│   ├── tsadcnn.py            # 主模型类和损失函数
│   ├── encoder.py            # 轨迹编码器
│   └── projection.py         # 投影头
├── utils/                     # 工具模块
│   ├── contrastive_data_loader.py  # 对比学习数据加载器
│   ├── normalization.py      # 数据归一化工具
│   └── metrics_pak.py        # P@K评估指标
├── scripts/                   # 脚本工具
│   └── preprocess_dataset.py # 数据预处理脚本
├── data/                      # 数据目录
├── logs_v4/                   # 日志目录
├── checkpoints/               # 模型检查点
├── train_simplified.py       # 简化训练脚本
├── generate_correct_dataset.py # 数据生成脚本
└── config_improved_v4.yaml   # 配置文件
```

### 核心组件关系图
```
训练脚本 (train_simplified.py)
    ↓
配置加载 (config_improved_v4.yaml)
    ↓
数据加载器 (ContrastiveTrajectoryDataset)
    ↓
模型创建 (TSADCNN)
    ├── 编码器 (TrackEncoder)
    └── 投影头 (ProjectionHead)
    ↓
训练循环
    ├── 前向传播
    ├── 损失计算 (contrastive_loss + symmetric_constraint)
    ├── 反向传播
    └── 参数更新
    ↓
模型评估 (metrics_pak)
    ├── P@K指标
    └── AP指标
```

## 网络架构详解

### 1. TSADCNN主模型 (`models/tsadcnn.py`)

#### 核心类和方法：

**TSADCNNConfig**
- 配置数据类，定义模型的所有超参数
- 包含输入维度、编码器参数、投影头参数、损失函数参数等

**TSADCNN**
- 主模型类，继承自`nn.Module`
- **关键方法**：
  - `__init__(cfg)`: 初始化编码器和投影头
  - `encode_trajectory(traj)`: 编码单个轨迹（用于推理）
  - `forward(old, new, labels)`: 训练时的前向传播

**损失函数**：
- `contrastive_loss()`: 对比损失，区分正负样本对
- `symmetric_constraint_from_corr()`: 对称约束损失，基于相关矩阵

### 2. 轨迹编码器 (`models/encoder.py`)

#### TrackEncoder类
**网络结构**：
1. **时间特征提取**：双向LSTM → 时间相关性图投影
2. **空间特征提取**：位置归一化 → 空间相关矩阵 → 3×3和1×1卷积 → 残差块堆叠
3. **特征融合**：时空特征拼接 → 全连接层
4. **维度提升**：最终特征映射

**关键方法**：
- `forward(track_sequence)`: 返回编码特征和相关矩阵

### 3. 投影头 (`models/projection.py`)

#### ProjectionHead类
- 将编码器输出映射到对比学习的嵌入空间
- 多层线性层 + BatchNorm + LeakyReLU + Dropout
- 支持可配置的层数和维度

## 数据流详解

### 1. 数据生成 (`generate_correct_dataset.py`)

**数据格式**：
- 32步完整轨迹 → 13步old + 6步gap + 13步new
- 特征维度：[x, y, vx, vy, ax, ay] (位置+速度+加速度，6维)
- 标签：同目标=1，不同目标=0

**运动模式**：
- CV (Constant Velocity): 匀速运动
- CA (Constant Acceleration): 匀加速运动  
- CT (Coordinated Turn): 协调转弯（小、中、大角度）

### 2. 数据加载 (`utils/contrastive_data_loader.py`)

#### ContrastiveTrajectoryDataset类

**数据预处理流程**：
```
原始轨迹数据
    ↓
格式检测 (CSV/PKL)
    ↓
数据解析和验证
    ↓
段级MinMax归一化
    ↓
张量转换
    ↓
批次采样 (可选场景分组)
```

**关键方法**：
- `_load_data()`: 数据加载和格式适配
- `_apply_segment_minmax_normalization()`: 段级归一化
- `__getitem__()`: 返回(old_traj, new_traj, label)

### 3. 归一化工具 (`utils/normalization.py`)

#### TrajectoryNormalizer类
- 支持MinMax归一化到指定范围
- 可处理批量轨迹数据
- 支持正向和逆向变换

## 训练流程详解

### 1. 训练脚本 (`train_simplified.py`)

#### 主要函数调用关系：
```
main()
├── load_config()                    # 加载配置
├── create_contrastive_data_loaders() # 创建数据加载器
├── create_model()                   # 创建模型
├── 训练循环
│   ├── train_epoch()               # 训练一个epoch
│   ├── validate_epoch()            # 验证
│   ├── scheduler.step()            # 学习率调度
│   └── save_checkpoint()           # 保存检查点
└── run_evaluation()                # 最终评估
```

#### 训练epoch流程：
```python
def train_epoch():
    for batch in train_loader:
        old_segments, new_segments, labels = batch
        
        # 前向传播
        z_old, z_new, losses = model(old_segments, new_segments, labels)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # 计算准确率
        accuracies = compute_accuracy(z_old, z_new, labels, threshold, margin)
```

### 2. 损失函数组成

**总损失** = 对比损失 + λ × 对称约束损失

#### 对比损失 (Contrastive Loss)
**代码位置**：`models/tsadcnn.py` 第57-75行

**公式**：
```
cos_sim = (z_old * z_new).sum(dim=1)  # 余弦相似度
d = sqrt(2 * (1 - cos_sim))           # 欧几里得距离
L_contrast = 0.5 * [l * d² + (1-l) * max(0, margin-d)²]
```

其中：
- `z_old`, `z_new`：L2归一化后的嵌入向量
- `l`：标签 (1=正样本, 0=负样本)
- `margin`：边界参数 (默认0.2)

#### 对称约束损失 (Symmetric Constraint Loss)
**代码位置**：`models/tsadcnn.py` 第77-81行

**公式**：
```
L_symmetric = mean((A - A^T)²)
```

其中：
- `A`：编码器输出的相关矩阵 [B, L, L]
- `A^T`：A的转置矩阵
- 约束相关矩阵的对称性，提高时序特征质量

#### 总损失
```
L_total = L_contrast + λ_symmetric * L_symmetric
```
- `λ_symmetric`：对称约束权重 (默认10.0)

### 3. 优化策略

- **优化器**：AdamW (权重衰减)
- **学习率调度**：CosineAnnealingLR
- **梯度裁剪**：防止梯度爆炸
- **混合精度**：可选的AMP训练

## 评估机制详解

### 1. 评估指标 (`utils/metrics_pak.py`)

#### 核心评估函数：

**compute_scene_level_pak_correct()**
- 计算场景级P@K指标
- 支持并行计算
- 返回各K值的精确度

**evaluate_contrastive_model_correct()**
- 完整的模型评估流程
- 计算P@K、AP和准确率指标
- 输出详细的评估报告

#### 评估流程：
```
模型推理
    ↓
嵌入向量提取
    ↓
相似度计算 (余弦相似度)
    ↓
场景内排序
    ↓
P@K计算 (前K个中正确关联的比例)
    ↓
AP计算 (平均精度)
```

### 2. 指标含义

- **P@K**: 在有K个目标的场景中，正确关联目标数与场景内目标总数K的比值
- **AP**: 全局的正确关联率，所有场景的平均关联精度
- **Accuracy**: 基于阈值的二分类准确率

**P@K计算方式**：
```
对于每个有K个目标的场景：
P@K = 该场景中正确关联的目标数 / K

最终P@K = 所有K目标场景的P@K平均值
```

**AP计算方式**：
```
AP = 所有场景的正确关联目标数 / 所有场景的总目标数
```

## 配置文件详解

### config_improved_v4.yaml 主要配置项：

```yaml
model:                          # 模型参数
  input_dim: 6                 # 输入特征维度 (x,y,vx,vy,ax,ay)
  encoder_hidden_dim: 128      # 编码器隐藏维度
  projection_output_dim: 64    # 投影输出维度

data:                          # 数据参数
  sequence_length: 13          # 轨迹段长度
  batch_size: 32              # 批次大小

training:                      # 训练参数
  epochs: 50                   # 训练轮数
  learning_rate: 0.001         # 学习率
  weight_decay: 0.01          # 权重衰减

loss:                          # 损失函数参数
  margin: 1.0                  # 对比损失边界
  lambda_sym: 0.1             # 对称约束权重

evaluation:                    # 评估参数
  k_values: [5, 10, 20]       # P@K的K值
  similarity_threshold: 0.5    # 相似度阈值
```

## 使用指南

### 1. 环境配置

```bash
# 安装依赖
pip install torch torchvision numpy scikit-learn pyyaml

# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 数据准备

```bash
# 生成训练数据
python generate_correct_dataset.py

# 预处理数据（可选）
python scripts/preprocess_dataset.py --input data/train.csv --output data/train_normalized.pkl
```

### 3. 模型训练

```bash
# 基础训练
python train_simplified.py --config config_improved_v4.yaml

# 指定参数训练
python train_simplified.py --config config_improved_v4.yaml --epochs 100 --lr 0.0005

# 恢复训练
python train_simplified.py --config config_improved_v4.yaml --resume checkpoints/latest_checkpoint.pth
```

### 4. 模型评估

```bash
# 仅评估模式
python train_simplified.py --config config_improved_v4.yaml --eval_only --checkpoint checkpoints/best_checkpoint.pth
```

## 扩展开发

### 1. 添加新的运动模式

在`generate_correct_dataset.py`中的`MotionGenerator`类添加新的运动模式：

```python
def generate_new_motion(self, steps, **params):
    # 实现新的运动模式
    pass
```

### 2. 自定义损失函数

在`models/tsadcnn.py`中添加新的损失函数：

```python
def custom_loss(z_old, z_new, labels, **kwargs):
    # 实现自定义损失
    pass
```

### 3. 新增评估指标

在`utils/metrics_pak.py`中添加新的评估函数：

```python
def compute_custom_metric(embeddings, labels):
    # 实现自定义指标
    pass
```

## 常见问题

### 1. 内存不足
- 减小batch_size
- 使用梯度累积
- 启用混合精度训练

### 2. 训练不收敛
- 检查学习率设置
- 调整损失函数权重
- 验证数据质量

### 3. 评估指标异常
- 确认数据格式正确
- 检查相似度阈值设置
- 验证场景ID和目标ID

## 性能优化建议

1. **数据加载优化**：
   - 使用多进程数据加载 (`num_workers > 0`)
   - 预处理数据并缓存
   - 使用场景分组批次采样

2. **训练优化**：
   - 启用混合精度训练
   - 使用梯度累积处理大批次
   - 定期保存检查点

3. **推理优化**：
   - 批量推理
   - 模型量化
   - ONNX导出

## 项目维护

### 日志管理
- 训练日志保存在 `logs_v4/` 目录
- 支持不同级别的日志输出
- 定期清理旧日志文件

### 检查点管理
- 自动保存最新和最佳检查点
- 支持断点续训
- 定期备份重要检查点

### 代码质量
- 遵循PEP 8代码规范
- 添加类型注解
- 编写单元测试

---

本文档涵盖了TSADCNN项目的核心架构、实现细节和使用方法。如有疑问，请参考代码注释或联系项目维护者。