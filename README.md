TSADCNN 最小上手项目

项目内容
- `model.py`：模型模块（LSTM 编码、时间相关矩阵、卷积残差空间提取、投影头）与对比损失。
- `train.py`：训练与评估脚本，记录每个 epoch 的指标到 `logs/train_metrics.txt`。
- `data_utils.py`：数据加载与 `DataLoader` 包装，支持 `.npy` 字典格式数据。
- `generate_dataset.py`：按文档规范生成场景与轨迹对，可选使用。
- `data/`：数据文件（`train_trajectories.npy`、`test_trajectories.npy`、`dataset_config.yaml`）。

依赖安装
- `pip install -r requirements.txt`
- 仅需要：`torch>=2.0`、`numpy>=1.24`、`PyYAML>=6.0`。

数据准备（可选）
- 重新生成数据集并写入 `data/`：
  - `python generate_dataset.py`
- 数据格式（字典）：
  - 关键键：`old_trajectories`、`new_trajectories`、`labels`、`scene_ids`、`old_target_flags`、`new_target_flags`、`motion_modes`、`meta`。
  - 维度：`old/new` 片段为 `[N, 13, 6]`，每步特征 `[x, y, vx, vy, ax, ay]`。

训练与评估
- 运行训练：
  - `python train.py`
- 指标记录：`
  - 每行格式：`Epoch i/E | Train Loss a Acc b | Val Loss c Acc d Top1 e`
  - `Acc`：逐对分类准确率（基于距离与 margin 的判定）；
  - `Top1`：链接 Top‑1 准确率（对每个旧轨迹在同场景候选中选择“距离最近”的新轨迹，并与真实标签比较）。

核心原理与度量
- 嵌入生成：`TSADCNN.encode` 输出归一化嵌入 `z`（`F.normalize`）。
- 距离度量：`D = ||z_old - z_new||_2`。
- 对比损失：
  - 正样本：`D^2`；负样本：`(max(0, margin - D))^2`；总损失为两者平均的 0.5 倍。
- 准确率：
  - 逐对 `Acc`：`D < margin` 视为关联，`D >= margin` 视为不关联；按样本数加权聚合为“正确样本数/总样本数”。
  - `Top1`：按 `(scene_id, old_target_flag)` 分组，对每组在候选新轨迹里选 `D` 最小者；若其标签为 1，记该组正确。

可调整项（在 `train.py`）
- 训练超参：`batch_size`、`epochs`、`lr`、`weight_decay`、`hidden_dim`、`num_layers`、`conv_channels`、`embed_dim`、`dropout`。
- 准确率与损失的 `margin`：与嵌入分布相关，可在验证集上扫 `0.1–1.0` 选择最优；归一化嵌入下，`margin=0.2` 较严格。
- DataLoader 并行：Windows 如遇多进程问题，可在 `create_data_loaders(..., num_workers=0)`。

常见问题排查
- 训练脚本退出码非零：
  - 检查数据文件是否存在：`data/train_trajectories.npy`、`data/test_trajectories.npy`；
  - 降低 `batch_size` 或切到 CPU（自动选择设备 `cuda`/`cpu`）。
- 日志为空：
  - 默认已用文件写入；也可使用 PowerShell：`python train.py | Tee-Object -FilePath logs/train_10epoch.txt`。
- 形状错误：
  - 确认输入片段为 `[13, 6]`，且 `input_dim=6`。


******************************************************************************************
- Top‑K 中的 K 表示“对每个旧轨迹，在同场景的候选新轨迹中取距离最小的前 K 个作为预测集合”。
含义与度量

- 目标：评估链接能力，不再用阈值，而是用距离排序。
- 计算方式（Hit@K / Top‑K）：
  - 对每个旧轨迹，计算它与同场景所有候选新轨迹的嵌入距离 D=||z_old−z_new||_2 ；
  - 按距离升序选出前 K 个候选；
  - 若这 K 个里至少有一个真实标签为 1 （同目标），该旧轨迹计为“命中”；
  - Top‑K = 命中的旧轨迹数 / 全部旧轨迹数。
- 特例： K=1 即你现在实现的 Top‑1（只看最近的一个）。
与数据里的“场景K值”的区别

- 数据生成中的 k_values （如 2/5/10/15/20 ）是“每个场景的目标数量”，不是评估指标的 K 。
- 指标里的 K 只能取候选数量的一个小值（常用 1、3、5），越大越宽松。

P@K（场景精度）
- 定义：在一个场景中有 K 个目标（即 K 对 old-new 轨迹），P@K = 该场景中“正确关联的对数”/ K。
- 解释：例如 P@20=0.9 表示此场景有 20 对目标轨迹，通过模型的关联算法，最终正确匹配了 18 对、错误 2 对，即 18/20=0.9。
- 评估步骤（建议一对一匹配）：
  - 按场景构建所有 old 与 new 的两两距离矩阵 `D[i,j] = ||z_old_i - z_new_j||_2`；
  - 使用一对一匹配算法（如匈牙利算法）在该场景内为 K 个 old 找到 K 个 new 的最优配对；
  - 与真实标注的“正确配对”比较，统计该场景内“配对正确”的数量；
  - P@K 为该场景的正确配对数 / K；跨数据集可对同 K 的场景做宏平均（或整体微平均）。
- 与 Top‑K（Hit@K）区别：Top‑K 是“对每个旧轨迹在候选中取前 K 个是否命中”，不要求一对一；P@K 是“在场景内做一对一全局匹配后，正确配对的比例”。