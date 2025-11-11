TSADCNN 最小上手项目

项目内容
- `model.py`：模型模块（LSTM 编码、时间相关矩阵、卷积残差空间提取、投影头）与对比损失。
- `train.py`：训练与评估脚本，记录每个 epoch 的指标到 `logs/train_metrics.txt`。
- `data_utils.py`：数据加载与 `DataLoader` 包装，支持 `.npy` 字典格式数据。
- `generate_dataset.py`：按文档规范生成场景与轨迹对，可选使用。
- `data/`：数据文件（`train_trajectories.npy`、`test_trajectories.npy`）。

依赖安装
- `pip install -r requirements.txt`
- 仅需要：`torch>=2.0`、`numpy>=1.24`。（评估使用匈牙利算法需 `scipy`，如未安装请另外 `pip install scipy`）

数据准备（可选）
- 重新生成数据集并写入 `data/`：
  - `python generate_dataset.py`
- 数据格式（字典）：
  - 关键键：`old_trajectories`、`new_trajectories`、`labels`、`scene_ids`、`old_target_flags`、`new_target_flags`、`motion_modes`、`meta`。
  - 维度：`old/new` 片段为 `[N, 13, 6]`，每步特征 `[x, y, vx, vy, ax, ay]`。

训练与评估
- 运行训练：
  - `python train.py`
  - 归一化选项：`--normalize {embed|minmax|both}`，默认 `both`
    - `embed`：仅启用嵌入 L2 归一化（`F.normalize`）。
    - `minmax`：仅启用输入特征 Min‑Max 归一化（基于训练集统计，验证/测试共享）。
    - `both`：同时启用输入 Min‑Max 与嵌入 L2（推荐默认）。
- 指标记录：
  - 日志文件按时间戳创建，且带常用超参数前缀：
    - 规则：`logs/train_metrics_YYYYMMDD-HHMMSS_e{epochs}_bs{batch_size}_lr{lr}_m{margin}.txt`
    - 示例：`logs/train_metrics_20251105-213843_e1_bs64_lr0.001_m0.7.txt`
    - 每次训练都会生成新文件，不会覆盖旧日志。
  - 每行格式：`Epoch i/E | Train Loss a AP_train w1 | Val Loss c AP_val w2 | P@2 u P@3 v P@5 ...`
  - 日志末尾会追加本次训练的参数块（Run Config），包含：`epochs`、`batch_size`、`lr`、`weight_decay`、`margin`、`num_workers`、`train_path`、`test_path`、`device` 以及主要模型结构参数，便于对比与复现。
  - 指标：
    - `P@K`：场景级一对一匹配精度（正确配对数 / 该场景目标数 K）；
    - `AP`：跨所有场景的微平均一对一匹配精度，`AP = (∑ 正确配对数) / (∑ K)`；

核心原理与度量
- 嵌入生成：`TSADCNN.encode` 输出归一化嵌入 `z`（`F.normalize`）。
- 距离度量：`D = ||z_old - z_new||_2`。
- 对比损失：
  - 正样本：`D^2`；负样本：`(max(0, margin - D))^2`；总损失为两者平均的 0.5 倍。
- 评估：
  - `P@K`：在场景内构建距离矩阵并做一对一全局匹配，统计正确配对比例。
  - `AP`：跨场景微平均正确配对率（与 `margin` 无关）。

可调整项（在 `train.py`）
- 训练超参：`batch_size`、`epochs`、`lr`、`weight_decay`、`hidden_dim`、`num_layers`、`conv_channels`、`embed_dim`、`dropout`。
- 准确率与损失的 `margin`：与嵌入分布相关，可在验证集上扫 `0.1–1.0` 选择最优；归一化嵌入下，`margin=0.2` 较严格。
- DataLoader 并行：Windows 如遇多进程问题，可在 `create_data_loaders(..., num_workers=0)`。
 - 归一化：`--normalize` 支持 `embed|minmax|both`，默认 `both`；不再提供 `none` 选项。

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
 - 说明：本项目评估以场景级 `P@K` 与总体 `AP` 为主；`Top‑K` 概念用于理解距离排序的命中率但未作为代码指标输出。
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

AP（Average Precision，按原文定义）
- 定义：跨所有场景的微平均正确配对率。
- 公式：`AP = (∑ 正确配对数) / (∑ 场景目标数K)`，等价于本项目日志中的 `P@all`。
- 说明：AP 与 `margin` 无关，完全基于一对一全局匹配的结果；相比 `Acc`（逐对阈值分类），AP 更能反映整体场景级关联质量。

改动说明

- 脚本更新： export_csv.py 默认使用 --mode strict
- 输出文件：
  - data/train_dataset.csv
  - data/test_dataset.csv
- 每一行对应一个样本对，数组字段以 JSON 字符串写入，保留原始形状 [13,6]
- 字段包含：
  - split 、 pair_index 、 scene_id 、 label 、 old_flag 、 new_flag 、 motion_mode
  - old_trajectory_json 、 new_trajectory_json
  - old_raw_json 、 new_raw_json （若原始米单位片段存在则写入，否则为空）
使用方式

- 严格镜像（默认）
  - python export_csv.py --out-dir data --mode strict
- 如果你仍需之前的“长表/元信息”两类 CSV，可以使用：
  - python export_csv.py --out-dir data --mode long
  - 会生成 train_pairs.csv 、 train_trajectories.csv 、 test_pairs.csv 、 test_trajectories.csv