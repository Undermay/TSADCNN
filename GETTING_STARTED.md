# TSADCNN 上手指南（Getting Started）

本指南帮助你在 Windows 环境快速完成：环境准备、数据生成与可视化、模型训练、模型评估，以及常见问题排查。文档中的命令均以 PowerShell 为例，默认在项目根目录 `e:\TraeWorkSpace\TSADCNN` 执行。

## 项目概览
- 目标：利用 TSADCNN 进行二维轨迹段（x, y, vx, vy）编码与双对比学习，支持轨迹关联评估与可视化。
- 关键脚本：
  - `generate_datasets.py` 生成训练/验证数据（支持场景信息）。
  - `visualize_datasets.py` 绘制各运动模型轨迹分布 PNG。
  - `train.py` 训练 TSADCNN 模型（验证阶段按论文计算场景级 AP 与 P@K）。
  - `evaluate.py` 对训练好的模型进行评估与可视化。
  - `config.quick.yaml`（建议新建）用于快速验证配置。

## 目录结构（精简后）
- `data/` 训练与验证数据（`train_data.npy`、`val_data.npy`）
- `models/` 模型结构与组件（`tsadcnn.py`、`encoder.py`、`projection.py`）
- `utils/` 数据加载、增强与评估指标（`data_loader.py`、`augmentation.py`、`metrics.py`）
- `outputs/` 训练或可视化输出目录（按需生成）
- `evaluation_results/` 评估结果与可视化（按命令生成）
- `train.py` / `evaluate.py` / `generate_datasets.py` / `visualize_datasets.py`
- `config.quick.yaml`（快速配置示例）

## 生成数据

- 生成包含场景信息的数据（默认每场景 `K=8`）：
- 命令：`python generate_datasets.py --output_dir data --num_targets 800 --sequence_length 64 --k_per_scene 8 --val_ratio 0.2`
- 结果：在 `data/` 下产生 `train_data.npy` 与 `val_data.npy`，内含 `segments/labels/scene_ids/local_target_ids`。

## 训练

- 使用快速配置训练：
- 命令：`python train.py --config config.quick.yaml`
- 监控日志：`logs/train.log`
- 关键验证指标：`SceneAP` 与 `P@K`（默认 `K=8`）。最佳模型以 `SceneAP` 作为唯一选择指标。

## 评估与可视化

- 评估（需要场景信息）：
- 命令：`python evaluate.py --config config.quick.yaml`
- 输出：打印 `Scene AP` 与 `P@K`，可选生成嵌入可视化图。

## 常见问题与排查
- CUDA 不可用或训练慢：将 `hardware.use_gpu` 设为 `false`，并调小 `batch_size`/`epochs`。
- 配置键不匹配导致 KeyError：使用 `config.quick.yaml`，与当前代码保持一致。
- 评估报错或可视化耗时：可降低 `num_targets` 或 `sequence_length` 以快速运行。

## 复现建议
- 完整训练：将 `training.epochs` 提升至 50~200，视显存调整 `batch_size`；先在干净数据训练，再做鲁棒性验证。
- 噪声等级对比：用多组参数生成数据，分别训练并对比 `SceneAP` 与 `P@K`。

## 清理与维护
- 可按需清理旧中间文件；如需恢复请重新运行生成脚本。