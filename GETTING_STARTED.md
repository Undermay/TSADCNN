# TSADCNN 上手指南（Getting Started）

本指南帮助你在 Windows 环境快速完成：环境准备、数据生成与可视化、模型训练、模型评估，以及常见问题排查。文档中的命令均以 PowerShell 为例，默认在项目根目录 `e:\TraeWorkSpace\TSADCNN` 执行。

## 项目概览
- 目标：利用 TSADCNN 进行二维轨迹段（x, y, vx, vy）编码与双对比学习，支持轨迹关联评估与可视化。
- 关键脚本：
  - `generate_datasets.py` 生成训练/测试数据（支持干净与带噪版本）。
  - `visualize_datasets.py` 绘制各运动模型轨迹分布 PNG。
  - `train.py` 训练 TSADCNN 模型。
  - `evaluate.py` 对训练好的模型进行评估与可视化（t-SNE、PCA、性能对比）。
  - `config.quick.yaml`（建议新建）用于快速验证配置，避免与现有 `config.yaml` 的键不匹配。

## 目录结构（精简后）
- `data/` 训练与测试数据（`train_data.npy`、`test_data.npy` 以及对应带噪版本）
- `models/` 模型结构与组件（`tsadcnn.py`、`encoder.py`、`projection.py`）
- `utils/` 数据加载、增强与评估指标（`data_loader.py`、`augmentation.py`、`metrics.py`）
- `outputs/` 训练或可视化输出目录（按需生成）
- `evaluation_results/` 评估结果与可视化（按命令生成）
- `train.py` / `evaluate.py` / `generate_datasets.py` / `visualize_datasets.py`
- `config.yaml`（原始配置，键可能与当前代码不完全匹配）

## 环境准备
- 安装 Python 3.9+ 和 pip。
- 创建虚拟环境（可选，推荐）：
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
- 安装依赖：
  - `pip install -r requirements.txt`
- 额外依赖（评估可视化）：
  - `pip install scikit-learn seaborn matplotlib`

## 数据生成与可视化
- 生成大规模数据（每类 10,000 训练段、1,000 测试段；同时生成带噪版本）：
  - `python generate_datasets.py --train_segments_per_mode 10000 --test_segments_per_mode 1000 --generate_noisy --noisy_pos_std 1.0 --noisy_vel_std 0.5`
- 生成小规模快速验证数据（建议在性能验证时使用，避免长时间运行）：
  - `python generate_datasets.py --train_segments_per_mode 100 --test_segments_per_mode 20 --generate_noisy --noisy_pos_std 0.5 --noisy_vel_std 0.2`
- 可视化（中文注释已处理字体兼容）：
  - 干净数据：`python visualize_datasets.py --data_variant clean --save_path outputs/motion_models.png`
  - 带噪数据：`python visualize_datasets.py --data_variant noisy --save_path outputs/motion_models_noisy.png`
- 浏览器预览（本地 HTTP 服务）：
  - `python -m http.server 8000`
  - 打开 `http://localhost:8000/outputs/motion_models.png`

## 快速训练与评估
为避免现有 `config.yaml` 与代码期望键不一致，推荐新建一个快速验证配置 `config.quick.yaml`，内容示例如下（放置于项目根）：

```
model:
  input_dim: 4
  encoder_hidden_dim: 128
  encoder_output_dim: 64
  projection_hidden_dim: 128
  projection_output_dim: 64
  temperature: 0.07
  encoder_layers: 2
  projection_layers: 2

training:
  batch_size: 128
  epochs: 1
  learning_rate: 0.0003
  weight_decay: 0.0001

data:
  train_path: ./data/train_data.npy
  val_path: ./data/test_data.npy
  sequence_length: 10

augmentation:
  noise_std: 0.1
  dropout_prob: 0.1
  time_shift_range: 2

evaluation:
  k_neighbors: 5
  distance_metric: cosine

hardware:
  use_gpu: false
  num_workers: 0

logging:
  log_dir: ./logs
  checkpoint_dir: ./checkpoints
```

- 训练（读取 `config.quick.yaml`）：
  - `python train.py --config config.quick.yaml`
  - 训练日志写入 `./logs/train.log`，检查点保存至 `./checkpoints/latest.pth`、`./checkpoints/best.pth`。
- 评估（支持可视化）：
  - `python evaluate.py --config config.quick.yaml --checkpoint ./checkpoints/latest.pth --output_dir evaluation_results --visualize`
  - 关键输出：
    - `evaluation_results/evaluation_results.yaml`
    - `evaluation_results/visualizations/encoder_embeddings_tsne.png`
    - `evaluation_results/visualizations/encoder_embeddings_pca.png`
    - `evaluation_results/visualizations/temporal_projections_tsne.png` / `..._pca.png`
    - `evaluation_results/visualizations/spatial_projections_tsne.png` / `..._pca.png`
    - `evaluation_results/visualizations/performance_comparison.png` / `recall_at_k.png`
- 浏览器预览评估图：
  - `python -m http.server 8000`
  - 打开 `http://localhost:8000/evaluation_results/visualizations/performance_comparison.png`

## 常见问题与排查
- 中文字符显示为方块或负号异常：
  - 已在 `visualize_datasets.py` 设置 `font.sans-serif = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']` 与 `axes.unicode_minus = False`。
- CUDA 不可用或训练慢：
  - 在 `config.quick.yaml` 中将 `hardware.use_gpu: false`，并将 `batch_size` 调小、`epochs` 设为 1。
- 配置键不匹配导致 KeyError：
  - 使用本文提供的 `config.quick.yaml`，避免 `config.yaml` 与当前代码期望不一致。
- 评估报错或可视化耗时：
  - `evaluate.py` 内部对 t-SNE/PCA 默认抽样至 1000 条，确保可视化速度。如需更快可减小数据规模。

## 复现建议
- 完整训练：将 `training.epochs` 提升至 50~200，`batch_size` 视显存调整；建议优先在干净数据训练，再在带噪数据上做鲁棒性验证。
- 噪声等级对比：用多组 `--noisy_pos_std` / `--noisy_vel_std` 生成数据，分别训练并在评估中对比 `accuracy`、`mean_ap`、`recall@K` 曲线。

## 清理与维护
- 清理旧的中间文件不影响当前核心流程；如需恢复某些演示文件，请重新运行对应生成脚本（数据、可视化、评估）。
- 建议将关键输出（检查点与评估图）分目录保存，避免误删。

—— 以上步骤即可完成从数据到训练、评估与可视化的完整最小流程。若需扩展或复现论文结论，可按“复现建议”部分逐步加大规模与训练轮数。