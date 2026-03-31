# PyTorch 训练脚本使用指南

本文档介绍如何使用 `train_pytorch.py` 进行 PI0/PI0.5 模型的训练。

## 📋 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置参数详解](#配置参数详解)
- [训练模式](#训练模式)
- [Checkpoint 管理](#checkpoint-管理)
- [显存优化](#显存优化)
- [常见问题](#常见问题)

---

## 概述

`train_pytorch.py` 是 OpenPI 项目的 PyTorch 训练入口，支持：

- ✅ 单 GPU 训练
- ✅ 多 GPU 分布式训练 (DDP)
- ✅ 多节点训练
- ✅ 梯度检查点（节省显存）
- ✅ 混合精度训练 (bfloat16/float32)
- ✅ WandB 日志记录
- ✅ Checkpoint 断点续训

---

## 环境要求

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 16GB 显存 | 24GB+ (如 RTX 4090, A100) |
| 内存 | 32GB | 64GB+ |
| 存储 | 100GB SSD | 500GB+ NVMe SSD |

### 软件依赖

```bash
# 安装项目依赖
cd openpi
pip install -e .

# 或使用 uv
uv sync
```

---

## 快速开始

### 1. 调试模式（验证环境）

```bash
# 使用 fake 数据快速验证环境配置
python scripts/train_pytorch.py debug --exp_name test_run
```

### 2. 单 GPU 训练

```bash
# 基本训练命令
python scripts/train_pytorch.py <config_name> --exp_name <experiment_name>

# 示例：使用 aloha_sim 配置训练
python scripts/train_pytorch.py pi0_aloha_sim --exp_name my_first_run
```

### 3. 多 GPU 训练（单节点）

```bash
# 使用 torchrun 启动分布式训练
torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
    scripts/train_pytorch.py <config_name> --exp_name <experiment_name>

# 示例：2 卡训练
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_aloha_sim --exp_name ddp_run
```

### 4. 多节点训练

```bash
# 在每个节点上运行
torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_this_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name <experiment_name>
```

---

## 配置参数详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_name` | str | 必填 | 配置名称（如 `pi0_aloha_sim`, `debug`） |
| `--exp_name` | str | 必填 | 实验名称，用于命名 checkpoint 目录 |
| `--batch_size` | int | 32 | 全局批次大小 |
| `--num_train_steps` | int | 30000 | 训练步数 |
| `--num_workers` | int | 2 | 数据加载并行数 |

### 学习率配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr_schedule.warmup_steps` | int | 1000 | 学习率预热步数 |
| `--lr_schedule.peak_lr` | float | 2.5e-5 | 峰值学习率 |
| `--lr_schedule.decay_steps` | int | 30000 | 衰减总步数 |
| `--lr_schedule.decay_lr` | float | 2.5e-6 | 最终学习率 |

### 优化器配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--optimizer.b1` | float | 0.9 | Adam β1 |
| `--optimizer.b2` | float | 0.95 | Adam β2 |
| `--optimizer.eps` | float | 1e-8 | Adam epsilon |
| `--optimizer.weight_decay` | float | 1e-10 | 权重衰减 |
| `--optimizer.clip_gradient_norm` | float | 1.0 | 梯度裁剪阈值 |

### 日志与保存

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--log_interval` | int | 100 | 日志打印间隔（步） |
| `--save_interval` | int | 1000 | Checkpoint 保存间隔（步） |
| `--keep_period` | int | 5000 | 保留 checkpoint 的周期 |
| `--wandb_enabled` | bool | True | 是否启用 WandB |

### 训练控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resume` | bool | False | 从最新 checkpoint 恢复训练 |
| `--overwrite` | bool | False | 覆盖已存在的 checkpoint 目录 |
| `--pytorch_training_precision` | str | "bfloat16" | 训练精度 |
| `--pytorch_weight_path` | str | None | 预训练权重路径（微调用） |

---

## 训练模式

### 从零训练

```bash
python scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name new_model \
    --batch_size 32 \
    --num_train_steps 50000
```

### 微调（Fine-tuning）

```bash
# 从预训练模型微调
python scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name finetuned_model \
    --pytorch_weight_path /path/to/pretrained/checkpoint \
    --lr_schedule.peak_lr 1e-5 \
    --num_train_steps 10000
```

### 断点续训

```bash
# 添加 --resume 标志从最新 checkpoint 恢复
python scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name my_experiment \
    --resume
```

---

## Checkpoint 管理

### 目录结构

```
checkpoints/
└── <config_name>/
    └── <exp_name>/
        ├── wandb_id.txt          # WandB run ID
        ├── 1000/                  # Step 1000 checkpoint
        │   ├── model.safetensors # 模型权重
        │   ├── optimizer.pt      # 优化器状态
        │   ├── metadata.pt       # 训练元数据
        │   └── assets/           # 归一化统计等
        ├── 2000/
        └── ...
```

### Checkpoint 内容

| 文件 | 格式 | 说明 |
|------|------|------|
| `model.safetensors` | safetensors | 模型权重（通用格式） |
| `optimizer.pt` | PyTorch | 优化器状态 |
| `metadata.pt` | PyTorch | 训练步数、时间戳等 |
| `assets/` | 目录 | 归一化统计信息 |

---

## 显存优化

### 对于 24GB 显存 GPU（如 RTX 4090）

```bash
# 设置显存分配策略
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name optimized_run \
    --batch_size 16 \
    --pytorch_training_precision bfloat16
```

### 显存不足时的调整策略

1. **减小批次大小**：`--batch_size 8` 或更小
2. **使用 bfloat16**：`--pytorch_training_precision bfloat16`
3. **减少 workers**：`--num_workers 1`
4. **梯度检查点**：脚本会自动启用（如果模型支持）

### 多 GPU 显存分配

```bash
# 8+ GPU 时自动启用额外优化
# - TF32 加速
# - cuDNN benchmark
# - 静态计算图
```

---

## 可用配置列表

运行以下命令查看所有可用配置：

```bash
python scripts/train_pytorch.py --help
```

常用配置：

| 配置名 | 说明 |
|--------|------|
| `debug` | 调试配置（fake 数据，快速验证） |
| `debug_pi05` | PI0.5 调试配置 |
| `pi0_aloha` | ALOHA 机器人推理配置 |
| `pi0_aloha_sim` | ALOHA 仿真训练配置 |
| `pi05_aloha` | PI0.5 ALOHA 配置 |
| `pi0_libero` | Libero 数据集微调配置 |
| `pi0_droid` | DROID 数据集配置 |

---

## 自定义数据集训练

### 1. 计算归一化统计

```bash
python scripts/compute_norm_stats.py <your_config_name>
```

### 2. 创建配置

在 `src/openpi/training/config.py` 中添加新配置：

```python
TrainConfig(
    name="my_custom_dataset",
    model=pi0_config.Pi0Config(),
    data=LeRobotAlohaDataConfig(
        repo_id="your-username/your-dataset",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=30_000,
)
```

### 3. 开始训练

```bash
python scripts/train_pytorch.py my_custom_dataset --exp_name first_run
```

---

## 常见问题

### Q1: 出现 CUDA Out of Memory 错误

```bash
# 解决方案
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_pytorch.py ...

# 或减小 batch_size
python scripts/train_pytorch.py ... --batch_size 8
```

### Q2: 训练速度慢

```bash
# 增加 data loader workers
python scripts/train_pytorch.py ... --num_workers 4

# 确保使用 bfloat16
python scripts/train_pytorch.py ... --pytorch_training_precision bfloat16
```

### Q3: WandB 连接问题

```bash
# 禁用 WandB
python scripts/train_pytorch.py ... --wandb_enabled False

# 或设置离线模式
WANDB_MODE=offline python scripts/train_pytorch.py ...
```

### Q4: 如何查看训练进度？

- **终端**：进度条显示 loss、lr、step
- **WandB**：访问 https://wandb.ai 查看详细图表
- **日志**：查看 checkpoint 目录中的日志

### Q5: DDP 训练时遇到 NCCL 错误

```bash
# 设置调试信息
NCCL_DEBUG=INFO torchrun ...

# 或尝试 gloo 后端（CPU 兼容）
# 代码会自动检测并选择合适的后端
```

---

## 完整训练示例

```bash
# RTX 4090 单卡完整训练命令
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name aloha_sim_v1 \
    --batch_size 16 \
    --num_train_steps 30000 \
    --lr_schedule.peak_lr 2.5e-5 \
    --lr_schedule.warmup_steps 1000 \
    --save_interval 2000 \
    --log_interval 50 \
    --pytorch_training_precision bfloat16 \
    --num_workers 4
```

---

## 参考资源

- [OpenPI 官方文档](https://github.com/Physical-Intelligence/openpi)
- [PI0 论文](https://arxiv.org/abs/xxx)
- [WandB 文档](https://docs.wandb.ai/)
- [PyTorch DDP 教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

