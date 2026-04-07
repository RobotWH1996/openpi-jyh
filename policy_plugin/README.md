# SARM + RA-BC Plugin

这个目录放的是 `pi0 / pi0.5` 的可选训练插件逻辑，默认关闭，不会影响历史训练和推理路径。

当前实现内容：

- `SARMRABCConfig`
  用于打开或关闭插件，并配置 `progress_path`、`head_mode`、`kappa`。
- `maybe_load_rabc_index(...)`
  从 `sarm_progress.parquet` 读取 SARM 预计算的 progress，并按 `action_horizon` 生成每个样本的 RA-BC 权重。
- `compute_weighted_mean_loss(...)`
  用 JAX 实现加权 loss 聚合，供 `scripts/train.py` 使用。

## 设计边界

- 只新增，不覆盖历史逻辑。
- 默认 `enabled=False`，旧配置不需要改。
- 目前只接入了 `JAX` 训练路径，也就是 [`scripts/train.py`](/media/yjh/新加卷/lh/code/openpi-jyh/scripts/train.py)。
- 目前只支持 LeRobot 的常规随机访问数据集，不支持当前仓库里的 RLDS loader，也不接入 PyTorch 训练脚本。

## 接入点

插件接在三个位置：

- 配置：[`src/openpi/training/config.py`](/media/yjh/新加卷/lh/code/openpi-jyh/src/openpi/training/config.py)
- 数据：[`src/openpi/training/data_loader.py`](/media/yjh/新加卷/lh/code/openpi-jyh/src/openpi/training/data_loader.py)
- 训练 loss：[`scripts/train.py`](/media/yjh/新加卷/lh/code/openpi-jyh/scripts/train.py)

`Pi0` 和 `Pi0.5` 模型主体没有被改成新的默认分支。插件只是在训练时给 batch 附加：

- `dataset_index`
- `rabc_weight`
- `rabc_delta`

然后在 loss 聚合时按权重做加权平均。

## RA-BC 公式

这里按公开文档中的 RA-BC 公式实现：

- `delta_i = phi_{t + chunk_size} - phi_t`
- `soft_i = clip((delta_i - (mu - 2 sigma)) / (4 sigma + eps), 0, 1)`
- `weight_i = 1(delta_i > kappa) + 1(0 <= delta_i <= kappa) * soft_i`

其中：

- `phi` 来自 SARM 预计算 progress
- `chunk_size` 直接复用当前策略的 `action_horizon`
- `mu / sigma` 在当前 progress 文件上统计

LeRobot 官方的 `dual` 模式虽然内部用了粗粒度/细粒度子任务标注，但导出的 `sarm_progress.parquet` 仍然是 episode 级的：

- `progress_sparse`
- `progress_dense`

也就是说，子任务结构已经被 SARM 折叠进最终 progress 里了。对 RA-BC 来说，直接使用导出的 `progress_sparse` 或 `progress_dense` 即可。

## 如何启用

最推荐的用法是：基于现有训练 config，新加一个专门给 SARM + RA-BC 用的 config。

### 步骤 1：准备 progress 文件

先准备好 `sarm_progress.parquet`。

按照 LeRobot 官方 `SARM dual` 模式，最常见也最推荐的文件就是官方直接导出的 `sarm_progress.parquet`。

LeRobot 官方导出的 `sarm_progress.parquet` 可以直接用于这版插件，不需要你额外做字段适配。

至少要有：

- `index`
- `episode_index`
- `frame_index`
- `progress_sparse`

如果训练的是 dual 模式，通常还会有：

- `progress_dense`

仓库里这版插件就是优先按这个官方格式适配的。

如果你自己额外扩展了更细的显式子任务列，也可以再有：

- `subtask_id`
- `subtask_frame_index`
- `subtask_progress_sparse` 或 `subtask_progress_dense`

推荐把这个文件放在数据集根目录下，文件名就叫：

```text
sarm_progress.parquet
```

这样可以直接走自动发现逻辑。

### 步骤 2：在训练 config 里新增一个配置

例如你现在想在 `pi05_droid` 上启用它，可以去 [`src/openpi/training/config.py`](/media/yjh/新加卷/lh/code/openpi-jyh/src/openpi/training/config.py) 里新增一个配置，写法类似这样：

```python
TrainConfig(
    name="pi05_droid_sarm_rabc",
    model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            # 如果你的 LeRobot 数据已经在本地，建议明确写上根目录
            lerobot_root="/path/to/your/lerobot/dataset",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
    num_train_steps=20_000,
    batch_size=32,
    policy_plugin=SARMRABCConfig(
        enabled=True,
        # 二选一：
        # 1. 显式指定 parquet 文件
        progress_path="/path/to/your/lerobot/dataset/sarm_progress.parquet",
        # 2. 或者设成 None，走自动发现
        # progress_path=None,
        head_mode="sparse",
        kappa=0.01,
    ),
)
```

如果你不想手动写 `progress_path`，可以：

- 把 `progress_path=None`
- 同时保证 `lerobot_root/sarm_progress.parquet` 存在

### 步骤 3：启动训练

加完 config 之后，直接运行：

```bash
python scripts/train.py pi05_droid_sarm_rabc --exp_name=exp_sarm_rabc
```

如果你是 `pi0`，做法完全一样，只需要把基础配置换成你自己的 `pi0_xxx`。

### 步骤 4：看日志确认是否生效

训练启动后，日志里应该能看到类似这些指标：

- `rabc_mean_weight`
- `rabc_delta_mean`
- `rabc_delta_std`

如果这些指标没有出现，通常说明插件没有真正打开，或者 batch 里没有成功注入 `rabc_weight`。

### 一种更轻量的启用方式

如果你是在 Python 里自己拿 `config` 再启动训练，也可以直接这样改：

在训练配置里打开 `policy_plugin.enabled`，并提供 progress 文件：

```python
from policy_plugin import SARMRABCConfig

config = dataclasses.replace(
    config,
    policy_plugin=SARMRABCConfig(
        enabled=True,
        progress_path="/path/to/sarm_progress.parquet",
        head_mode="sparse",
        kappa=0.01,
    ),
)
```

然后再把这个 `config` 传给 [`scripts/train.py`](/media/yjh/新加卷/lh/code/openpi-jyh/scripts/train.py) 里的 `main(config)`。

也可以依赖自动发现：

- 如果 `data.lerobot_root` 已设置，默认找 `<lerobot_root>/sarm_progress.parquet`
- 否则默认找 `~/.cache/huggingface/lerobot/<repo_id>/sarm_progress.parquet`

## Progress 文件要求

### 官方 LeRobot dual 格式

这是现在最应该使用的格式，也是 LeRobot SARM 文档里描述的输出。

推荐列：

- `index`
- `episode_index`
- `frame_index`
- `progress_sparse`
- `progress_dense`

一个典型示例：

| index | episode_index | frame_index | progress_sparse | progress_dense |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.00 | 0.00 |
| 1 | 0 | 1 | 0.08 | 0.03 |
| 2 | 0 | 2 | 0.18 | 0.11 |
| 3 | 0 | 3 | 0.31 | 0.22 |
| 4 | 0 | 4 | 0.47 | 0.41 |
| 5 | 0 | 5 | 0.62 | 0.58 |

这里：

- `progress_sparse` 对应高层阶段进度
- `progress_dense` 对应细粒度阶段进度

如果你想按第三种 `dual` 类型使用：

- `head_mode="sparse"` 就读 `progress_sparse`
- `head_mode="dense"` 就读 `progress_dense`

也就是说：

- 直接导出官方 `dual` 的 parquet
- 放到 `lerobot_root/sarm_progress.parquet`，或者用 `progress_path` 指过去
- 在训练配置里打开 `policy_plugin.enabled=True`

就可以直接训练，不需要先手工改 parquet 字段。

### 扩展格式

如果你自己在数据侧保留了显式子任务边界，也可以额外提供：

- `subtask_id`
- `subtask_frame_index`
- `subtask_progress_sparse`
- `subtask_progress_dense`

当前代码对这些列是兼容的，但它们不是使用 LeRobot 官方 SARM 工具的必要条件。

## 训练时可观测指标

启用后会额外记录：

- `rabc_mean_weight`
- `rabc_delta_mean`
- `rabc_delta_std`

这些指标可以帮助判断 `kappa` 是否过低或过高。

## 当前限制

- 只支持 JAX 训练入口：[`scripts/train.py`](/media/yjh/新加卷/lh/code/openpi-jyh/scripts/train.py)
- 不支持 [`scripts/train_pytorch.py`](/media/yjh/新加卷/lh/code/openpi-jyh/scripts/train_pytorch.py)
- 不支持当前仓库里的 RLDS loader
- `chunk_size` 直接等于当前模型的 `action_horizon`

## 最小可用示例

如果你只是想先确认通路通了，最小需要满足这 4 件事：

1. 你的训练 config 使用的是普通 LeRobot 数据集，不是 RLDS。
2. `policy_plugin.enabled=True`。
3. `sarm_progress.parquet` 路径能被找到。
4. 训练日志里能看到 `rabc_mean_weight`。
