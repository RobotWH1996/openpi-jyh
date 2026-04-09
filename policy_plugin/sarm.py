"""SARM + RA-BC（Reward-Aligned Behavior Cloning）训练插件。

从 LeRobot 导出的 ``sarm_progress.parquet`` 读取逐帧进度，构造样本权重与进度差分，
并向训练循环提供 :func:`compute_weighted_mean_loss` 做加权损失聚合。
"""

from __future__ import annotations

import dataclasses  # 使用 @dataclass(frozen=True) 生成不可变配置与索引对象
import logging  # 记录加载统计与缺失下标告警
import pathlib  # 解析 parquet 在磁盘上的绝对路径
from typing import Literal  # 将 head_mode 限定为 sparse | dense

import jax.numpy as jnp  # RA-BC 权重计算与损失聚合（JAX 数组）
import numpy as np  # 索引数组与 parquet 解析后的 CPU 向量
import polars as pl  # 快速读取 parquet 与列操作

logger = logging.getLogger("openpi")

# 从 parquet 读哪一列 SARM 头：稀疏或稠密进度预测
HeadMode = Literal["sparse", "dense"]


@dataclasses.dataclass(frozen=True)
class SARMRABCConfig:
    """可选 SARM + RA-BC 插件配置。

    仅影响训练：关闭时数据流与历史 pi0 / pi0.5 一致。
    默认期望 LeRobot 导出的 ``sarm_progress.parquet``（如 dual 模式的 ``progress_sparse`` / ``progress_dense``）。
    """

    enabled: bool = False  # 总开关；False 时不加载索引，损失为不加权平均
    progress_path: str | None = None  # 显式 parquet 路径；None 则按 LeRobot 目录规则推断
    head_mode: HeadMode = "sparse"  # 选用 progress_sparse 或 progress_dense 风格列名
    kappa: float = 0.01  # _compute_rabc_weights 中阈值：delta > kappa 时权重为 1.0
    epsilon: float = 1e-6  # 分母与软权重分母上的数值下界，避免除零


@dataclasses.dataclass(frozen=True)
class SARMRABCIndex:
    """按数据集全局下标预计算的权重与 delta 数组，供 RA-BC 查表。"""

    weights: np.ndarray  # 每个训练样本下标 (0..dataset_len-1) 对应一个浮点权重
    deltas: np.ndarray  # 每个下标的进度差分（用于指标与权重计算）
    mean_delta: float  # parquet 覆盖行上 delta 的均值（日志用）
    std_delta: float  # parquet 覆盖行上 delta 的标准差（日志用）

    def lookup(self, dataset_index: int) -> tuple[np.float32, np.float32]:
        """按数据集行下标返回 (weight, delta)；越界时返回中性值 (1, 0)。"""

        # 负下标或超出长度：等价于均匀权重、delta 为 0
        if dataset_index < 0 or dataset_index >= len(self.weights):
            return np.float32(1.0), np.float32(0.0)
        # 正常路径：该全局下标对应的标量权重与 delta
        return np.float32(self.weights[dataset_index]), np.float32(self.deltas[dataset_index])


def maybe_load_rabc_index(
    plugin_config: SARMRABCConfig,
    *,
    repo_id: str | None,
    lerobot_root: str | None,
    action_horizon: int,
    dataset_len: int,
) -> SARMRABCIndex | None:
    """读取 parquet，构造与数据集对齐的权重/delta 向量；插件关闭时返回 None。"""

    # 未启用：调用方走不加权训练（Observation 不含 rabc 字段）
    if not plugin_config.enabled:
        return None

    # 确定 sarm_progress.parquet 位置：显式路径、lerobot 根目录或 HF 缓存
    progress_path = _resolve_progress_path(plugin_config, repo_id=repo_id, lerobot_root=lerobot_root)
    # 读入整张表到 Polars
    frame = pl.read_parquet(progress_path)
    # 选择 progress_sparse / progress_dense（或子任务变体列）
    progress_column = _select_progress_column(frame, plugin_config.head_mode)
    # 最小列集合：index + 选中的进度列
    required_columns = {"index", progress_column}

    # 带子任务信息的 parquet 必须含 subtask_id 才能分段
    has_subtask = "subtask_id" in frame.columns
    if has_subtask:
        required_columns.add("subtask_id")

    # 列不齐则立即报错
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"SARM progress file missing columns: {sorted(missing)}")

    # 构造长度等于 dataset_len 的权重与 delta 向量，与训练集下标对齐
    weights, deltas, mean_delta, std_delta = _build_index_arrays(
        frame=frame,
        dataset_len=dataset_len,
        progress_column=progress_column,
        chunk_size=action_horizon,
        kappa=plugin_config.kappa,
        epsilon=plugin_config.epsilon,
    )

    # 单行摘要，便于实验日志查看
    logger.info(
        "Loaded SARM + RA-BC weights from %s (head=%s, progress=%s, subtask=%s, mean_weight=%.4f, delta_mean=%.4f, delta_std=%.4f)",
        progress_path,
        plugin_config.head_mode,
        progress_column,
        has_subtask,
        float(weights.mean()),
        mean_delta,
        std_delta,
    )
    # 封装为不可变快照，供 DataLoader 侧按 index 查询
    return SARMRABCIndex(weights=weights, deltas=deltas, mean_delta=mean_delta, std_delta=std_delta)


def compute_weighted_mean_loss(
    per_sample_loss: jnp.ndarray,
    sample_weights: jnp.ndarray | None,
    sample_deltas: jnp.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """用 RA-BC 权重聚合逐样本损失；可选记录本 batch 内 delta 的统计量。

    若 ``sample_weights`` 为 None，则退化为普通算术平均（与历史行为一致）。
    """

    # 本 batch 无 RA-BC：与 jnp.mean(per_sample_loss) 等价
    if sample_weights is None:
        return jnp.mean(per_sample_loss), {}

    # 与损失同 dtype，优化更稳定
    weights = jnp.asarray(sample_weights, dtype=per_sample_loss.dtype)
    # 全为 ~0 时避免分母为 0
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(epsilon, dtype=per_sample_loss.dtype))
    # 标量 batch 损失：sum_i w_i * loss_i / sum_i w_i
    weighted_loss = jnp.sum(per_sample_loss * weights) / denom

    # 监控用：始终记录本 batch 权重均值
    metrics = {
        "rabc_mean_weight": jnp.mean(weights),
    }
    # 若传入 delta：记录分布（同一函数也用于仅取指标、loss 置零的路径）
    if sample_deltas is not None:
        deltas = jnp.asarray(sample_deltas, dtype=per_sample_loss.dtype)
        metrics["rabc_delta_mean"] = jnp.mean(deltas)
        metrics["rabc_delta_std"] = jnp.std(deltas)
    return weighted_loss, metrics


def _resolve_progress_path(
    plugin_config: SARMRABCConfig,
    *,
    repo_id: str | None,
    lerobot_root: str | None,
) -> pathlib.Path:
    """根据配置与数据集标识解析 ``sarm_progress.parquet`` 的绝对路径。"""

    # 用户显式指定路径优先
    if plugin_config.progress_path is not None:
        return pathlib.Path(plugin_config.progress_path).expanduser().resolve()

    # 本地 LeRobot 数据集根：<root>/sarm_progress.parquet
    if lerobot_root is not None:
        return (pathlib.Path(lerobot_root).expanduser() / "sarm_progress.parquet").resolve()

    # 没有真实数据集 id 无法推断 HF 缓存路径
    if repo_id is None or repo_id == "fake":
        raise ValueError("SARM + RA-BC requires either progress_path or a real LeRobot dataset.")

    # 默认 HuggingFace LeRobot 缓存目录布局
    return (pathlib.Path("~/.cache/huggingface/lerobot").expanduser() / repo_id / "sarm_progress.parquet").resolve()


def _build_index_arrays(
    *,
    frame: pl.DataFrame,
    dataset_len: int,
    progress_column: str,
    chunk_size: int,
    kappa: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """对 parquet 排序、按段算 delta、算 RA-BC 权重，再 scatter 到长度为 dataset_len 的数组。"""

    # 默认：每个下标先设为权重 1、delta 0，再被 parquet 覆盖
    default_weights = np.ones((dataset_len,), dtype=np.float32)
    default_deltas = np.zeros((dataset_len,), dtype=np.float32)

    # 时间顺序：episode / 子任务 / 帧，用于 delta 计算
    sorted_frame = frame.sort(_build_sort_columns(frame))

    # 导出表中的全局数据集下标（映射回训练 DataLoader 顺序）
    global_indices = sorted_frame["index"].to_numpy().astype(np.int64, copy=False)
    # 每行 SARM 预测的进度，通常在 [0,1]
    progress = sorted_frame[progress_column].to_numpy().astype(np.float32, copy=False)
    # episode_index 与/或 subtask_id 定义 delta 的分段边界
    segment_columns = _build_segment_columns(sorted_frame)
    # 相对未来 chunk_size 步的进度变化；在段边界处截断，不跨段
    deltas = _compute_segmented_deltas(
        progress,
        [sorted_frame[name].to_numpy() for name in segment_columns],
        chunk_size,
    )

    # 将原始 delta 映射为 [0,1] 软权重，再按 kappa 分段
    weights = _compute_rabc_weights(jnp.asarray(deltas), kappa=kappa, epsilon=epsilon)
    weights = np.asarray(weights, dtype=np.float32)
    deltas = np.asarray(deltas, dtype=np.float32)

    # 只写入落在训练集合法范围内的下标
    valid_mask = (global_indices >= 0) & (global_indices < dataset_len)
    default_weights[global_indices[valid_mask]] = weights[valid_mask]
    default_deltas[global_indices[valid_mask]] = deltas[valid_mask]

    # parquet 未覆盖的下标保持默认并告警
    missing_count = dataset_len - int(valid_mask.sum())
    if missing_count > 0:
        logger.warning(
            "SARM progress file does not cover %d dataset indices. Falling back to weight=1 for those samples.",
            missing_count,
        )

    # 此处 mean/std 针对 parquet 行向量 deltas，而非整段 dataset_len 填零后全体
    return default_weights, default_deltas, float(deltas.mean()), float(deltas.std())


def _select_progress_column(frame: pl.DataFrame, head_mode: HeadMode) -> str:
    """根据实际列名选择子任务级或全局进度列。"""

    # 扩展 schema 下优先用的列名
    subtask_column = f"subtask_progress_{head_mode}"
    global_column = f"progress_{head_mode}"

    # 官方 LeRobot SARM 导出以 episode 级进度为主；仅当存在扩展列时才用显式子任务进度
    if "subtask_id" in frame.columns and subtask_column in frame.columns:
        return subtask_column
    if "subtask_id" in frame.columns and global_column in frame.columns:
        logger.warning(
            "Found subtask-aware SARM file without %s. Falling back to %s inside each subtask.",
            subtask_column,
            global_column,
        )
        return global_column
    return global_column


def _build_sort_columns(frame: pl.DataFrame) -> list[str]:
    """返回排序列名列表，使行顺序为 episode → 子任务 → 帧。"""

    sort_columns = []
    if "episode_index" in frame.columns:
        sort_columns.append("episode_index")
    if "subtask_id" in frame.columns:
        sort_columns.append("subtask_id")
    if "subtask_frame_index" in frame.columns:
        sort_columns.append("subtask_frame_index")
    elif "frame_index" in frame.columns:
        sort_columns.append("frame_index")
    sort_columns.append("index")
    return sort_columns


def _build_segment_columns(frame: pl.DataFrame) -> list[str]:
    """定义 delta 连续段的列（episode 与/或 subtask）。"""

    segment_columns = []
    if "episode_index" in frame.columns:
        segment_columns.append("episode_index")
    if "subtask_id" in frame.columns:
        segment_columns.append("subtask_id")
    return segment_columns


def _compute_segmented_deltas(
    progress: np.ndarray,
    segment_arrays: list[np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    """在每个 episode/子任务段内算进度 delta，不跨段做未来帧回看。"""

    # 无分段列：整段视为一个 segment
    if not segment_arrays:
        return _compute_global_deltas(progress, chunk_size)

    # 按连续段逐段填充 delta
    deltas = np.zeros_like(progress, dtype=np.float32)
    segment_start = 0
    while segment_start < len(progress):
        # 向右扩展直到离开同一 episode/子任务
        segment_end = segment_start + 1
        while segment_end < len(progress) and _same_segment(segment_arrays, segment_start, segment_end):
            segment_end += 1
        # 仅对本切片做回看（未来帧不超过段尾）
        deltas[segment_start:segment_end] = _compute_global_deltas(progress[segment_start:segment_end], chunk_size)
        segment_start = segment_end
    return deltas


def _same_segment(segment_arrays: list[np.ndarray], left: int, right: int) -> bool:
    """判断第 right 行是否与第 left 行属于同一 episode/子任务。"""

    return all(array[left] == array[right] for array in segment_arrays)


def _compute_global_deltas(progress: np.ndarray, chunk_size: int) -> np.ndarray:
    """单行 delta = 未来帧进度 − 当前进度（未来下标限制在段末）。"""

    episode_len = len(progress)
    # 用于差分的未来帧下标：min(t + chunk_size, 段末)
    future_indices = np.minimum(np.arange(episode_len) + chunk_size, episode_len - 1)
    # progress_未来 − progress_当前
    return progress[future_indices] - progress


def _compute_rabc_weights(deltas: jnp.ndarray, *, kappa: float, epsilon: float) -> jnp.ndarray:
    """将进度 delta 映射为样本权重：大正→1，负→0，中间用软权重。"""

    # 本 batch 内统计量，用于软权重斜坡
    mu = jnp.mean(deltas)
    sigma = jnp.std(deltas)
    # 相对 (mu - 2σ) 归一化到 [0,1] 的软权重
    soft = jnp.clip((deltas - (mu - 2.0 * sigma)) / (4.0 * sigma + epsilon), 0.0, 1.0)
    # 分段：强正进度→全权重；弱正→软权重；负→0
    return jnp.where(deltas > kappa, 1.0, jnp.where(deltas >= 0.0, soft, 0.0))
