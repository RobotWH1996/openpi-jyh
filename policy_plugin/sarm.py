from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Literal

import jax.numpy as jnp
import numpy as np
import polars as pl

logger = logging.getLogger("openpi")

HeadMode = Literal["sparse", "dense"]


@dataclasses.dataclass(frozen=True)
class SARMRABCConfig:
    """Optional SARM + RA-BC plugin configuration.

        The plugin is training-only. When disabled, the pipeline behaves exactly like
        the historical pi0 / pi0.5 training path.

    By default this plugin expects the official LeRobot-exported
    `sarm_progress.parquet`, especially the `dual` mode outputs
    (`progress_sparse` / `progress_dense`).
    """

    enabled: bool = False
    progress_path: str | None = None
    head_mode: HeadMode = "sparse"
    kappa: float = 0.01
    epsilon: float = 1e-6


@dataclasses.dataclass(frozen=True)
class SARMRABCIndex:
    weights: np.ndarray
    deltas: np.ndarray
    mean_delta: float
    std_delta: float

    def lookup(self, dataset_index: int) -> tuple[np.float32, np.float32]:
        if dataset_index < 0 or dataset_index >= len(self.weights):
            return np.float32(1.0), np.float32(0.0)
        return np.float32(self.weights[dataset_index]), np.float32(self.deltas[dataset_index])


def maybe_load_rabc_index(
    plugin_config: SARMRABCConfig,
    *,
    repo_id: str | None,
    lerobot_root: str | None,
    action_horizon: int,
    dataset_len: int,
) -> SARMRABCIndex | None:
    if not plugin_config.enabled:
        return None

    progress_path = _resolve_progress_path(plugin_config, repo_id=repo_id, lerobot_root=lerobot_root)
    frame = pl.read_parquet(progress_path)
    progress_column = _select_progress_column(frame, plugin_config.head_mode)
    required_columns = {"index", progress_column}

    has_subtask = "subtask_id" in frame.columns
    if has_subtask:
        required_columns.add("subtask_id")

    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"SARM progress file missing columns: {sorted(missing)}")

    weights, deltas, mean_delta, std_delta = _build_index_arrays(
        frame=frame,
        dataset_len=dataset_len,
        progress_column=progress_column,
        chunk_size=action_horizon,
        kappa=plugin_config.kappa,
        epsilon=plugin_config.epsilon,
    )

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
    return SARMRABCIndex(weights=weights, deltas=deltas, mean_delta=mean_delta, std_delta=std_delta)


def compute_weighted_mean_loss(
    per_sample_loss: jnp.ndarray,
    sample_weights: jnp.ndarray | None,
    sample_deltas: jnp.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    if sample_weights is None:
        return jnp.mean(per_sample_loss), {}

    weights = jnp.asarray(sample_weights, dtype=per_sample_loss.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(epsilon, dtype=per_sample_loss.dtype))
    weighted_loss = jnp.sum(per_sample_loss * weights) / denom

    metrics = {
        "rabc_mean_weight": jnp.mean(weights),
    }
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
    if plugin_config.progress_path is not None:
        return pathlib.Path(plugin_config.progress_path).expanduser().resolve()

    if lerobot_root is not None:
        return (pathlib.Path(lerobot_root).expanduser() / "sarm_progress.parquet").resolve()

    if repo_id is None or repo_id == "fake":
        raise ValueError("SARM + RA-BC requires either progress_path or a real LeRobot dataset.")

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
    default_weights = np.ones((dataset_len,), dtype=np.float32)
    default_deltas = np.zeros((dataset_len,), dtype=np.float32)

    sorted_frame = frame.sort(_build_sort_columns(frame))

    global_indices = sorted_frame["index"].to_numpy().astype(np.int64, copy=False)
    progress = sorted_frame[progress_column].to_numpy().astype(np.float32, copy=False)
    segment_columns = _build_segment_columns(sorted_frame)
    deltas = _compute_segmented_deltas(
        progress,
        [sorted_frame[name].to_numpy() for name in segment_columns],
        chunk_size,
    )

    weights = _compute_rabc_weights(jnp.asarray(deltas), kappa=kappa, epsilon=epsilon)
    weights = np.asarray(weights, dtype=np.float32)
    deltas = np.asarray(deltas, dtype=np.float32)

    valid_mask = (global_indices >= 0) & (global_indices < dataset_len)
    default_weights[global_indices[valid_mask]] = weights[valid_mask]
    default_deltas[global_indices[valid_mask]] = deltas[valid_mask]

    missing_count = dataset_len - int(valid_mask.sum())
    if missing_count > 0:
        logger.warning(
            "SARM progress file does not cover %d dataset indices. Falling back to weight=1 for those samples.",
            missing_count,
        )

    return default_weights, default_deltas, float(deltas.mean()), float(deltas.std())


def _select_progress_column(frame: pl.DataFrame, head_mode: HeadMode) -> str:
    subtask_column = f"subtask_progress_{head_mode}"
    global_column = f"progress_{head_mode}"

    # Official LeRobot SARM export uses episode-level progress columns. Keep those as
    # the default path and only consume explicit subtask progress when users provide an
    # extended parquet schema.
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
    if not segment_arrays:
        return _compute_global_deltas(progress, chunk_size)

    deltas = np.zeros_like(progress, dtype=np.float32)
    segment_start = 0
    while segment_start < len(progress):
        segment_end = segment_start + 1
        while segment_end < len(progress) and _same_segment(segment_arrays, segment_start, segment_end):
            segment_end += 1
        deltas[segment_start:segment_end] = _compute_global_deltas(progress[segment_start:segment_end], chunk_size)
        segment_start = segment_end
    return deltas


def _same_segment(segment_arrays: list[np.ndarray], left: int, right: int) -> bool:
    return all(array[left] == array[right] for array in segment_arrays)


def _compute_global_deltas(progress: np.ndarray, chunk_size: int) -> np.ndarray:
    episode_len = len(progress)
    future_indices = np.minimum(np.arange(episode_len) + chunk_size, episode_len - 1)
    return progress[future_indices] - progress


def _compute_rabc_weights(deltas: jnp.ndarray, *, kappa: float, epsilon: float) -> jnp.ndarray:
    mu = jnp.mean(deltas)
    sigma = jnp.std(deltas)
    soft = jnp.clip((deltas - (mu - 2.0 * sigma)) / (4.0 * sigma + epsilon), 0.0, 1.0)
    return jnp.where(deltas > kappa, 1.0, jnp.where(deltas >= 0.0, soft, 0.0))
