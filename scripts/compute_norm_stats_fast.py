"""Fast norm stats computation — reads parquet directly, skips video decoding.

Reproduces the exact same delta-action logic as the standard pipeline:
  action_chunk[t] = [action[t], action[t+1], ..., action[t+H-1]]
  delta: action_chunk[t] -= state[t]  (for masked dims)

Usage:
  uv run scripts/compute_norm_stats_fast.py --config-name pi05_cr100_open_door_lora
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def main(config_name: str) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    lerobot_root = Path(data_config.lerobot_root) if data_config.lerobot_root else None
    if lerobot_root is None:
        raise ValueError("lerobot_root is not set in data config")

    parquet_files = sorted(lerobot_root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {lerobot_root}/data")

    left_arm_hand_only = getattr(config.data, "left_arm_hand_only", True)
    use_delta = getattr(config.data, "use_delta_joint_actions", True)
    action_horizon = config.model.action_horizon

    dim = 13 if left_arm_hand_only else 26
    mask_len = 7

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()

    for pf in tqdm.tqdm(parquet_files, desc="Reading parquets"):
        table = pq.read_table(pf, columns=["observation.state", "action"])
        states_all = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions_all = np.array(table.column("action").to_pylist(), dtype=np.float32)

        if left_arm_hand_only:
            states_all = states_all[:, :dim]
            actions_all = actions_all[:, :dim]

        n = len(states_all)

        for t in range(n):
            state = states_all[t]
            state_stats.update(state[np.newaxis])

            end = min(t + action_horizon, n)
            action_chunk = actions_all[t:end].copy()

            if len(action_chunk) < action_horizon:
                pad = np.tile(action_chunk[-1:], (action_horizon - len(action_chunk), 1))
                action_chunk = np.concatenate([action_chunk, pad], axis=0)

            if use_delta:
                action_chunk[:, :mask_len] -= state[:mask_len]

            action_stats.update(action_chunk[np.newaxis])

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    output_dir = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)
    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)
