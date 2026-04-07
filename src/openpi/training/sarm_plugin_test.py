import jax.numpy as jnp
import numpy as np
import polars as pl

from policy_plugin.sarm import SARMRABCConfig
from policy_plugin.sarm import compute_weighted_mean_loss
from policy_plugin.sarm import maybe_load_rabc_index


def test_compute_weighted_mean_loss():
    loss, metrics = compute_weighted_mean_loss(
        jnp.asarray([1.0, 3.0, 5.0]),
        jnp.asarray([1.0, 0.0, 1.0]),
        jnp.asarray([0.2, -0.5, 0.7]),
    )

    np.testing.assert_allclose(np.asarray(loss), 3.0)
    np.testing.assert_allclose(np.asarray(metrics["rabc_mean_weight"]), 2.0 / 3.0)
    np.testing.assert_allclose(np.asarray(metrics["rabc_delta_mean"]), (0.2 - 0.5 + 0.7) / 3.0)


def test_maybe_load_rabc_index_respects_episode_boundaries(tmp_path):
    progress_path = tmp_path / "sarm_progress.parquet"
    pl.DataFrame(
        {
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 0, 1, 1, 1],
            "frame_index": [0, 1, 2, 0, 1, 2],
            "progress_sparse": [0.0, 0.4, 1.0, 0.0, 0.2, 1.0],
        }
    ).write_parquet(progress_path)

    rabc_index = maybe_load_rabc_index(
        SARMRABCConfig(enabled=True, progress_path=str(progress_path), kappa=0.5),
        repo_id="dummy/repo",
        lerobot_root=None,
        action_horizon=2,
        dataset_len=6,
    )

    assert rabc_index is not None
    np.testing.assert_allclose(rabc_index.deltas, [1.0, 0.6, 0.0, 1.0, 0.8, 0.0], rtol=1e-6, atol=1e-6)
    assert rabc_index.weights[0] == 1.0
    assert rabc_index.weights[1] == 1.0
    assert rabc_index.weights[3] == 1.0
    assert rabc_index.weights[4] == 1.0
    assert 0.0 <= rabc_index.weights[2] <= 1.0
    assert 0.0 <= rabc_index.weights[5] <= 1.0


def test_maybe_load_rabc_index_respects_subtask_boundaries_and_subtask_progress(tmp_path):
    progress_path = tmp_path / "sarm_progress.parquet"
    pl.DataFrame(
        {
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3, 4, 5],
            "subtask_id": [0, 0, 0, 1, 1, 1],
            "subtask_frame_index": [0, 1, 2, 0, 1, 2],
            "progress_sparse": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "subtask_progress_sparse": [0.0, 0.4, 1.0, 0.0, 0.5, 1.0],
        }
    ).write_parquet(progress_path)

    rabc_index = maybe_load_rabc_index(
        SARMRABCConfig(enabled=True, progress_path=str(progress_path), kappa=0.5),
        repo_id="dummy/repo",
        lerobot_root=None,
        action_horizon=2,
        dataset_len=6,
    )

    assert rabc_index is not None
    np.testing.assert_allclose(rabc_index.deltas, [1.0, 0.6, 0.0, 1.0, 0.5, 0.0], rtol=1e-6, atol=1e-6)
    assert rabc_index.weights[0] == 1.0
    assert rabc_index.weights[1] == 1.0
    assert rabc_index.weights[3] == 1.0
    assert 0.0 <= rabc_index.weights[2] <= 1.0
    assert 0.0 <= rabc_index.weights[4] <= 1.0
    assert 0.0 <= rabc_index.weights[5] <= 1.0
