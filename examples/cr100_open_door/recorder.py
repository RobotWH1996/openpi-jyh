"""
Data recording, saving, and visualization for CR100 policy inference runs.

Handles:
  - Accumulating per-inference observation/action records
  - Accumulating per-step published action records
  - Saving images, .npz data, and matplotlib plots on exit
"""

import os
import signal
from datetime import datetime

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


JOINT_LABELS = [
    'L_Arm_J1', 'L_Arm_J2', 'L_Arm_J3', 'L_Arm_J4', 'L_Arm_J5', 'L_Arm_J6', 'L_Arm_J7',
    'L_Hand_1', 'L_Hand_2', 'L_Hand_3', 'L_Hand_4', 'L_Hand_5', 'L_Hand_6',
    'R_Arm_J1', 'R_Arm_J2', 'R_Arm_J3', 'R_Arm_J4', 'R_Arm_J5', 'R_Arm_J6', 'R_Arm_J7',
    'R_Hand_1', 'R_Hand_2', 'R_Hand_3', 'R_Hand_4', 'R_Hand_5', 'R_Hand_6',
]

JOINT_GROUPS = [
    ("Left Arm (7 joints)",   slice(0, 7)),
    ("Left Hand (6 joints)",  slice(7, 13)),
    ("Right Arm (7 joints)",  slice(13, 20)),
    ("Right Hand (6 joints)", slice(20, 26)),
]


class InferenceRecorder:
    """Accumulates inference and publish logs, then saves them to disk with plots."""

    def __init__(self, logger=None):
        self.record_log: list[dict] = []
        self.publish_log: list[dict] = []
        self._logger = logger

    def _log(self, msg: str, level: str = "info"):
        if self._logger:
            getattr(self._logger, level)(msg)

    # ---- accumulation ----

    def add_inference(self, inference_id: int, timestamp: float, inference_time: float,
                      obs_state: np.ndarray, obs_images: dict[str, np.ndarray],
                      action_chunk: np.ndarray):
        self.record_log.append({
            "inference_id": inference_id,
            "timestamp": timestamp,
            "inference_time": inference_time,
            "obs_state": obs_state.copy(),
            "obs_images": {k: v.copy() for k, v in obs_images.items()},
            "action_chunk": action_chunk.copy(),
        })

    def add_publish(self, action: np.ndarray):
        self.publish_log.append({
            "step": len(self.publish_log),
            "timestamp": __import__("time").time(),
            "action": action.copy(),
        })

    # ---- save ----

    def save(self):
        if not self.record_log:
            self._log("No inference data recorded, skip saving.", "warn")
            return

        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self._do_save()
        finally:
            signal.signal(signal.SIGINT, original_sigint)

    def _do_save(self):
        self._log("Saving recorded data and generating plots (please wait)...")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", timestamp_str)
        os.makedirs(save_dir, exist_ok=True)

        obs_states = np.array([r["obs_state"] for r in self.record_log])
        inference_ids = np.array([r["inference_id"] for r in self.record_log])
        inference_times_arr = np.array([r["inference_time"] for r in self.record_log])
        published_actions = (
            np.array([r["action"] for r in self.publish_log])
            if self.publish_log
            else np.empty((0, 26))
        )

        self._save_images(save_dir)
        self._save_npz(save_dir, obs_states, published_actions, inference_ids, inference_times_arr)
        self._plot_obs_vs_action(save_dir, obs_states)
        self._plot_action_trajectory(save_dir, published_actions)
        self._plot_obs_trajectory(save_dir, obs_states, inference_times_arr)
        self._log(f"All plots saved to {save_dir}")

    def _save_images(self, save_dir: str):
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        camera_names_seen: set[str] = set()
        for r in self.record_log:
            inf_id = r["inference_id"]
            for cam_name, img_chw in r["obs_images"].items():
                camera_names_seen.add(cam_name)
                img_hwc = np.transpose(img_chw, (1, 2, 0))
                img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(images_dir, f"inf{inf_id:04d}_{cam_name}.jpg"),
                    img_bgr,
                )
        self._log(
            f"Images saved to {images_dir}/ "
            f"({len(self.record_log)} inferences × {len(camera_names_seen)} cameras)"
        )

    def _save_npz(self, save_dir, obs_states, published_actions, inference_ids, inference_times):
        path = os.path.join(save_dir, "data.npz")
        np.savez(
            path,
            obs_states=obs_states,
            published_actions=published_actions,
            inference_ids=inference_ids,
            inference_times=inference_times,
        )
        self._log(f"Data saved to {path}")

    # ---- plots ----

    def _plot_obs_vs_action(self, save_dir, obs_states):
        n_inf = len(obs_states)
        first_actions = []
        for r in self.record_log:
            chunk = r["action_chunk"]
            first_actions.append(chunk[0] if len(chunk.shape) > 1 else chunk)
        first_actions = np.array(first_actions)

        fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=True)
        fig.suptitle(
            f"Observation (input) vs First Action (output) — per inference  [{n_inf} inferences]",
            fontsize=14, fontweight='bold',
        )
        x = np.arange(n_inf)
        for ax, (group_name, slc) in zip(axes, JOINT_GROUPS):
            for j in range(slc.stop - slc.start):
                idx = slc.start + j
                ax.plot(x, obs_states[:, idx], '-', alpha=0.7, label=f'obs {JOINT_LABELS[idx]}')
                ax.plot(x, first_actions[:, idx], '--', alpha=0.7, label=f'act {JOINT_LABELS[idx]}')
            ax.set_ylabel("Radians")
            ax.set_title(group_name)
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            for xi in x:
                ax.axvline(xi, color='gray', alpha=0.08, linewidth=0.5)
        axes[-1].set_xlabel("Inference Step")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "obs_vs_action_per_inference.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self._log(f"Plot saved: {path}")

    def _plot_action_trajectory(self, save_dir, published_actions):
        n_pub = len(published_actions)
        if n_pub == 0:
            return

        fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=True)
        fig.suptitle(
            f"Published Action Trajectory — continuous  [{n_pub} steps]",
            fontsize=14, fontweight='bold',
        )
        x = np.arange(n_pub)

        inf_boundaries = []
        step_cursor = 0
        for r in self.record_log:
            inf_boundaries.append(step_cursor)
            chunk = r["action_chunk"]
            step_cursor += chunk.shape[0] if len(chunk.shape) > 1 else 1

        for ax, (group_name, slc) in zip(axes, JOINT_GROUPS):
            for j in range(slc.stop - slc.start):
                idx = slc.start + j
                ax.plot(x, published_actions[:, idx], '-', linewidth=1.2, label=JOINT_LABELS[idx])
            for bi, boundary in enumerate(inf_boundaries):
                if boundary < n_pub:
                    ax.axvline(boundary, color='red', alpha=0.35, linewidth=0.8,
                               linestyle='--', label='inference' if bi == 0 else None)
            ax.set_ylabel("Radians")
            ax.set_title(group_name)
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Published Action Step")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "action_trajectory.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self._log(f"Plot saved: {path}")

    def _plot_obs_trajectory(self, save_dir, obs_states, inference_times):
        n_inf = len(obs_states)
        x = np.arange(n_inf)
        fig = plt.figure(figsize=(18, 24))
        gs = GridSpec(5, 1, figure=fig, height_ratios=[3, 3, 3, 3, 1.5])
        fig.suptitle(
            f"Observation State Trajectory  [{n_inf} inferences]",
            fontsize=14, fontweight='bold',
        )
        for gi, (group_name, slc) in enumerate(JOINT_GROUPS):
            ax = fig.add_subplot(gs[gi])
            for j in range(slc.stop - slc.start):
                idx = slc.start + j
                ax.plot(x, obs_states[:, idx], '-o', markersize=2, linewidth=1.2,
                        label=JOINT_LABELS[idx])
            ax.set_ylabel("Radians")
            ax.set_title(group_name)
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        ax_time = fig.add_subplot(gs[4])
        ax_time.bar(x, inference_times * 1000, color='steelblue', alpha=0.7)
        ax_time.set_ylabel("Inference Time (ms)")
        ax_time.set_xlabel("Inference Step")
        ax_time.set_title("Per-inference Latency")
        ax_time.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "observation_trajectory.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self._log(f"Plot saved: {path}")
