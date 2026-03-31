import logging
from threading import Lock
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control (NumPy version).

    Operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity
    """

    def __init__(self, rtc_enabled: bool = True, blend_steps: int = 0):
        self.queue: Optional[np.ndarray] = None
        self.original_queue: Optional[np.ndarray] = None
        self.lock = Lock()
        self.last_index = 0
        self.rtc_enabled = rtc_enabled
        self.blend_steps = blend_steps

    def get(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        return len(self.queue) - self.last_index

    def empty(self) -> bool:
        if self.queue is None:
            return True
        return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        return self.last_index

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def get_left_over(self) -> Optional[np.ndarray]:
        """Get leftover original actions for RTC prev_chunk_left_over."""
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index:]

    def merge(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        estimated_delay: int,
        real_delay: Optional[int] = None,
        action_index_before_inference: Optional[int] = None,
    ):
        real_delay_val = None
        truncate_delay = None
        with self.lock:
            if self.rtc_enabled:
                if real_delay is not None:
                    real_delay_val = real_delay
                elif action_index_before_inference is not None:
                    real_delay_val = self.get_action_index() - action_index_before_inference
                else:
                    real_delay_val = 0
                truncate_delay = real_delay_val
                self._replace_actions_queue(
                    new_original_actions,
                    new_processed_actions,
                    truncate_delay,
                )
            else:
                self._append_actions_queue(new_original_actions, new_processed_actions)

        if real_delay_val is not None:
            logger.info(
                f"RTC: Truncate at {truncate_delay}, "
                f"estimated={estimated_delay}, real={real_delay_val}"
            )

    def _replace_actions_queue(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        truncate_delay: int,
    ):
        truncate_idx = max(0, min(truncate_delay, len(new_original_actions)))

        if self.queue is not None and self.last_index > 0:
            if truncate_idx < len(new_processed_actions):
                next_action = new_processed_actions[truncate_idx]
                if self.last_index < len(self.queue):
                    old_action_at_same_pos = self.queue[self.last_index]
                    diff_aligned = np.abs(next_action - old_action_at_same_pos)
                    logger.info(
                        f"RTC Merge: diff(new[{truncate_idx}] vs old[{self.last_index}]) "
                        f"max={np.max(diff_aligned):.4f} (dim {np.argmax(diff_aligned)}), "
                        f"mean={np.mean(diff_aligned):.4f}"
                    )

                last_executed_idx = self.last_index - 1
                if last_executed_idx >= 0 and last_executed_idx < len(self.queue):
                    last_action = self.queue[last_executed_idx]
                    diff_prev = np.abs(next_action - last_action)
                    logger.info(
                        f"RTC Merge: diff(new[{truncate_idx}] vs old[{last_executed_idx}]) "
                        f"max={np.max(diff_prev):.4f} (actual jump)"
                    )

        new_original = new_original_actions[truncate_idx:].copy()
        new_processed = new_processed_actions[truncate_idx:].copy()

        if (
            self.blend_steps > 0
            and self.queue is not None
            and self.last_index < len(self.queue)
        ):
            old_remaining = self.queue[self.last_index:]
            blend_len = min(self.blend_steps, len(old_remaining), len(new_processed))
            if blend_len > 0:
                for i in range(blend_len):
                    alpha = (i + 1) / (blend_len + 1)
                    new_processed[i] = (1 - alpha) * old_remaining[i] + alpha * new_processed[i]
                    new_original[i] = (1 - alpha) * self.original_queue[
                        self.last_index + i
                    ] + alpha * new_original[i]

        self.original_queue = new_original
        self.queue = new_processed
        self.last_index = 0

    def _append_actions_queue(
        self, new_original_actions: np.ndarray, new_processed_actions: np.ndarray
    ):
        if self.queue is None:
            self.original_queue = new_original_actions.copy()
            self.queue = new_processed_actions.copy()
            return

        self.original_queue = self.original_queue[self.last_index:]
        self.queue = self.queue[self.last_index:]
        self.original_queue = np.concatenate([self.original_queue, new_original_actions])
        self.queue = np.concatenate([self.queue, new_processed_actions])
        self.last_index = 0
