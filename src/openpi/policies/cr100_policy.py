"""CR100 Open Door policy transforms.

This is intentionally written in the same style as `aloha_policy.py`:
- A small repack step produces {images, state, actions, prompt}
- Policy inputs validate cameras, normalize image dtypes/shapes, and build
  the model-standard keys: {image, image_mask, state, actions, prompt}

Dataset (as observed from local parquet):
- state dim: 26 (no gripper): [0:7) left arm joints, [7:13) left hand, [13:20) right arm, [20:26) right hand
- action dim: 26 (same layout as state)
- cameras: cam_high, cam_left_wrist, cam_right_wrist

Use ``left_arm_hand_only=True`` to keep only the first 13 dims (left arm + left hand).
"""

# First 13 components: left arm (7) + left hand (6). Full dual-arm layout is 26.
LEFT_ARM_HAND_DIM: int = 13
FULL_DUAL_ARM_DIM: int = 26

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CR100Inputs(transforms.DataTransformFn):
    """Inputs for the CR100 policy.

    Expected inputs (after repack):
    - images: dict[name, img] where img is [C,H,W] or [H,W,C]
    - state: [26] or [13] when ``left_arm_hand_only`` (slice applied here from 26-dim raw data)
    - actions: [action_horizon, 26] or [action_horizon, 13] (training only)
    - prompt: optional str
    """

    model_type: _model.ModelType
    # If True, keep only left arm + left hand (first ``LEFT_ARM_HAND_DIM`` components).
    # Pass from ``LeRobotCR100OpenDoorDataConfig.left_arm_hand_only`` (do not rely on a default here).
    left_arm_hand_only: bool

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Base image must exist.
        base_image = _parse_image(in_images["cam_high"])

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Map wrist cameras to model-standard keys.
        extra_image_sources = {
            "left_wrist_0_rgb": ("cam_left_wrist",),
            "right_wrist_0_rgb": ("cam_right_wrist",),
        }
        for dest, candidates in extra_image_sources.items():
            src = next((c for c in candidates if c in in_images), None)
            if src is not None:
                images[dest] = _parse_image(in_images[src])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                # Same convention as `libero_policy.py`.
                image_masks[dest] = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        state = np.asarray(data["state"])
        if self.left_arm_hand_only:
            state = state[..., :LEFT_ARM_HAND_DIM]

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.left_arm_hand_only:
                actions = actions[..., :LEFT_ARM_HAND_DIM]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CR100Outputs(transforms.DataTransformFn):
    """Outputs for the CR100 policy.

    Truncates model actions to the dataset action size: 13 if ``left_arm_hand_only``, else 26.
    Keep in sync with ``CR100Inputs.left_arm_hand_only``. Pass the same flag as in
    ``LeRobotCR100OpenDoorDataConfig`` (typically ``outputs=[CR100Outputs(left_arm_hand_only=cfg.left_arm_hand_only)]``).
    """

    left_arm_hand_only: bool

    @property
    def action_dim(self) -> int:
        return LEFT_ARM_HAND_DIM if self.left_arm_hand_only else FULL_DUAL_ARM_DIM

    def __call__(self, data: dict) -> dict:
        d = self.action_dim
        return {"actions": np.asarray(data["actions"])[:, :d]}
