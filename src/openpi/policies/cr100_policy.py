"""CR100 Open Door policy transforms.

This is intentionally written in the same style as `aloha_policy.py`:
- A small repack step produces {images, state, actions, prompt}
- Policy inputs validate cameras, normalize image dtypes/shapes, and build
  the model-standard keys: {image, image_mask, state, actions, prompt}

Dataset (as observed from local parquet):
- state dim: 26 (no gripper)
- action dim: 26 (no gripper)
- cameras: cam_high, cam_low, cam_left_wrist, cam_right_wrist
"""

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
    - state: [26]
    - actions: [action_horizon, 26] (training only)
    - prompt: optional str
    """

    model_type: _model.ModelType

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

        # Prefer wrist cameras; fall back to cam_low if a wrist camera is missing.
        extra_image_sources = {
            "left_wrist_0_rgb": ("cam_left_wrist"),
            "right_wrist_0_rgb": ("cam_right_wrist"),
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

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CR100Outputs(transforms.DataTransformFn):
    """Outputs for the CR100 policy."""

    action_dim: int = 26

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, : self.action_dim]}
