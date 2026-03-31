import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
# ===== [RTC] 新增 import =====
import openpi.rtc.configuration_rtc as _rtc_config
from openpi.rtc.modeling_rtc_jax import RTCProcessorJax
# ===== [RTC] 新增 import 结束 =====
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # ===== [RTC] 新增字段 =====
    rtc_config: _rtc_config.RTCConfig | None = None
    # ===== [RTC] 新增字段结束 =====

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        rtc_cfg = self.rtc_config or _rtc_config.RTCConfig()
        rtc_processor = RTCProcessorJax(rtc_cfg)
        return Pi0(self, rtc_processor=rtc_processor, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    # def get_freeze_filter(self) -> nnx.filterlib.Filter:
    #     """Returns the freeze filter based on the model config."""
    #     filters = []
    #     has_lora = False
    #     gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
    #     action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
    #     if "lora" in self.paligemma_variant:
    #         filters.append(
    #             gemma_params_filter,
    #         )
    #         if "lora" not in self.action_expert_variant:
    #             # If only freeze gemma params, exclude action expert params.
    #             filters.append(
    #                 nnx.Not(action_expert_params_filter),
    #             )
    #         has_lora = True
    #     elif "lora" in self.action_expert_variant:
    #         filters.append(
    #             action_expert_params_filter,
    #         )
    #         has_lora = True

    #     if has_lora:
    #         # If any lora is used, exclude all lora params.
    #         filters.append(
    #             nnx.Not(nnx_utils.PathRegex(".*lora.*")),
    #         )
    #     if not filters:
    #         return nnx.Nothing
    #     return nnx.All(*filters)

    def get_freeze_filter(
        self,
        *,
        freeze_all_non_lora: bool = False,
        trainable_projection: bool = False,
    ) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.

        Args:
            freeze_all_non_lora: If True, freeze ALL non-LoRA parameters (VLM, SigLIP, projections,
                and non-LoRA action expert params). Only LoRA parameters remain trainable.
                If False (default), use the original per-component freeze logic.
            trainable_projection: Only used when freeze_all_non_lora=True. If True, also keep
                projection layers trainable (action_in_proj, state_proj, action_time_mlp_*,
                action_out_proj) in addition to LoRA params.
        """
        if freeze_all_non_lora:
            has_lora = "lora" in self.paligemma_variant or "lora" in self.action_expert_variant
            if not has_lora:
                raise ValueError("freeze_all_non_lora=True requires at least one LoRA variant")
            if trainable_projection:
                lora_filter = nnx_utils.PathRegex(".*lora.*")
                llm_filter = nnx_utils.PathRegex(".*llm.*")
                img_filter = nnx_utils.PathRegex(".*img.*")
                return nnx.All(
                    nnx.Any(llm_filter, img_filter),
                    nnx.Not(lora_filter),
                )
            return nnx.Not(nnx_utils.PathRegex(".*lora.*"))

        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)