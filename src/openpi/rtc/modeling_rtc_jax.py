import math
from typing import Callable

import jax
import jax.numpy as jnp
from openpi.rtc.configuration_rtc import RTCAttentionSchedule
from openpi.rtc.configuration_rtc import RTCConfig
from openpi.rtc.debug_tracker_jax import TrackerJax


class RTCProcessorJax:
    """JAX implementation of Real-Time Chunking processor."""

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config
        self.tracker = None
        if rtc_config.debug:
            self.tracker = TrackerJax(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    def track(
        self,
        time: float | jax.Array,
        x_t: jax.Array | None = None,
        v_t: jax.Array | None = None,
        x1_t: jax.Array | None = None,
        correction: jax.Array | None = None,
        err: jax.Array | None = None,
        weights: jax.Array | None = None,
        guidance_weight: float | jax.Array | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        if self.tracker is None or not self.tracker.enabled:
            return

        def _track_callback(
            time, x_t, v_t, x1_t, correction, err, weights, guidance_weight
        ):
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

        def _ensure_array(x):
            return x if x is not None else jnp.array(0.0)

        jax.debug.callback(
            _track_callback,
            time,
            _ensure_array(x_t),
            _ensure_array(v_t),
            _ensure_array(x1_t),
            _ensure_array(correction),
            _ensure_array(err),
            _ensure_array(weights),
            _ensure_array(guidance_weight),
        )

    def get_all_debug_steps(self) -> list:
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def get_prefix_weights(
        self, start: int | jax.Array, end: int | jax.Array, total: int
    ) -> jax.Array:
        """Generates prefix weights for RTC guidance.

        Region layout: [0, start) -> weight=1.0 (frozen),
                        [start, end) -> decaying (intermediate),
                        [end, total) -> weight=0.0 (fresh).
        """
        start = jnp.minimum(start, end)
        idx = jnp.arange(total)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = jnp.zeros(total)
            weights = jnp.where(idx < start, 1.0, weights)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = jnp.ones(total)
            weights = jnp.where(idx >= end, 0.0, weights)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            denom = jnp.maximum(end - start, 1e-6)
            linear_decay = 1.0 - (idx - start) / denom
            weights = jnp.where(idx < start, 1.0, linear_decay)
            weights = jnp.where(idx >= end, 0.0, weights)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            denom = jnp.maximum(end - start, 1e-6)
            linear_decay = 1.0 - (idx - start) / denom
            linear_decay = jnp.clip(linear_decay, 0.0, 1.0)
            exp_weights = linear_decay * jnp.expm1(linear_decay) / (math.e - 1)
            weights = jnp.where(idx < start, 1.0, exp_weights)
            weights = jnp.where(idx >= end, 0.0, weights)
        else:
            weights = jnp.zeros(total)

        return weights

    def compute_guidance(
        self,
        x_t: jax.Array,
        time: float | jax.Array,
        prev_chunk_left_over: jax.Array | None,
        inference_delay: int | None,
        execution_horizon: int | None,
        model_fn: Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        """Computes the guided velocity v_t using RTC."""
        tau = jnp.asarray(1.0 - time)

        if prev_chunk_left_over is None:
            v_t = model_fn(x_t)
            return v_t

        squeezed = False
        if x_t.ndim < 3:
            x_t = jnp.expand_dims(x_t, axis=0)
            squeezed = True

        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = jnp.expand_dims(prev_chunk_left_over, axis=0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        batch_size, action_chunk_size, action_dim = x_t.shape

        execution_horizon = jnp.minimum(
            execution_horizon, prev_chunk_left_over.shape[1]
        )

        if (
            prev_chunk_left_over.shape[1] < action_chunk_size
            or prev_chunk_left_over.shape[2] < action_dim
        ):
            padded = jnp.zeros(
                (batch_size, action_chunk_size, action_dim),
                dtype=prev_chunk_left_over.dtype,
            )
            padded = padded.at[
                :, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]
            ].set(prev_chunk_left_over)
            prev_chunk_left_over = padded

        weights_1d = self.get_prefix_weights(
            inference_delay, execution_horizon, action_chunk_size
        )
        weights = weights_1d[None, :, None]  # (1, T, 1)

        def forward_fn(x):
            v = model_fn(x)
            x1 = x - time * v
            return x1, v

        (x1_val, v_val), vjp_fn = jax.vjp(forward_fn, x_t)

        err = (prev_chunk_left_over - x1_val) * weights
        correction = vjp_fn((err, jnp.zeros_like(v_val)))[0]

        max_guidance_weight = jnp.asarray(self.rtc_config.max_guidance_weight)

        squared_one_minus_tau = (1 - tau) ** 2
        inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau

        c = jnp.nan_to_num(
            (1 - tau) / tau, nan=max_guidance_weight, posinf=max_guidance_weight
        )

        guidance_weight = c * inv_r2
        guidance_weight = jnp.nan_to_num(
            guidance_weight, nan=0.0, posinf=max_guidance_weight
        )
        guidance_weight = jnp.minimum(guidance_weight, max_guidance_weight)

        v_t_guided = v_val - guidance_weight * correction

        if squeezed:
            v_t_guided = jnp.squeeze(v_t_guided, axis=0)
            x1_val = jnp.squeeze(x1_val, axis=0)
            correction = jnp.squeeze(correction, axis=0)
            err = jnp.squeeze(err, axis=0)

        self.track(
            time=time,
            x1_t=x1_val,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return v_t_guided
