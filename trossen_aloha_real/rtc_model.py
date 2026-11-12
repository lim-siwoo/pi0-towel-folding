"""Training-free RTC guidance for the pinned JAX π0 model.

Copied from /home/dit/control_center/pi0_rtc_model.py (2026-11-04). Only change: the
pi0_execution import was replaced by the same constants. Validated there against
openpi commit 215abfb, which is the commit of ~/pi0_aloha_towel/openpi.

Algorithm: Black, Galliker & Levine, arXiv:2506.07339, equations 2--5.
Reference: Physical-Intelligence/real-time-chunking-kinetix/src/model.py.
This implementation uses OpenPI's reversed time convention (1=noise, 0=action).
No model weights or upstream source files are modified.
"""
from __future__ import annotations

import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
# Was imported from control_center/pi0_execution.py (same values).
PREDICTION_STEPS = 50
RTC_DENOISING_STEPS = 10
RTC_MAX_GUIDANCE_WEIGHT = 5.0


def prefix_weights(delay, length, horizon=PREDICTION_STEPS):
    start = jnp.minimum(delay, length)
    index = jnp.arange(horizon)
    linear = jnp.clip((length - index) / (length - start + 1), 0, 1)
    soft = linear * jnp.expm1(linear) / jnp.expm1(jnp.array(1.0))
    return jnp.where(index < start, 1.0, jnp.where(index < length, soft, 0.0))


def guided_velocity(velocity_fn, x, time_value, previous, weights,
                    max_weight=RTC_MAX_GUIDANCE_WEIGHT):
    """The sign changes because integration runs from t=1 down to t=0."""
    def denoise(value):
        velocity = velocity_fn(value)
        return value - time_value * velocity, velocity

    clean, pullback, velocity = jax.vjp(denoise, x, has_aux=True)
    correction = pullback((previous - clean) * weights[None, :, None])[0]
    denominator = jnp.maximum(time_value * (1 - time_value), 1e-8)
    weight = jnp.minimum(
        (time_value**2 + (1 - time_value)**2) / denominator, max_weight)
    return velocity - weight * correction


class RtcSampler(nnx.Module):
    def __init__(self, model):
        self.model = model

    def sample_actions(self, rng, observation, previous, delay, prefix_length,
                       *, noise=None):
        model = self.model
        observation = model_lib.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch, model.action_horizon, model.action_dim))
        prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
        _, cache = model.PaliGemma.llm(
            [prefix, None], mask=make_attn_mask(prefix_mask, prefix_ar),
            positions=jnp.cumsum(prefix_mask, axis=1) - 1)
        weights = prefix_weights(delay, prefix_length, model.action_horizon)
        dt = -1.0 / RTC_DENOISING_STEPS

        def step(carry, _):
            x, t = carry

            def velocity(value):
                suffix, mask, ar, adarms = model.embed_suffix(
                    observation, value, jnp.broadcast_to(t, batch))
                prefix_attention = jnp.broadcast_to(
                    prefix_mask[:, None, :],
                    (batch, suffix.shape[1], prefix.shape[1]))
                attention = jnp.concatenate(
                    [prefix_attention, make_attn_mask(mask, ar)], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
                (_, output), _ = model.PaliGemma.llm(
                    [None, suffix], mask=attention, positions=positions,
                    kv_cache=cache, adarms_cond=[None, adarms])
                return model.action_out_proj(output[:, -model.action_horizon:])

            # The released expert emits bfloat16, whereas the VJP correction
            # is float32. Match branch dtypes without changing the released
            # unguided dt*v rounding (covered by the same-noise reference test).
            delta = jax.lax.cond(
                prefix_length > 0,
                lambda _:dt*guided_velocity(velocity, x, t, previous, weights),
                lambda _:(dt*velocity(x)).astype(x.dtype), operand=None)
            return (x + delta, t + dt), None

        (actions, _), _ = jax.lax.scan(
            step, (noise, jnp.array(1.0, dtype=noise.dtype)),
            None, length=RTC_DENOISING_STEPS)
        return actions


class RtcPolicy:
    def __init__(self, policy):
        if policy._is_pytorch_model:
            raise ValueError("This RTC sampler requires the pinned JAX checkpoint.")
        self.policy = policy
        self.sampler = RtcSampler(policy._model)
        self.sample = nnx_utils.module_jit(self.sampler.sample_actions)
        self.rng = policy._rng

    def prepare_inputs(self, observation, previous):
        # Previous actions are ABSOLUTE ALOHA outputs, not stale normalized
        # deltas. Apply the exact released input pipeline again, relative to
        # the new observation: ALOHA conversion -> delta -> norm -> padding.
        inputs = self.policy._input_transform({
            **jax.tree.map(lambda x:np.array(x, copy=True) if isinstance(x, np.ndarray) else x,
                           observation),
            "actions":np.asarray(previous, dtype=np.float32).copy(),
        })
        previous_model = np.asarray(inputs.pop("actions"), dtype=np.float32)
        if previous_model.shape != (PREDICTION_STEPS, self.policy._model.action_dim):
            raise ValueError("RTC prefix has the wrong normalized model-space shape.")
        batch = jax.tree.map(lambda x:jnp.asarray(x)[None, ...], inputs)
        return batch, previous_model

    def infer(self, observation, previous, delay, prefix_length, *, noise=None):
        inputs, previous_model = self.prepare_inputs(observation, previous)
        self.rng, rng = jax.random.split(self.rng)
        started = time.monotonic()
        generated = self.sample(
            rng, model_lib.Observation.from_dict(inputs),
            jnp.asarray(previous_model)[None],
            jnp.asarray(delay, dtype=jnp.int32),
            jnp.asarray(prefix_length, dtype=jnp.int32),
            noise=None if noise is None else jnp.asarray(noise)[None])
        # Materialize before timing/IPC: asynchronous GPU launch is not latency.
        outputs = self.policy._output_transform({
            "state":np.asarray(inputs["state"][0]),
            "actions":np.asarray(generated[0]),
        })
        return {
            **outputs, "normalized_prefix":previous_model,
            "model_ms":(time.monotonic() - started) * 1000,
        }
