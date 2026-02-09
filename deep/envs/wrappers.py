import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from typing import Any

class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)

# --- Running Mean/Std Utilities ---
@struct.dataclass
class RunningMeanStdState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float

def update_running_mean_std(state, x):
    # Inside vmap, x is a single sample, not a batch.
    # Sample mean is just x. Sample var is 0. Count is 1.
    batch_mean = x
    batch_var = jnp.zeros_like(x)
    batch_count = 1

    delta = batch_mean - state.mean
    tot_count = state.count + batch_count

    new_mean = state.mean + delta * batch_count / tot_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count
    return RunningMeanStdState(mean=new_mean, var=new_var, count=tot_count)

# --- Wrapper 1: Normalize Observations ---
@struct.dataclass
class NormalizeObsEnvState:
    mean_std: RunningMeanStdState
    env_state: Any


class AddChannelWrapper(GymnaxWrapper):
    """Add a channel dimension to observations for CNNs."""

    def observation_space(self, params):
        orig = self._env.observation_space(params)
        return spaces.Box(
            low=orig.low,
            high=orig.high,
            shape=orig.shape + (1,),
            dtype=orig.dtype,
        )

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return obs[..., None], state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return obs[..., None], state, reward, done, info

class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        # Initialize running stats
        obs_shape = self._env.observation_space(params).shape
        mean_std = RunningMeanStdState(
            mean=jnp.zeros(obs_shape),
            var=jnp.ones(obs_shape),
            count=1e-4
        )
        return self._normalize(obs, mean_std), NormalizeObsEnvState(mean_std, state)

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        # Update stats
        new_mean_std = update_running_mean_std(state.mean_std, obs)
        # Normalize
        norm_obs = self._normalize(obs, new_mean_std)
        return norm_obs, NormalizeObsEnvState(new_mean_std, env_state), reward, done, info

    def _normalize(self, obs, mean_std):
        return jnp.clip((obs - mean_std.mean) / jnp.sqrt(mean_std.var + 1e-8), -10.0, 10.0)

# --- Wrapper 2: Normalize Rewards ---
@struct.dataclass
class NormalizeRewardEnvState:
    mean_std: RunningMeanStdState
    return_val: float
    env_state: Any

class NormalizeRewardWrapper(GymnaxWrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        mean_std = RunningMeanStdState(
            mean=jnp.array(0.0), # We usually don't subtract mean for rewards, but we track variance
            var=jnp.array(1.0),
            count=1e-4
        )
        return obs, NormalizeRewardEnvState(mean_std, 0.0, state)

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        
        # 1. Update running return (discounted)
        # If done, we reset the running return to 0 (effectively handled by (1-done))
        # Note: In PPO we technically want to normalize based on return variance, not reward variance.
        new_return_val = state.return_val * self.gamma * (1 - done) + reward
        
        # 2. Update stats based on the *return*, not the immediate reward
        new_mean_std = update_running_mean_std(state.mean_std, new_return_val)
        
        # 3. Normalize Reward (scale only, do not shift mean)
        # We divide by the std of the *returns*
        norm_reward = jnp.clip(reward / jnp.sqrt(new_mean_std.var + 1e-8), -10.0, 10.0)

        return obs, NormalizeRewardEnvState(new_mean_std, new_return_val, env_state), norm_reward, done, info
    