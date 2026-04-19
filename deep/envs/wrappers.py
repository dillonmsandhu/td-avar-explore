import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from gymnax.environments.environment import Environment, EnvState, EnvParams
from typing import Any


class UniversalObservationWrapper(GymnaxWrapper):
    """Base wrapper that syncs standard and ghost observations."""
    
    def observation(self, obs: jax.Array, env_state: EnvState, params: EnvParams) -> jax.Array:
        """Override this method to transform the observation."""
        raise NotImplementedError

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.observation(obs, state, params), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        
        # 1. Transform the main obs
        obs = self.observation(obs, env_state, params)
        
        # 2. Automatically transform the ghost obs if it exists!
        if "real_next_obs" in info:
            info["real_next_obs"] = self.observation(info["real_next_obs"], env_state, params)
            
        return obs, env_state, reward, done, info

class TerminalInfoWrapper(GymnaxWrapper):
    """Wrapper that injects the true terminal state and observation into info."""
    
    # FIX 1: Pass env_name into the initialization
    def __init__(self, env, is_goal_env=""):
        super().__init__(env)
        self.is_goal_env = is_goal_env

    def step(self, key, state, action, params=None):
        if params is None:
            params = self._env.default_params

        # 1. Split the RNG key
        key_step, key_reset = jax.random.split(key)
        
        # 2. Get the true transition (no auto-reset applied yet)
        obs_st, state_st, reward, done, info = self._env.step_env(
            key_step, state, action, params
        )
        
        # 3. Get the reset transition
        obs_re, state_re = self._env.reset_env(key_reset, params)

        # 4. Inject the true terminal observation and state into the info dict
        info["real_next_obs"] = obs_st
        info["real_next_state"] = state_st

        # Absorbing Logic: Is it a timeout?
        is_timeout = (state_st.time >= params.max_steps_in_episode)

        # 3. Build the boolean flag dynamically
        if self.is_goal_env:
            # It is a goal ONLY if it is done AND it is NOT a timeout.
            terminal = jnp.logical_and(done, jnp.logical_not(is_timeout))
            is_goal = jnp.logical_and(terminal, reward > 0)
        else:
            # For Breakout/Survival games, done is never a goal.
            # FIX 2: Name the variable correctly to match the export
            is_goal = jnp.array(False, dtype=jnp.bool_)

        info["is_goal"] = is_goal
            
        # 5. Apply the standard Gymnax auto-reset logic
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        return obs, state, reward, done, info

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


class AddChannelWrapper(UniversalObservationWrapper):
    def observation_space(self, params):
        orig = self._env.observation_space(params)
        return spaces.Box(
            low=orig.low,
            high=orig.high,
            shape=orig.shape + (1,),
            dtype=orig.dtype,
        )

    def observation(self, obs, env_state, params):
        # Only add the channel if it's missing (keeps it safe for 1D/2D)
        if obs.ndim < 3: 
            return obs[..., None]
        return obs

class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        # Initialize running stats
        obs_shape = self._env.observation_space(params).shape
        mean_std = RunningMeanStdState(
            mean=jnp.zeros(obs_shape),
            var=jnp.ones(obs_shape),
            count=1e-4
        )
        return self._normalize(obs, mean_std), NormalizeObsEnvState(mean_std, env_state)

    def step(self, key, state, action, params=None):
        # 1. Step the underlying environment
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        
        # 2. Update running statistics using the RAW observation
        new_mean_std = update_running_mean_std(state.mean_std, obs)
        
        # 3. Normalize the main observation
        norm_obs = self._normalize(obs, new_mean_std)
        
        # 4. CRITICAL FIX: Normalize the ghost observation if it exists!
        if "real_next_obs" in info:
            info["real_next_obs"] = self._normalize(info["real_next_obs"], new_mean_std)
            
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
            mean=jnp.array(0.0, dtype=jnp.float32), 
            var=jnp.array(1.0, dtype=jnp.float32),
            count=jnp.array(1e-4, dtype=jnp.float32) # Good practice to make this an array too!
        )
        initial_return_val = jnp.zeros((), dtype=jnp.float32)
        
        return obs, NormalizeRewardEnvState(mean_std, initial_return_val, state)

    def step(self, key, state, action, params=None):
            obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
            
            current_return = state.return_val * self.gamma + reward
            new_mean_std = update_running_mean_std(state.mean_std, current_return)
            
            scale = jnp.sqrt(new_mean_std.var + 1e-8)
            
            norm_reward = jnp.clip(reward / scale, -10.0, 10.0)

            # 4. Mask for next step
            new_return_val = current_return * (1.0 - done)

            return obs, NormalizeRewardEnvState(new_mean_std, new_return_val, env_state), norm_reward, done, info
    
class SubtractOneRewardWrapper(GymnaxWrapper):
    """A debugging wrapper that subtracts 1.0 from every environment reward."""
    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        # Step the inner environment
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        
        # Apply the dense penalty
        modified_reward = reward - 1.0
        
        return obs, env_state, modified_reward, done, info