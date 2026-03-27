import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Any
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    # New fields for discounted tracking
    discounted_episode_returns: float
    returned_discounted_episode_returns: float
    current_discount: float


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths (including discounted returns)."""

    def __init__(self, env, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            discounted_episode_returns=0.0,
            returned_discounted_episode_returns=0.0,
            current_discount=1.0,
        )
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[jax.Array, LogEnvState, jax.Array, bool, dict[Any, Any]]:
        
        # ---> THE CRITICAL FIX IS HERE <---
        # We unwrap the LogEnvState and pass ONLY the inner environment's state down!
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        # Standard Return Calculation
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1

        # Discounted Return Calculation
        # Add current reward * current gamma (gamma^t)
        new_discounted_episode_return = (
            state.discounted_episode_returns + reward * state.current_discount
        )
        # Prepare gamma for next step (gamma^(t+1))
        new_discount = state.current_discount * self.gamma

        state = LogEnvState(
            env_state=env_state,
            # Standard
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            # Discounted
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            returned_discounted_episode_returns=(
                state.returned_discounted_episode_returns * (1 - done)
                + new_discounted_episode_return * done
            ),
            # Reset discount to 1.0 if done, otherwise use next step's discount
            current_discount=new_discount * (1 - done) + 1.0 * done,
        )

        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_discounted_episode_returns"] = state.returned_discounted_episode_returns
        info["returned_episode"] = done
        
        return obs, state, reward, done, info