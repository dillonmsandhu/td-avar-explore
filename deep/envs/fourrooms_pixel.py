import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from typing import Any, Callable, Dict, Tuple
from .fourrooms_custom import FourRooms, EnvState, EnvParams, FourRoomsExactValue

class FourRoomsPixels(FourRooms):
    """
    A Pixel version of FourRooms. 
    Defaults to a 21x21 grid upscaled to an 84x84 single-channel image.
    """
    def __init__(self, N: int = 21, pixel_size: int = 84):
        # Initialize the underlying logical grid with use_visual_obs=True
        super().__init__(N=N, use_visual_obs=True)
        
        if pixel_size % N != 0:
            raise ValueError(f"pixel_size ({pixel_size}) must be divisible by N ({N})")
        
        self.pixel_size = pixel_size
        self.scale = pixel_size // N  # For 21 -> 84, scale is 4

    def get_obs(self, state: EnvState, params: EnvParams | None = None, key=None) -> jax.Array:
        # 1. Get the underlying 3-channel N x N grid
        grid_obs = super().get_obs(state, params, key)
        
        # 2. Collapse to 1 channel using distinct intensity values:
        # Empty = 0.0, Wall = 0.33, Goal = 0.66, Agent = 1.0
        # We use jnp.maximum so the agent (1.0) remains visible when on top of the goal (0.66)
        single_channel = jnp.maximum(
            grid_obs[..., 0] * 0.33,  # Walls
            jnp.maximum(
                grid_obs[..., 2] * 0.66,  # Goal
                grid_obs[..., 1] * 1.0    # Agent
            )
        )
        
        # 3. Upscale to 84x84 by repeating pixels
        pixel_obs = jnp.repeat(jnp.repeat(single_channel, self.scale, axis=0), self.scale, axis=1)
        
        # Add the channel dimension -> (84, 84, 1)
        return jnp.expand_dims(pixel_obs, axis=-1)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        
        # Step the underlying logical environment
        obs, next_state, reward, done, info = super().step_env(key, state, action, params)
        
        # Inject the underlying 21x21 3-channel grid into the info dictionary
        info["underlying_grid"] = super().get_obs(next_state, params)
        
        return obs, next_state, reward, done, info

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (self.pixel_size, self.pixel_size, 1), jnp.float32)

    @property
    def name(self) -> str:
        return "FourRoomsPixels-v0"


class FourRoomsPixelExactValue(FourRoomsExactValue):
    """
    Exact policy evaluation wrapper that feeds the 84x84x1 single-channel 
    observations into the neural network, while relying on the identical 
    underlying state dynamics for the exact linear solver.
    """
    def __init__(
        self,
        size: int = 21,
        pixel_size: int = 84,
        fail_prob: float = 1.0 / 3.0,
        gamma: float = 0.99,
        episodic: bool = True,
        absorbing: bool = False,
        goal_pos: Tuple[int, int] | None = None,
    ):
        if pixel_size % size != 0:
            raise ValueError(f"pixel_size ({pixel_size}) must be divisible by size ({size})")
            
        self.pixel_size = pixel_size
        self.scale = pixel_size // size
        
        # Initialize the exact evaluator standard components
        super().__init__(
            size=size,
            fail_prob=fail_prob,
            gamma=gamma,
            episodic=episodic,
            absorbing=absorbing,
            use_visual_obs=True, # Forces parent to prep for visual structures
            goal_pos=goal_pos
        )

    def _build_obs_stack(self) -> jax.Array:
        """
        Overrides the observation stack generation to map all reachable states 
        into the 84x84x1 pixel space prior to the forward pass.
        """
        # Build the standard N x N feature maps using numpy (for fast init)
        agent_maps = np.zeros((self.num_states, self.N, self.N), dtype=np.float32)
        goal_map = np.zeros((self.N, self.N), dtype=np.float32)
        goal_map[int(self.goal[0]), int(self.goal[1])] = 1.0
        wall_map = np.asarray(self.occupied_map, dtype=np.float32)
        
        for i, (y, x) in enumerate(np.asarray(self.coords)):
            agent_maps[i, int(y), int(x)] = 1.0
            
        goal_stack = np.broadcast_to(goal_map, agent_maps.shape)
        wall_stack = np.broadcast_to(wall_map, agent_maps.shape)
        
        # Collapse into 1 channel (Walls=0.33, Goal=0.66, Agent=1.0)
        single_channel = np.maximum(
            wall_stack * 0.33,
            np.maximum(goal_stack * 0.66, agent_maps * 1.0)
        )
        
        # Upscale to pixel_size x pixel_size (axis 1 and 2 represent H and W)
        pixel_stack = np.repeat(np.repeat(single_channel, self.scale, axis=1), self.scale, axis=2)
        
        # Add channel dimension (num_states, 84, 84, 1)
        pixel_stack = np.expand_dims(pixel_stack, axis=-1)
        
        return jnp.asarray(pixel_stack, dtype=jnp.float32)