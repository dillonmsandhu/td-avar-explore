import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct
from gymnax.environments import environment, spaces
from typing import Any, Tuple, Dict

@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    goal: jax.Array
    time: int

@struct.dataclass
class EnvParams(environment.EnvParams):
    fail_prob: float = 1.0 / 3
    resample_init_pos: bool = True
    resample_goal_pos: bool = True
    max_steps_in_episode: int = 500

def generate_four_rooms_map(N: int) -> jax.Array:
    grid = jnp.ones((N, N), dtype=jnp.bool_)
    grid = grid.at[0, :].set(False)
    grid = grid.at[-1, :].set(False)
    grid = grid.at[:, 0].set(False)
    grid = grid.at[:, -1].set(False)
    
    mid = jnp.int32(N // 2)
    grid = grid.at[mid, :].set(False)
    grid = grid.at[:, mid].set(False)
    
    q1 = jnp.int32(mid // 2)
    q3 = jnp.int32(mid + (N - mid) // 2)
    
    grid = grid.at[mid, q1].set(True)
    grid = grid.at[mid, q3].set(True)
    grid = grid.at[q1, mid].set(True)
    grid = grid.at[q3, mid].set(True)
    
    return grid

class FourRooms(environment.Environment[EnvState, EnvParams]):
    def __init__(self, N: int = 13, use_visual_obs: bool = False):
        super().__init__()
        self.N = jnp.int32(N)
        self.env_map = generate_four_rooms_map(N)
        self.occupied_map = (1 - self.env_map).astype(jnp.float32)
        
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)
        self.use_visual_obs = use_visual_obs

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=int(self.N * 4))

    def step_env(
        self, key: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        key_random, key_action = jax.random.split(key)
        
        choose_random = jax.random.uniform(key_random, ()) < params.fail_prob
        random_action = jax.random.randint(key_action, (), 0, 4, dtype=jnp.int32)
        # Ensure action is treated as int32 for indexing
        action = jax.lax.select(choose_random, random_action, jnp.int32(action))

        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state.pos)
        
        reward = jnp.all(new_pos == state.goal).astype(jnp.float32)
        state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(state, params)
        
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(self, key: jax.Array, params: EnvParams) -> Tuple[jax.Array, EnvState]:
        k1, k2 = jax.random.split(key)
        idx_goal = jax.random.randint(k1, (), 0, self.coords.shape[0], dtype=jnp.int32)
        goal = self.coords[idx_goal]
        idx_pos = jax.random.randint(k2, (), 0, self.coords.shape[0], dtype=jnp.int32)
        pos = self.coords[idx_pos]
        
        state = EnvState(pos=pos, goal=goal, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        if not self.use_visual_obs:
            return jnp.concatenate([state.pos, state.goal]).astype(jnp.float32)
        else:
            agent_map = jnp.zeros((self.N, self.N), dtype=jnp.float32).at[state.pos[0], state.pos[1]].set(1.0)
            goal_map = jnp.zeros((self.N, self.N), dtype=jnp.float32).at[state.goal[0], state.goal[1]].set(1.0)
            return jnp.stack([self.occupied_map, agent_map, goal_map], axis=-1)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        done_goal = jnp.all(state.pos == state.goal)
        return jnp.logical_or(done_goal, done_steps)

    # --- Gymnax API Requirements ---

    @property
    def name(self) -> str:
        return "FourRoomsCustom-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        if self.use_visual_obs:
            return spaces.Box(0.0, 1.0, (int(self.N), int(self.N), 3), jnp.float32)
        return spaces.Box(0.0, float(self.N), (4,), jnp.float32)