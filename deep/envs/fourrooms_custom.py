from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    goal: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    fail_prob: float = 0.0
    resample_init_pos: bool = False
    resample_goal_pos: bool = False
    max_steps_in_episode: int = 1e5


def generate_four_rooms_map(size: int) -> jax.Array:
    """Build a standard Four Rooms free-space mask (True = free, False = wall)."""
    if size < 7:
        raise ValueError(f"Four Rooms size must be >= 7, got {size}.")
    if size % 2 == 0:
        raise ValueError(f"Four Rooms size must be odd, got {size}.")

    grid = jnp.ones((size, size), dtype=jnp.bool_)

    # Outer walls.
    grid = grid.at[0, :].set(False)
    grid = grid.at[-1, :].set(False)
    grid = grid.at[:, 0].set(False)
    grid = grid.at[:, -1].set(False)

    # Cross walls.
    mid = size // 2
    grid = grid.at[mid, :].set(False)
    grid = grid.at[:, mid].set(False)

    # Doorways.
    q1 = mid // 2
    q3 = mid + (size - mid) // 2
    grid = grid.at[mid, q1].set(True)
    grid = grid.at[mid, q3].set(True)
    grid = grid.at[q1, mid].set(True)
    grid = grid.at[q3, mid].set(True)
    return grid


class FourRooms(environment.Environment[EnvState, EnvParams]):
    def __init__(self, N: int = 13, use_visual_obs: bool = False):
        super().__init__()
        self.N = int(N)
        self.use_visual_obs = use_visual_obs
        #use_tabular_obs

        self.env_map = generate_four_rooms_map(self.N)
        self.occupied_map = (1.0 - self.env_map.astype(jnp.float32))

        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        default_pos = jnp.array([1, 1], dtype=jnp.int32)
        default_goal = jnp.array([self.N - 2, self.N - 2], dtype=jnp.int32)
        first_valid = self.coords[0]
        last_valid = self.coords[-1]
        self.default_pos = jnp.where(self.env_map[1, 1], default_pos, first_valid)
        self.default_goal = jnp.where(
            self.env_map[self.N - 2, self.N - 2], default_goal, last_valid
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.N * self.N)

    def _sample_coord(self, key: jax.Array) -> jax.Array:
        idx = jax.random.randint(key, (), 0, self.coords.shape[0], dtype=jnp.int32)
        return self.coords[idx]

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        key_rand, key_action = jax.random.split(key)
        random_action = jax.random.randint(key_action, (), 0, self.num_actions, dtype=jnp.int32)
        take_random = jax.random.uniform(key_rand, ()) < params.fail_prob
        action = jax.lax.select(take_random, random_action, jnp.asarray(action, dtype=jnp.int32))

        proposed_pos = state.pos + self.directions[action]
        can_move = self.env_map[proposed_pos[0], proposed_pos[1]]
        new_pos = jnp.where(can_move, proposed_pos, state.pos)

        reward = jnp.all(new_pos == state.goal).astype(jnp.float32)
        next_state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(next_state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(next_state, params)),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            {"discount": self.discount(next_state, params)},
        )

    def reset_env(self, key: jax.Array, params: EnvParams) -> Tuple[jax.Array, EnvState]:
        key_goal, key_pos = jax.random.split(key)
        sampled_goal = self._sample_coord(key_goal)
        sampled_pos = self._sample_coord(key_pos)

        goal = jax.lax.select(
            params.resample_goal_pos, sampled_goal, self.default_goal
        )
        pos = jax.lax.select(params.resample_init_pos, sampled_pos, self.default_pos)

        # Avoid trivial one-step terminals at reset.
        same_pos_goal = jnp.all(pos == goal)
        goal_idx = jnp.argmax(jnp.all(self.coords == goal, axis=1))
        alt_pos = self.coords[(goal_idx + 1) % self.coords.shape[0]]
        pos = jnp.where(same_pos_goal, alt_pos, pos)

        state = EnvState(pos=pos, goal=goal, time=0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams | None = None, key=None) -> jax.Array:
        if not self.use_visual_obs:
            return jnp.concatenate([state.pos, state.goal]).astype(jnp.float32)
        # if use_tabular_obs:
        # one hot 169, matching the evaluator
        # List of states: jnp.stack([y, x], axis=1).astype(jnp.int32)
        # Look up index in jnp.stack([y, x], axis=1).astype(jnp.int32)

        agent_map = (
            jnp.zeros((self.N, self.N), dtype=jnp.float32)
            .at[state.pos[0], state.pos[1]]
            .set(1.0)
        )
        goal_map = (
            jnp.zeros((self.N, self.N), dtype=jnp.float32)
            .at[state.goal[0], state.goal[1]]
            .set(1.0)
        )
        return jnp.stack([self.occupied_map, agent_map, goal_map], axis=-1)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        done_goal = jnp.all(state.pos == state.goal)
        return jnp.logical_or(done_goal, done_steps)

    @property
    def name(self) -> str:
        return "FourRoomsCustom-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        if self.use_visual_obs:
            return spaces.Box(0.0, 1.0, (self.N, self.N, 3), jnp.float32)
        return spaces.Box(0.0, float(self.N), (4,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "pos": spaces.Box(0, self.N - 1, (2,), jnp.int32),
                "goal": spaces.Box(0, self.N - 1, (2,), jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )


class FourRoomsExactValue:
    """Exact policy evaluation for Four Rooms under fixed goal."""

    def __init__(
        self,
        size: int = 13,
        fail_prob: float = 1.0 / 3.0,
        gamma: float = 0.99,
        episodic: bool = True,
        absorbing: bool = False, # Added absorbing flag
        use_visual_obs: bool = True,
        goal_pos: Tuple[int, int] | None = None,
    ):
        self.N = int(size)
        self.fail_prob = float(fail_prob)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.absorbing = absorbing
        self.use_visual_obs = use_visual_obs

        self.env_map = generate_four_rooms_map(self.N)
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        
        self.num_states = int(self.coords.shape[0])
        self.num_total_states = self.num_states # No dummy terminal state
        self.num_actions = 4
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        default_goal = jnp.array([self.N - 2, self.N - 2], dtype=jnp.int32)
        if goal_pos is None:
            self.goal = (
                default_goal
                if bool(self.env_map[default_goal[0], default_goal[1]])
                else self.coords[-1]
            )
        else:
            g = jnp.array(goal_pos, dtype=jnp.int32)
            if not bool(self.env_map[g[0], g[1]]):
                raise ValueError(f"goal_pos={goal_pos} is not a free cell.")
            self.goal = g

        default_start = jnp.array([1, 1], dtype=jnp.int32)
        self.start = (
            default_start
            if bool(self.env_map[default_start[0], default_start[1]])
            else self.coords[0]
        )
        self.start_idx = self._coord_to_idx(self.start)
        self.goal_idx = self._coord_to_idx(self.goal)

        # Free cells are valid for error aggregation; walls are ignored.
        self.reachable_mask = self.env_map.astype(jnp.float32)

        self.obs_stack = self._build_obs_stack()
        
        # Build Episodic/Absorbing P and Continuing P
        self.P, self.R_extrinsic = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))

    def _build_obs_stack(self) -> jax.Array:
        if self.use_visual_obs:
            agent_maps = np.zeros((self.num_states, self.N, self.N), dtype=np.float32)
            goal_map = np.zeros((self.N, self.N), dtype=np.float32)
            goal_map[int(self.goal[0]), int(self.goal[1])] = 1.0
            wall_map = np.asarray(self.occupied_map, dtype=np.float32)
            for i, (y, x) in enumerate(np.asarray(self.coords)):
                agent_maps[i, int(y), int(x)] = 1.0
            goal_stack = np.broadcast_to(goal_map, agent_maps.shape)
            wall_stack = np.broadcast_to(wall_map, agent_maps.shape)
            obs = np.stack([wall_stack, agent_maps, goal_stack], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)

        # Vector obs: [pos_y, pos_x, goal_y, goal_x].
        goal_vec = np.asarray(self.goal, dtype=np.float32)
        pos = np.asarray(self.coords, dtype=np.float32)
        goal_tile = np.broadcast_to(goal_vec, pos.shape)
        return jnp.asarray(np.concatenate([pos, goal_tile], axis=1), dtype=jnp.float32)

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        proposed = pos + self.directions[action]
        can_move = self.env_map[proposed[0], proposed[1]]
        return jnp.where(can_move, proposed, pos)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        R = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        p_rand = self.fail_prob / self.num_actions

        for s_idx in range(self.num_states):
            # 1. Goal State Dynamics (Terminal transitions)
            if s_idx == self.goal_idx:
                next_idx = self.start_idx if continuing else self.goal_idx
                P[s_idx, :, next_idx] = 1.0
                continue # No extrinsic reward for actions taken FROM the goal

            # 2. Standard State Dynamics
            pos = self.coords[s_idx]
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    p_exec = p_rand + (1.0 - self.fail_prob if executed_a == chosen_a else 0.0)
                    next_pos = self._step_pos(pos, executed_a)
                    next_idx = self._coord_to_idx(next_pos)
                    
                    P[s_idx, chosen_a, next_idx] += p_exec
                    
                    # Reward is granted when transitioning INTO the goal
                    if next_idx == self.goal_idx:
                        R[s_idx, chosen_a] += p_exec * 1.0

        return jnp.asarray(P), jnp.asarray(R)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        R_pi = jnp.einsum("sa,sa->s", pi, R_env)
        A = jnp.eye(self.num_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def get_value_grid(self, values: jax.Array, all= False) -> jax.Array:
        """Map per-state values to N x N grid (walls = 0)."""
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]
        print('shape of arg to get value grid is ', values.shape)
        grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
        return grid.at[self.coords[:, 0], self.coords[:, 1]].set(values)

    def compute_true_values(
        self,
        network: Any,
        params: Any,
        get_int_rew_per_state: Callable[[jax.Array], jax.Array],
        all = False
    ) -> Tuple[jax.Array, jax.Array, Any]:
        # 1. Forward Pass
        out = network.apply(params, self.obs_stack)
        
        if len(out) == 3:
            pi_dist, v_net_ext, v_net_int = out
            v_net_tuple = (
                v_net_ext.squeeze(),
                v_net_int.squeeze(),
            )
        else:
            pi_dist, v_net = out
            v_net_tuple = v_net.squeeze()

        pi = pi_dist.probs
        
        # 2. Extract intrinsic rewards natively
        r_int_s = get_int_rew_per_state(self.obs_stack)

        # 3. Mask goal state if purely Episodic (Not Absorbing)
        if self.episodic and not self.absorbing:
            r_int_s = r_int_s.at[self.goal_idx].set(0.0)

        # 4. Target selection and Reward projection
        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum("sam,m->sa", target_P, r_int_s)

        # 5. Solve Systems
        v_e = self.solve_linear_system(pi, self.P, self.R_extrinsic)
        v_i = self.solve_linear_system(pi, target_P, R_int_sa)


        return self.get_value_grid(v_e, all), self.get_value_grid(v_i, all), v_net_tuple
