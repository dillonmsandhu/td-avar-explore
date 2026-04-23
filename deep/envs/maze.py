from typing import Any, Callable, Dict, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax.experimental import sparse as jsparse

@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    goal: jax.Array
    time: jax.Array

@struct.dataclass
class EnvParams(environment.EnvParams):
    fail_prob: float = 0.0
    resample_init_pos: bool = False
    resample_goal_pos: bool = False
    max_steps_in_episode: int = int(1e6)

def generate_sparse_maze_map(grid_size: int = 100) -> jax.Array:
    """
    Builds a 'Comb Maze'. 
    A central corridor with multiple deep dead-end branches.
    Returns a boolean mask where True = free space, False = wall.
    """
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    
    # 1. Main Corridor (y=50, x from 10 to 90) -> 81 states
    grid[50, 10:91] = True
    
    # 2. Deep Dead-End Branches -> 7 branches, each 30 steps deep (15 up, 15 down)
    for x in [20, 30, 40, 50, 60, 70, 80]:
        grid[35:66, x] = True
        
    return jnp.asarray(grid)


class SparseMaze(environment.Environment[EnvState, EnvParams]):
    def __init__(self, N: int = 100):
        super().__init__()
        self.N = int(N)
        
        # 1. Map and Coordinates
        self.env_map = generate_sparse_maze_map(self.N)
        self.occupied_map = (1.0 - self.env_map.astype(jnp.float32))

        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        # Start at the far left of the corridor, Goal at the far right
        self.default_pos = jnp.array([50, 10], dtype=jnp.int32)
        self.default_goal = jnp.array([50, 90], dtype=jnp.int32)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self, key: jax.Array, state: EnvState, action: int | jax.Array, params: EnvParams):
        key_rand, key_action = jax.random.split(key)
        random_action = jax.random.randint(key_action, (), 0, 4, dtype=jnp.int32)
        take_random = jax.random.uniform(key_rand, ()) < params.fail_prob
        action = jax.lax.select(take_random, random_action, jnp.asarray(action, dtype=jnp.int32))

        proposed_pos = state.pos + self.directions[action]
        can_move = self.env_map[proposed_pos[0], proposed_pos[1]]
        new_pos = jnp.where(can_move, proposed_pos, state.pos)

        # Reward for entering the goal
        reward = jnp.all(new_pos == state.goal).astype(jnp.float32)
        next_state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(next_state, params)
        
        return (
            jax.lax.stop_gradient(self.get_obs(next_state, params)),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            {"discount": self.discount(next_state, params), "real_next_obs": self.get_obs(next_state, params)}
        )

    def reset_env(self, key: jax.Array, params: EnvParams):
        # Deterministic Start/Goal for the Oracle Baseline
        pos = self.default_pos
        goal = self.default_goal
        state = EnvState(pos=pos, goal=goal, time=0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams | None = None, key=None) -> jax.Array:
        # Full 100x100x3 Visual Observation
        agent_map = jnp.zeros((self.N, self.N), dtype=jnp.float32).at[state.pos[0], state.pos[1]].set(1.0)
        goal_map = jnp.zeros((self.N, self.N), dtype=jnp.float32).at[state.goal[0], state.goal[1]].set(1.0)
        return jnp.stack([self.occupied_map, agent_map, goal_map], axis=-1)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        reached_goal = jnp.all(state.pos == state.goal)
        return reached_goal

    @property
    def name(self) -> str: return "SparseMaze-v0"
    @property
    def num_actions(self) -> int: return 4
    def action_space(self, params: EnvParams | None = None): return spaces.Discrete(4)
    def observation_space(self, params: EnvParams): return spaces.Box(0.0, 1.0, (self.N, self.N, 3), jnp.float32)
    def state_space(self, params: EnvParams):
        return spaces.Dict({"pos": spaces.Box(0, self.N - 1, (2,), jnp.int32)})

class SparseMazeExactValue:
    def __init__(self, size: int = 100, fail_prob: float = 0.0, gamma: float = 0.99, episodic: bool = True, absorbing: bool = False):
        self.N = int(size)
        self.fail_prob = float(fail_prob)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.absorbing = absorbing

        self.env_map = generate_sparse_maze_map(self.N)
        self.reachable_mask = self.env_map.astype(jnp.float32)
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        
        self.num_states = int(self.coords.shape[0])
        self.num_total_states = self.num_states
        self.num_actions = 4
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

        self.start = jnp.array([50, 10], dtype=jnp.int32)
        self.goal = jnp.array([50, 90], dtype=jnp.int32)
        self.start_idx = self._coord_to_idx(self.start)
        self.goal_idx = self._coord_to_idx(self.goal)

        # N^2 -> S projection structures
        self.obs_stack = self._build_obs_stack()
        self.P, self.R_extrinsic = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))

    def _build_obs_stack(self) -> jax.Array:
        # Batch construct observations for all S reachable states
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

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        proposed = pos + self.directions[action]
        can_move = self.env_map[proposed[0], proposed[1]]
        return jnp.where(can_move, proposed, pos)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        R = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        p_rand = self.fail_prob / self.num_actions

        for s_idx in range(self.num_states):
            if s_idx == self.goal_idx:
                next_idx = self.start_idx if continuing else self.goal_idx
                P[s_idx, :, next_idx] = 1.0
                continue 

            pos = self.coords[s_idx]
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    p_exec = p_rand + (1.0 - self.fail_prob if executed_a == chosen_a else 0.0)
                    next_pos = self._step_pos(pos, executed_a)
                    next_idx = self._coord_to_idx(next_pos)
                    
                    P[s_idx, chosen_a, next_idx] += p_exec
                    if next_idx == self.goal_idx:
                        R[s_idx, chosen_a] += p_exec * 1.0

        return jnp.asarray(P), jnp.asarray(R)

    def get_value_grid(self, values: jax.Array, all=False) -> jax.Array:
        """Map the S-dimensional value vector back to the 100x100 spatial grid."""
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]
        grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
        return grid.at[self.coords[:, 0], self.coords[:, 1]].set(values)

    def compute_optimal_intrinsic_values(self, network: Any, params: Any, 
    get_int_rew_per_state: Callable, all=False
    ):
        # 1. Forward Pass
        def _net_step(unused, x):
            res = network.apply(params, x[None, ...])
            # squeeze(0) removes the batch dim added by [None, ...]
            return None, jax.tree_map(lambda arr: arr.squeeze(0), res)

        _, out = jax.lax.scan(_net_step, None, self.obs_stack)
        
        # Unpack based on your 3-head Actor-Critic (dist, value, i_value)
        # Note: We avoid 'len(leaves)' if we know the tuple structure
        try:
            pi_dist, v_ext_raw, v_int_raw = out
            is_3_head = True
        except (ValueError, TypeError):
            pi_dist, v_ext_raw = out
            v_int_raw = jnp.zeros_like(v_ext_raw)
            is_3_head = False

        # Ensure shapes are (S,) and (S, A)
        v_ext_s = v_ext_raw.squeeze() # (S,)
        v_int_s = v_int_raw.squeeze() # (S,)
        pi = pi_dist.probs # Should be (S, A)
        
        if pi.ndim == 3: # Handle case where probs has an extra dim from scan
            pi = pi.squeeze(1)

        # 2. Extract Rewards
        r_int_s = get_int_rew_per_state(self.obs_stack)
        if self.episodic and not self.absorbing:
            r_int_s = r_int_s.at[self.goal_idx].set(0.0)

        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum("sam,m->sa", target_P, r_int_s)

        # 3. Fast Value Iteration (over 291 states, not 10,000)
        def value_iteration(P, R, gamma, eps=1e-8):
            def bellman_step(v_curr):
                return jnp.max(R + gamma * jnp.einsum("sam, m -> sa", P, v_curr), axis=1)
            # 1000 iterations is mathematically excessive for a 291-state MDP, guaranteeing convergence
            return jax.lax.fori_loop(0, 1000, lambda i, v: bellman_step(v), jnp.zeros(self.num_states))

        v_i_star = value_iteration(target_P, R_int_sa, self.gamma)
        v_e_star = value_iteration(self.P, self.R_extrinsic, self.gamma)

        return self.get_value_grid(v_e_star, all), self.get_value_grid(v_i_star, all), v_net_tuple

    # def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
    #     P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
    #     R_pi = jnp.einsum("sa,sa->s", pi, R_env)
    #     A = jnp.eye(self.num_states) - self.gamma * P_pi
    #     return jnp.linalg.solve(A, R_pi)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        
        # 2. Compute dense and convert to sparse
        P_pi_dense = jnp.einsum('sa, sam -> sm', pi, P_env)
        P_pi_sparse = jsparse.BCOO.fromdense(P_pi_dense)

        def body_fn(v, _):
            v_new = R_pi + self.gamma * (P_pi_sparse @ v)
            return v_new, None

        # Initial state
        init_v = jnp.zeros(self.num_states)

        # 4. Run the compiled scan loop. 
        # We pass xs=None and specify the length explicitly.
        final_v, _ = jax.lax.scan(body_fn, init_v, None, length=1_000)
        
        return final_v
        
    def compute_true_values(
        self,
        network: Any,
        params: Any,
        get_int_rew_per_state: Callable[[jax.Array], jax.Array],
        all = False
    ) -> Tuple[jax.Array, jax.Array, Any]:
        
        # 1. Forward Pass
        def _net_step(unused, x):
            res = network.apply(params, x[None, ...])
            # squeeze(0) removes the batch dim added by [None, ...]
            return None, jax.tree_map(lambda arr: arr.squeeze(0), res)

        _, out = jax.lax.scan(_net_step, None, self.obs_stack)
        
        # Unpack based on your 3-head Actor-Critic (dist, value, i_value)
        # Note: We avoid 'len(leaves)' if we know the tuple structure
        try:
            pi_dist, v_ext_raw, v_int_raw = out
            is_3_head = True
        except (ValueError, TypeError):
            pi_dist, v_ext_raw = out
            v_int_raw = jnp.zeros_like(v_ext_raw)
            is_3_head = False

        # Ensure shapes are (S,) and (S, A)
        v_ext_s = v_ext_raw.squeeze() # (S,)
        v_int_s = v_int_raw.squeeze() # (S,)
        pi = pi_dist.probs # Should be (S, A)
        
        if pi.ndim == 3: # Handle case where probs has an extra dim from scan
            pi = pi.squeeze(1)

        # 2. Process Rewards
        r_int_s = get_int_rew_per_state(self.obs_stack)
        if self.episodic and not self.absorbing:
            r_int_s = r_int_s.at[self.goal_idx].set(0.0)

        # 3. Solver
        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum("sam,m->sa", target_P, r_int_s)

        # Compute on-policy ground truth via Linear System
        v_e_star = self.solve_linear_system(pi, self.P, self.R_extrinsic)
        v_i_star = self.solve_linear_system(pi, target_P, R_int_sa)

        # 4. Format for Metric
        v_net_tuple = (
            self.get_value_grid(v_ext_s),
            self.get_value_grid(v_int_s) if is_3_head else jnp.zeros_like(v_ext_s)
        )

        return (
            self.get_value_grid(v_e_star, all), 
            self.get_value_grid(v_i_star, all), 
            v_net_tuple
        )