import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Callable, Tuple, Optional
PyTree = Any
from jax.experimental import sparse as jsparse

class DeepSeaExactValue:
    def __init__(self, size: int, unscaled_move_cost: float = 0.01, gamma: float = 0.99, episodic: bool = False, absorbing: bool = False, dense: bool=False):
        self.N = size
        self.cost = unscaled_move_cost
        self.gamma = gamma
        self.episodic = episodic
        self.absorbing = absorbing 
        
        self.num_grid_states = size * size
        self.num_total_states = self.num_grid_states
        self.num_actions = 2
        self.dense = dense
        
        
        # 1. Pre-compute Observations Stack (N^2 x N x N x 1)
        self.obs_stack = self._create_obs_stack()
        self.reachable_mask = jnp.tril(jnp.ones((self.N, self.N)))
        
        # 2. Pre-compute Transition Matrices
        self.P, self.R_extrinsic = self._build_env_dynamics()
        self.P_cont = self._build_env_dynamics_continuing()

    def _create_obs_stack(self):
        """Creates a stack of one-hot observations for all grid states."""
        obs_stack = np.zeros((self.num_grid_states, self.N, self.N), dtype=np.float32)
        idx = 0
        for r in range(self.N):
            for c in range(self.N):
                obs_stack[idx, r, c] = 1.0
                idx += 1
        return jnp.array(obs_stack)[...,None]
    
    # def _build_env_dynamics(self):
    #     """
    #     Matches BSuite/Gymnax:
    #     - Row N-1 is the last playable row.
    #     - ANY action at N-1 leads to row N (terminal).
    #     - The 'Right' action (Action 1) at N-1, Column N-1 gives the goal reward.
    #     """
    #     num_states = self.num_total_states
    #     num_actions = self.num_actions
        
    #     P = np.zeros((num_states, num_actions, num_states))
    #     R = np.zeros((num_states, num_actions))
        
    #     # We need a 'virtual' sink state to represent row N (terminal)
    #     # However, to keep the matrix N^2, we usually make the bottom row states 
    #     # absorb into themselves to represent the end of the episode.
        
    #     for r in range(self.N):
    #         for c in range(self.N):
    #             curr_idx = r * self.N + c
                
    #             # --- Terminal Logic (Row N-1) ---
    #             if r == self.N - 1:
    #                 # In Gymnax, any action at row N-1 ends the game.
    #                 # We model this as a self-loop (absorbing)
    #                 P[curr_idx, :, curr_idx] = 1.0
                    
    #                 # Check for the Goal Reward: Right action at Bottom-Right
    #                 if c == self.N - 1:
    #                     # Action 1 is 'Right' in our logic
    #                     R[curr_idx, 1] = 1.0 - (self.cost / self.N)
    #                 continue

    #             # --- Standard Logic (Rows 0 to N-2) ---
    #             next_r = r + 1
                
    #             # Action 0: Left
    #             next_c_left = max(0, c - 1)
    #             P[curr_idx, 0, next_r * self.N + next_c_left] = 1.0

    #             # Action 1: Right
    #             next_c_right = min(self.N - 1, c + 1)
    #             P[curr_idx, 1, next_r * self.N + next_c_right] = 1.0
    #             R[curr_idx, 1] = -(self.cost / self.N)
                
    #     return jnp.array(P), jnp.array(R)

    def _build_env_dynamics(self):
        """
        Episodic & Absorbing Dynamics.
        Rows 0 to N-2: Standard Playable.
        Row N-1: Terminal Self-Loops.
        """
        num_states = self.num_total_states
        num_actions = self.num_actions
        
        P = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions))
        
        # Playable Rows (0 to N-2)
        for r in range(self.N - 1):
            for c in range(self.N):
                curr_idx = r * self.N + c
                
                # --- Action 0: Left ---
                next_r = r + 1
                next_c = max(0, c - 1)
                next_idx = next_r * self.N + next_c
                P[curr_idx, 0, next_idx] = 1.0

                # --- Action 1: Right ---
                next_r_right = r + 1
                next_c_right = min(self.N - 1, c + 1)
                next_idx_right = next_r_right * self.N + next_c_right
                P[curr_idx, 1, next_idx_right] = 1.0
                R[curr_idx, 1] = -(self.cost / self.N)

                # Goal Reward (Transition INTO bottom right cell)
                if next_r_right == self.N - 1 and next_c_right == self.N - 1:
                    R[curr_idx, 1] += 1.0 

        # Terminal Row (r = N-1) - Self loops forever
        for c in range(self.N):
            curr_idx = (self.N - 1) * self.N + c
            P[curr_idx, :, curr_idx] = 1.0
            
        return jnp.array(P), jnp.array(R)

    def _build_env_dynamics_continuing(self):
        """
        Continuing Dynamics. 
        Row N-1 teleports back to (0,0) instead of self-looping.
        """
        num_states = self.num_total_states
        num_actions = self.num_actions
        
        P = np.zeros((num_states, num_actions, num_states))
        start_idx = 0
        
        # Playable Rows (0 to N-2)
        for r in range(self.N - 1):
            for c in range(self.N):
                curr_idx = r * self.N + c
                
                next_c = max(0, c - 1)
                P[curr_idx, 0, (r+1)*self.N + next_c] = 1.0

                next_c_right = min(self.N - 1, c + 1)
                P[curr_idx, 1, (r+1)*self.N + next_c_right] = 1.0
        
        # Terminal Row (r = N-1) teleports to Start
        for c in range(self.N):
            curr_idx = (self.N - 1) * self.N + c
            P[curr_idx, :, start_idx] = 1.0
            
        return jnp.array(P)


    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        """
        Exact Policy Evaluation via Backward Induction.
        Note: P_env is intentionally unused here because DeepSea's deterministic 
        downward topology allows us to use O(N) spatial rolling instead of O(N^2) matrices.
        """
        # 1. Reshape flat vectors to grid topology
        R_grid = R_env.reshape((self.N, self.N, self.num_actions))
        pi_grid = pi.reshape((self.N, self.N, self.num_actions))
        
        # 2. Terminal Row (r = N-1) Expectation
        # r^pi(s) = sum_a pi(a|s) R(s,a)
        r_pi_bottom = jnp.sum(pi_grid[self.N-1] * R_grid[self.N-1], axis=-1)
        
        v_bottom = jnp.where(
            self.absorbing, 
            r_pi_bottom / (1.0 - self.gamma), 
            jnp.zeros(self.N)
        )

        # 3. Backward Step Definition
        def backward_row_step(v_next_row, scan_inputs):
            r_row, pi_row = scan_inputs
            
            # Deterministic DeepSea transitions (Left=0, Right=1)
            v_left = jnp.roll(v_next_row, shift=1).at[0].set(v_next_row[0])
            v_right = jnp.roll(v_next_row, shift=-1).at[-1].set(v_next_row[-1])
            
            # Q(s,a) = R(s,a) + gamma * V(s')
            q_row = r_row + self.gamma * jnp.stack([v_left, v_right], axis=-1)
            
            # V^pi(s) = sum_{a} pi(a|s) * Q(s,a)
            v_curr_row = jnp.sum(pi_row * q_row, axis=-1)
            
            return v_curr_row, v_curr_row

        # 4. Zip inputs and scan from row N-2 down to 0
        rows_to_process = (
            jnp.flip(R_grid[:-1], axis=0),
            jnp.flip(pi_grid[:-1], axis=0)
        )
        
        _, v_rest = jax.lax.scan(backward_row_step, v_bottom, rows_to_process)
        
        # 5. Reconstruct the grid and flatten to match the original (S,) output footprint
        v_grid = jnp.flip(jnp.concatenate([v_bottom[None, :], v_rest], axis=0), axis=0)
        
        return v_grid.flatten()

    # def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
    #     """
    #     Solves V = R^pi + gamma * P^pi V using Iterative Evaluation.
    #     Bypasses cuSolver LU decomposition limits for large state spaces.
    #     """
    #     # 1. Precompute State-to-State Dynamics and Rewards (O(|S|^2 |A|) time, once)
    #     # Shape: (S, S)
    #     P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        
    #     # Shape: (S,)
    #     R_pi = jnp.einsum("sa,sa->s", pi, R_env)

    #     # 2. Define the iterative Bellman operator
    #     def body_fn(v, _):
    #         # O(|S|^2) matrix-vector product per iteration via cuBLAS
    #         v_new = R_pi + self.gamma * (P_pi @ v)
    #         return v_new, None

    #     init_v = jnp.zeros(self.num_total_states, dtype=jnp.float32)
        
    #     # 3. Execute fixed-length scan for JIT compatibility
    #     # 1500 iterations guarantees convergence for gamma=0.99 within fp32 precision.
    #     final_v, _ = jax.lax.scan(body_fn, init_v, None, length=500)
        
    #     return final_v

    # def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array):
    #     P_pi = jnp.einsum('sa, sam -> sm', pi, P_env)
    #     R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        
    #     I = jnp.eye(self.num_total_states)
    #     A_mat = I - self.gamma * P_pi
        
    #     return jnp.linalg.solve(A_mat, R_pi)
    
    # use value iteration due to sparse P matrix
    # def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
    #     R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        
    #     # Compute dense and convert to sparse
    #     P_pi_dense = jnp.einsum('sa, sam -> sm', pi, P_env)
    #     P_pi_sparse = jsparse.BCOO.fromdense(P_pi_dense)

    #     def body_fn(v, _):
    #         v_new = R_pi + self.gamma * (P_pi_sparse @ v)
    #         return v_new, None

    #     init_v = jnp.zeros(self.num_total_states)

    #     final_v, _ = jax.lax.scan(body_fn, init_v, None, length=500)
        
    #     return final_v

    def get_value_grid(self, V_flat: jax.Array) -> jax.Array:
        """Reshapes flat value vector to N x N grid."""
        return V_flat.reshape((self.N, self.N))

    # def compute_true_values(self, network: Any, params: PyTree, get_int_rew: Callable, all=None
    #     ) -> Tuple[jax.Array, jax.Array, Any]:
            
    #         # 1. Forward Pass
    #         out = network.apply(params, self.obs_stack)
            
    #         def safe_squeeze(v):
    #             return v[..., 0] if v.ndim > 1 else v

    #         if len(out) == 2:
    #             pi_dist, v_net = out
    #             v_net_grid = self.get_value_grid(safe_squeeze(v_net))
    #         elif len(out) == 3:
    #             pi_dist, v_net_ext, v_net_int = out
    #             v_net_grid = (
    #                 self.get_value_grid(safe_squeeze(v_net_ext)), 
    #                 self.get_value_grid(safe_squeeze(v_net_int))
    #             )
            
    #         pi_matrix = pi_dist.probs 

    #         # 2. Get intrinsic rewards for all N^2 states
    #         r_int_s = get_int_rew(self.obs_stack)

    #         # 3. Mask terminal row if purely Episodic (Not Absorbing)
    #         if self.episodic and not self.absorbing:
    #             # Zero out the intrinsic reward for the entire bottom row
    #             mask = jnp.ones_like(r_int_s)
    #             terminal_start_idx = (self.N - 1) * self.N
    #             mask = mask.at[terminal_start_idx:].set(0.0)
    #             r_int_s = r_int_s * mask

    #         # 4. Target selection and Reward projection
    #         r_int_s = get_int_rew(self.obs_stack)

    #         # 2. Project to State-Action rewards
    #         # R_int_sa[s, a] = sum_{s'} P(s, a, s') * r_int(s')
    #         # This properly assigns the reward of the terminal state to the 
    #         # action taken in row N-2 that lands there.
    #         target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
    #         R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)

    #         # 3. Handle transitions AFTER the terminal state
    #         terminal_start_idx = (self.N - 1) * self.N
    #         goal_idx = self.num_total_states - 1

    #         # We create a mask for the STATES WE ACT FROM, not the states we land in.
    #         # Shape (S, 1) will naturally broadcast across the action dimension (A).
    #         sa_mask = jnp.ones((self.num_total_states, 1))

    #         # Zero out rewards for taking actions from ANY state in row N-1
    #         sa_mask = sa_mask.at[terminal_start_idx:].set(0.0)

    #         # If the goal is an absorbing infinite sinkhole, restore its ability 
    #         # to yield reward when taking an action from it.
    #         if self.absorbing:
    #             sa_mask = sa_mask.at[goal_idx].set(1.0)

    #         # Apply the mask to the state-action rewards
    #         R_int_sa = R_int_sa * sa_mask

    #         if self.dense:
    #             # 1. Subtract 1 from every single transition
    #             R_ext_modified = self.R_extrinsic - 1.0
                
    #             # 2. Zero out the infinite sinkhole (Terminal Row)
    #             terminal_start_idx = (self.N - 1) * self.N
    #             mask = jnp.ones_like(R_ext_modified)
    #             mask = mask.at[terminal_start_idx:, :].set(0.0) 
                
    #             Re = R_ext_modified * mask
    #         else:
    #             Re = self.R_extrinsic

    #         # 5. Solve
    #         v_e_true = self.solve_linear_system(pi_matrix, self.P, Re)
    #         v_i_true = self.solve_linear_system(pi_matrix, target_P, R_int_sa)

    #         return self.get_value_grid(v_e_true), self.get_value_grid(v_i_true), v_net_grid
    
    # def compute_optimal_intrinsic_values(self, network: Any, params: PyTree, get_int_rew: Callable, all=None
    #     ) -> Tuple[jax.Array, jax.Array, Any]:
            
    #         # 1. Forward Pass
    #         out = network.apply(params, self.obs_stack)
    #         def safe_squeeze(v): return v[..., 0] if v.ndim > 1 else v

    #         if len(out) == 2:
    #             pi_dist, v_net = out
    #             v_net_grid = self.get_value_grid(safe_squeeze(v_net))
    #         elif len(out) == 3:
    #             pi_dist, v_net_ext, v_net_int = out
    #             v_net_grid = (
    #                 self.get_value_grid(safe_squeeze(v_net_ext)), 
    #                 self.get_value_grid(safe_squeeze(v_net_int))
    #             )

    #         # 2. Extract Rewards
    #         r_int_s = get_int_rew(self.obs_stack)
    #         if self.episodic and not self.absorbing:
    #             terminal_start_idx = (self.N - 1) * self.N
    #             mask = jnp.ones_like(r_int_s).at[terminal_start_idx:].set(0.0)
    #             r_int_s = r_int_s * mask

    #         target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
    #         R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)

    #         # 3. Solver
    #         def solve_v_star(R_sa):
    #             R_grid = R_sa.reshape((self.N, self.N, self.num_actions))
    #             v_bottom = jnp.where(self.absorbing, R_grid[self.N-1, :, 0] / (1 - self.gamma), jnp.zeros(self.N))

    #             def backward_row_step(v_next_row, r_row):
    #                 v_left = jnp.roll(v_next_row, shift=1).at[0].set(v_next_row[0])
    #                 v_right = jnp.roll(v_next_row, shift=-1).at[-1].set(v_next_row[-1])
    #                 q_row = r_row + self.gamma * jnp.stack([v_left, v_right], axis=-1)
    #                 v_curr_row = jnp.max(q_row, axis=-1)
    #                 return v_curr_row, v_curr_row

    #             rows_to_process = jnp.flip(R_grid[:-1], axis=0)
    #             _, v_rest = jax.lax.scan(backward_row_step, v_bottom, rows_to_process)
    #             return jnp.flip(jnp.concatenate([v_bottom[None, :], v_rest], axis=0), axis=0)

    #         # 4. Extrinsic Logic
    #         if self.dense:
    #             R_ext_modified = (self.R_extrinsic - 1.0)
    #             terminal_start_idx = (self.N - 1) * self.N
    #             # Create mask to zero out row N-1
    #             mask = jnp.ones_like(R_ext_modified).at[terminal_start_idx:, :].set(0.0)
    #             Re = R_ext_modified * mask
    #         else:
    #             Re = self.R_extrinsic

    #         v_i_star_grid = solve_v_star(R_int_sa)
    #         v_e_star_grid = solve_v_star(Re)

    #         return v_e_star_grid, v_i_star_grid, v_net_grid

    def compute_true_values(self, network: Any, params: PyTree, get_int_rew: Callable, all=None) -> Tuple[jax.Array, jax.Array, Any]:
        # 1. Forward Pass
        out = network.apply(params, self.obs_stack)
        
        def safe_squeeze(v):
            return v[..., 0] if v.ndim > 1 else v

        if len(out) == 2:
            pi_dist, v_net = out
            v_net_grid = self.get_value_grid(safe_squeeze(v_net))
        elif len(out) == 3:
            pi_dist, v_net_ext, v_net_int = out
            v_net_grid = (
                self.get_value_grid(safe_squeeze(v_net_ext)), 
                self.get_value_grid(safe_squeeze(v_net_int))
            )
        
        pi_matrix = pi_dist.probs 

        # 2. Get intrinsic rewards for all N^2 states
        r_int_s = get_int_rew(self.obs_stack)

        # 3. Target selection and Reward projection
        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)

        terminal_start_idx = (self.N - 1) * self.N
        goal_idx = self.num_total_states - 1

        # We create a mask for the STATES WE ACT FROM
        sa_mask = jnp.ones((self.num_total_states, 1))

        # Zero out the whole terminal row EXCEPT the goal state
        # This explicitly leaves sa_mask[goal_idx] as 1.0
        sa_mask = sa_mask.at[terminal_start_idx : goal_idx].set(0.0)

        # If episodic but NOT absorbing, transitions after the goal should also yield 0
        if not self.absorbing:
            sa_mask = sa_mask.at[goal_idx].set(0.0)

        # Apply the mask ONCE
        R_int_sa = R_int_sa * sa_mask

        # 5. Extrinsic logic
        if self.dense:
            R_ext_modified = self.R_extrinsic - 1.0
            mask = jnp.ones_like(R_ext_modified)
            mask = mask.at[terminal_start_idx:, :].set(0.0) 
            Re = R_ext_modified * mask
        else:
            Re = self.R_extrinsic

        # 6. Solve
        v_e_true = self.solve_linear_system(pi_matrix, self.P, Re)
        v_i_true = self.solve_linear_system(pi_matrix, target_P, R_int_sa)

        return self.get_value_grid(v_e_true), self.get_value_grid(v_i_true), v_net_grid

    def compute_optimal_intrinsic_values(self, network: Any, params: PyTree, get_int_rew: Callable, all=None) -> Tuple[jax.Array, jax.Array, Any]:
        # 1. Forward Pass
        out = network.apply(params, self.obs_stack)
        def safe_squeeze(v): return v[..., 0] if v.ndim > 1 else v

        if len(out) == 2:
            pi_dist, v_net = out
            v_net_grid = self.get_value_grid(safe_squeeze(v_net))
        elif len(out) == 3:
            pi_dist, v_net_ext, v_net_int = out
            v_net_grid = (
                self.get_value_grid(safe_squeeze(v_net_ext)), 
                self.get_value_grid(safe_squeeze(v_net_int))
            )

        # 2. Extract Rewards and Project
        r_int_s = get_int_rew(self.obs_stack)
        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)

        # 3. Handle transitions AFTER the terminal state (Matches True Values)
        terminal_start_idx = (self.N - 1) * self.N
        goal_idx = self.num_total_states - 1

        sa_mask = jnp.ones((self.num_total_states, 1))
        sa_mask = sa_mask.at[terminal_start_idx:].set(0.0)
        
        if self.absorbing:
            sa_mask = sa_mask.at[goal_idx].set(1.0)

        R_int_sa = R_int_sa * sa_mask

        # 4. Solver
        def solve_v_star(R_sa):
            R_grid = R_sa.reshape((self.N, self.N, self.num_actions))
            # v_bottom perfectly handles the sa_mask! 
            # Non-goals will be 0 / (1-gamma) = 0. Goal will be r / (1-gamma).
            v_bottom = jnp.where(self.absorbing, R_grid[self.N-1, :, 0] / (1 - self.gamma), jnp.zeros(self.N))

            def backward_row_step(v_next_row, r_row):
                v_left = jnp.roll(v_next_row, shift=1).at[0].set(v_next_row[0])
                v_right = jnp.roll(v_next_row, shift=-1).at[-1].set(v_next_row[-1])
                q_row = r_row + self.gamma * jnp.stack([v_left, v_right], axis=-1)
                v_curr_row = jnp.max(q_row, axis=-1)
                return v_curr_row, v_curr_row

            rows_to_process = jnp.flip(R_grid[:-1], axis=0)
            _, v_rest = jax.lax.scan(backward_row_step, v_bottom, rows_to_process)
            return jnp.flip(jnp.concatenate([v_bottom[None, :], v_rest], axis=0), axis=0)

        # 5. Extrinsic Logic
        if self.dense:
            R_ext_modified = (self.R_extrinsic - 1.0)
            terminal_start_idx = (self.N - 1) * self.N
            mask = jnp.ones_like(R_ext_modified).at[terminal_start_idx:, :].set(0.0)
            Re = R_ext_modified * mask
        else:
            Re = self.R_extrinsic

        v_i_star_grid = solve_v_star(R_int_sa)
        v_e_star_grid = solve_v_star(Re)

        return v_e_star_grid, v_i_star_grid, v_net_grid