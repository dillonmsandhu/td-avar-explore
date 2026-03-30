import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Callable, Tuple, Optional
PyTree = Any

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

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array):
        P_pi = jnp.einsum('sa, sam -> sm', pi, P_env)
        R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        
        I = jnp.eye(self.num_total_states)
        A_mat = I - self.gamma * P_pi
        
        return jnp.linalg.solve(A_mat, R_pi)

    def get_value_grid(self, V_flat: jax.Array) -> jax.Array:
        """Reshapes flat value vector to N x N grid."""
        return V_flat.reshape((self.N, self.N))

    def compute_true_values(self, network: Any, params: PyTree, get_int_rew: Callable, all=None
        ) -> Tuple[jax.Array, jax.Array, Any]:
            
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

            # 3. Mask terminal row if purely Episodic (Not Absorbing)
            if self.episodic and not self.absorbing:
                # Zero out the intrinsic reward for the entire bottom row
                mask = jnp.ones_like(r_int_s)
                terminal_start_idx = (self.N - 1) * self.N
                mask = mask.at[terminal_start_idx:].set(0.0)
                r_int_s = r_int_s * mask

            # 4. Target selection and Reward projection
            target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
            R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)

            if self.dense:
                # 1. Subtract 1 from every single transition
                R_ext_modified = self.R_extrinsic - 1.0
                
                # 2. Zero out the infinite sinkhole (Terminal Row)
                terminal_start_idx = (self.N - 1) * self.N
                mask = jnp.ones_like(R_ext_modified)
                mask = mask.at[terminal_start_idx:, :].set(0.0) 
                
                Re = R_ext_modified * mask
            else:
                Re = self.R_extrinsic

            # 5. Solve
            v_e_true = self.solve_linear_system(pi_matrix, self.P, Re)
            v_i_true = self.solve_linear_system(pi_matrix, target_P, R_int_sa)

            return self.get_value_grid(v_e_true), self.get_value_grid(v_i_true), v_net_grid
