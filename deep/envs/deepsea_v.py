import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Callable, Tuple, Optional
PyTree = Any

class DeepSeaExactValue:
    def __init__(self, size: int, unscaled_move_cost: float = 0.01, gamma: float = 0.99):
        self.N = size
        self.cost = unscaled_move_cost
        self.gamma = gamma
        self.num_grid_states = size * size
        self.num_total_states = self.num_grid_states + 1  # +1 for Absorbing Terminal
        self.terminal_idx = self.num_grid_states
        self.reachable_mask = jnp.tril(jnp.ones((size,size)))
        self.num_actions = 2
        # 1. Pre-compute Observations Stack (N^2 x N x N)
        self.obs_stack = self._create_obs_stack()
        
        # 2. Pre-compute Transition Matrices
        # A. Episodic P (Standard): Falling off grid -> Terminal State
        self.P, self.R_extrinsic = self._build_env_dynamics()
        
        # B. Continuing P (LSTD): Falling off grid -> Start State
        self.P_cont = self._build_env_dynamics_continuing()

    def _create_obs_stack(self):
        """Creates a stack of one-hot observations for all grid states."""
        # Shape: (N^2, N, N, 1)
        obs_stack = np.zeros((self.num_grid_states, self.N, self.N), dtype=np.float32)
        idx = 0
        for r in range(self.N):
            for c in range(self.N):
                obs_stack[idx, r, c] = 1.0
                idx += 1
        return jnp.array(obs_stack)[...,None]

    def _build_env_dynamics(self):
        """Constructs P and R for the standard EPISODIC setting."""
        num_states = self.num_total_states
        num_actions = self.num_actions
        
        P = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions))
        
        for r in range(self.N):
            for c in range(self.N):
                curr_idx = r * self.N + c
                
                # --- Action 0: Left ---
                next_r = r + 1
                next_c = max(0, c - 1)
                next_idx = self.terminal_idx if next_r >= self.N else next_r * self.N + next_c
                
                P[curr_idx, 0, next_idx] = 1.0
                R[curr_idx, 0] = 0.0 

                # --- Action 1: Right ---
                next_r_right = r + 1
                next_c_right = min(self.N - 1, c + 1)
                next_idx_right = self.terminal_idx if next_r_right >= self.N else next_r_right * self.N + next_c_right

                P[curr_idx, 1, next_idx_right] = 1.0
                R[curr_idx, 1] = -(self.cost / self.N)

        # Episodic Goal Reward: Last step before Terminal
        bottom_right_idx = (self.N - 1) * self.N + (self.N - 1)
        R[bottom_right_idx, 1] += 1.0 

        # Terminal Loop
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :] = 0.0
        
        return jnp.array(P), jnp.array(R)

    def _build_env_dynamics_continuing(self):
        """
        Constructs P and R for the CONTINUING setting.
        Transitions that would hit 'Terminal' instead loop back to 'Start' (State 0).
        """
        num_states = self.num_total_states
        num_actions = self.num_actions
        
        P = np.zeros((num_states, num_actions, num_states))
        start_idx = 0
        
        for r in range(self.N):
            for c in range(self.N):
                curr_idx = r * self.N + c
                
                # --- Action 0: Left ---
                next_r = r + 1
                next_c = max(0, c - 1)
                # Loop to Start if falling off
                next_idx = start_idx if next_r >= self.N else next_r * self.N + next_c
                
                P[curr_idx, 0, next_idx] = 1.0

                # --- Action 1: Right ---
                next_r_right = r + 1
                next_c_right = min(self.N - 1, c + 1)
                # Loop to Start if falling off
                next_idx_right = start_idx if next_r_right >= self.N else next_r_right * self.N + next_c_right

                P[curr_idx, 1, next_idx_right] = 1.0
        
        # Continuing Goal Reward: On the edge transitioning Bottom-Right -> Start
        bottom_right_idx = (self.N - 1) * self.N + (self.N - 1)

        # Dummy Terminal Loop (for shape consistency)
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        
        return jnp.array(P)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array):
        """
        Generic solver: (I - gamma * P_pi)^-1 * R_pi
        Using a specific P matrix (Episodic or Continuing).
        """
        # P^pi[s, s'] = sum_a (pi[s, a] * P[s, a, s'])
        P_pi = jnp.einsum('sa, sam -> sm', pi, P_env)
        
        # R^pi[s] = sum_a (pi[s, a] * R[s, a])
        R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        
        I = jnp.eye(self.num_total_states)
        A_mat = I - self.gamma * P_pi
        
        # Solve for Value
        V = jnp.linalg.solve(A_mat, R_pi)
        return V

    def get_value_grid(self, V_flat: jax.Array) -> jax.Array:
        """Reshapes flat value vector to N x N grid."""
        return V_flat[:self.num_grid_states].reshape((self.N, self.N))

    def compute_true_values(self, network: Any, params: PyTree, get_features: Callable, get_int_rew: Callable
        ) -> Tuple[jax.Array, jax.Array, Any]:
            """
            Computes V_e (Episodic) and V_i (Continuing).
            """
            # 1. Get Network Output
            out = network.apply(params, self.obs_stack)
            if len(out) == 2:
                pi_probs, v_net = out
                v_net_grid = self.get_value_grid(v_net[..., 0] if v_net.ndim > 1 else v_net)
            elif len(out) == 3:
                pi_probs, v_net_ext, v_net_int = out
                v_net_grid = (
                    self.get_value_grid(v_net_ext), 
                    self.get_value_grid(v_net_int)
                )
            
            # Construct full policy matrix (N^2 + 1, 2)
            # Append dummy policy for terminal state
            terminal_policy = jnp.array([[1.0, 0.0]])
            pi_full = jnp.vstack([pi_probs.probs, terminal_policy])

            # 2. Compute Intrinsic Reward Vector (State-Based)
            # Note: We use the Sum-based get_int_rew (no 'N' argument needed)
            feats = get_features(self.obs_stack)
            r_next_grid = get_int_rew(feats) # shape (N^2,)
            
            # Append 0.0 for terminal state
            r_next_all = jnp.concatenate([r_next_grid, jnp.array([0.0])])

            # 3. Project Intrinsic Reward to (s,a) using CONTINUING Dynamics
            # We use P_cont to ensure that actions leading to "Start" get the reward of "Start".
            R_int_sa = jnp.einsum('sam, m -> sa', self.P_cont, r_next_all)

            # 4. Solve Extrinsic Value (Uses Episodic P)
            v_e_true = self.solve_linear_system(pi_full, self.P, self.R_extrinsic)

            # 5. Solve Intrinsic Value (Uses Continuing P)
            v_i_true = self.solve_linear_system(pi_full, self.P_cont, R_int_sa)

            return self.get_value_grid(v_e_true), self.get_value_grid(v_i_true), v_net_grid
        