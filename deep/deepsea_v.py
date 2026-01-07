import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Callable, Tuple
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
        # 1. Pre-compute Observations Stack (N^2 x N x N)
        self.obs_stack = self._create_obs_stack()
        
        # 2. Pre-compute Transition Matrix P (M x A x M) 
        #    and Extrinsic Reward Matrix R (M x A)
        self.P, self.R_extrinsic = self._build_env_dynamics()

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
            """
            Constructs the static P and R matrices.
            Actions: 0 = Left, 1 = Right
            """
            num_states = self.num_total_states
            num_actions = 2
            
            P = np.zeros((num_states, num_actions, num_states))
            R = np.zeros((num_states, num_actions))
            
            curr_idx = 0 # Initialize to ensure scope safety if N=0 (edge case)

            # Loop over all non-terminal grid states
            for r in range(self.N):
                for c in range(self.N):
                    curr_idx = r * self.N + c
                    
                    # --- Action 0: Left ---
                    # Transition: r -> r+1, c -> max(0, c-1)
                    next_r = r + 1
                    next_c = max(0, c - 1)
                    
                    if next_r >= self.N:
                        next_idx = self.terminal_idx
                    else:
                        next_idx = next_r * self.N + next_c
                    
                    P[curr_idx, 0, next_idx] = 1.0
                    R[curr_idx, 0] = 0.0  # Zero cost for Left

                    # --- Action 1: Right ---
                    # Transition: r -> r+1, c -> c+1
                    next_r_right = r + 1
                    next_c_right = c + 1 
                    
                    if next_c_right >= self.N:
                        next_c_right = self.N - 1
                    
                    if next_r_right >= self.N:
                        next_idx_right = self.terminal_idx
                    else:
                        next_idx_right = next_r_right * self.N + next_c_right

                    P[curr_idx, 1, next_idx_right] = 1.0
                    
                    # --- Standard Move Cost ---
                    # Apply the cost to ALL right moves initially
                    R[curr_idx, 1] = -(self.cost / self.N)
                    

            # --- Goal Reward Override ---
            # The loop finishes with curr_idx = (N-1)*N + (N-1), which is the Bottom-Right state.
            R[curr_idx, 1] += 1.0 

            # --- Terminal State Handling ---
            # Absorbing state: transitions to itself, 0 reward
            P[self.terminal_idx, :, self.terminal_idx] = 1.0
            R[self.terminal_idx, :] = 0.0
            
            return jnp.array(P), jnp.array(R)

    def solve_true_value(self, pi_grid: jax.Array, R_matrix: jax.Array):
            """
            Computes V_pi = (I - gamma * P_pi)^-1 * R_pi using a tabular policy.

            Args:
                pi_grid: Shape (N^2, 2). The policy probabilities for the grid states.
                R_custom: Optional Shape (M, 2). If provided, computes value w.r.t this reward
                        matrix instead of self.R_extrinsic. Useful for intrinsic rewards.
            """

            # 2. Append dummy policy for the terminal state
            # The terminal state absorbs, so policy doesn't strictly matter, 
            # but we need dimensions to match (M x 2).
            terminal_policy = jnp.array([[1.0, 0.0]]) 
            pi = jnp.vstack([pi_grid, terminal_policy])
            
            # 3. Compute P^pi (Transition matrix induced by policy)
            # P^pi[s, s'] = sum_a (pi[s, a] * P[s, a, s'])
            # Shape: (M, M)
            P_pi = jnp.einsum('sa, sam -> sm', pi, self.P)
            
            # 4. Compute R^pi (Expected reward vector induced by policy)
            # R^pi[s] = sum_a (pi[s, a] * R[s, a])
            # Shape: (M,)
            R_pi = jnp.einsum('sa, sa -> s', pi, R_matrix)
            
            # 5. Solve Linear System: (I - gamma * P^pi) V = R^pi
            I = jnp.eye(self.num_total_states)
            A = I - self.gamma * P_pi
            
            V_true = jnp.linalg.solve(A, R_pi)
            
            return V_true
    
    def solve_true_intrinsic_and_extrinsic_value(self, pi_grid: jax.Array, Re: jax.Array, Ri: jax.Array):
            """
            Computes V_pi = (I - gamma * P_pi)^-1 * [Re; R_i].
            Returns: V_e, V_i
            Args:
                pi_grid: Shape (N^2, 2). The policy probabilities for the grid states.
                R_custom: Optional Shape (M, 2). If provided, computes value w.r.t this reward
                        matrix instead of self.R_extrinsic. Useful for intrinsic rewards.
            
            """

            # 2. Append dummy policy for the terminal state
            # The terminal state absorbs, so policy doesn't strictly matter, 
            # but we need dimensions to match (M x 2).
            terminal_policy = jnp.array([[1.0, 0.0]]) 
            pi = jnp.vstack([pi_grid, terminal_policy])
            
            # 3. Compute P^pi (Transition matrix induced by policy)
            # P^pi[s, s'] = sum_a (pi[s, a] * P[s, a, s'])
            # Shape: (M, M)
            P_pi = jnp.einsum('sa, sam -> sm', pi, self.P)
            
            # 4. Compute R^pi (Expected reward vector induced by policy)
            # R^pi[s] = sum_a (pi[s, a] * R[s, a])
            # Shape: (M,)
            eR_pi = jnp.einsum('sa, sa -> s', pi, Re)
            iR_pi = jnp.einsum('sa, sa -> s', pi, Ri)
            
            # 5. Solve Linear System: (I - gamma * P^pi) V = R^pi
            I = jnp.eye(self.num_total_states)
            A = I - self.gamma * P_pi
            A_inv = jnp.linalg.solve(A, I)
            V_e = A_inv @ eR_pi
            V_i = A_inv @ iR_pi
            
            return V_e, V_i
    
    def get_value_grid(self, V_flat: jax.Array) -> jax.Array:
            """
            Reshapes the flat value vector (N^2 + 1) into an N x N grid.
            Discards the terminal state value.
            """
            # First N^2 elements (remove absorbing state)
            V_grid_flat = V_flat[:self.num_grid_states]
            
            # Reshape to (N, N)
            return V_grid_flat.reshape((self.N, self.N))

    def compute_true_values(self, network: Any, params: PyTree,lstd_state: Dict, get_features: Callable, get_int_rew: Callable
        ) -> Tuple[jax.Array, jax.Array, Any]:
            """
            Computes ground truth Extrinsic and Intrinsic Values and returns them 
            alongside the network's value estimates (reshaped to grids).
            """

            # 1. Get Network Output (Handle 2-tuple vs 3-tuple)
            out = network.apply(params, self.obs_stack)
            
            if len(out) == 2:
                # Standard PPO: (pi, v_total)
                pi, v_net = out
            elif len(out) == 3:
                # Split Value PPO: (pi, v_ext, v_int)
                pi, v_net_ext, v_net_int = out
                v_net = (v_net_ext, v_net_int)

            # 2. Helper to compute Intrinsic Reward Matrix R(s,a)
            def compute_intrinsic_rew_matrix():
                """Computes R_i(s,a) = E[r_i(s') | s,a]"""
                feats = get_features(self.obs_stack)
                # Calculate intrinsic reward for the next state 
                r_next_grid = get_int_rew(lstd_state['S'], feats, lstd_state['N']) # (N^2,)
                # Add absorbing terminal state reward (0.0)
                r_next_all = jnp.concatenate([r_next_grid, jnp.array([0.0])]) # (N^2 + 1,)
                # Project to state-action: sum_{s'} P(s' | s, a) * R(s')
                R_int_sa = jnp.einsum('sam, m -> sa', self.P, r_next_all) # (N^2 + 1, A)
                return R_int_sa
            
            # 3. Compute Intrinsic Reward Matrix
            R_int_sa = compute_intrinsic_rew_matrix()

            # 4. Solve True Values (Simultaneously)
            v_e_true, v_i_true = self.solve_true_intrinsic_and_extrinsic_value(pi.probs, Re=self.R_extrinsic, Ri=R_int_sa)
            
            # 5. Reshape everything to (N, N) grids
            v_e_grid, v_i_grid, v_net_grid = jax.tree.map(self.get_value_grid, (v_e_true, v_i_true, v_net))
            
            return v_e_grid, v_i_grid, v_net_grid