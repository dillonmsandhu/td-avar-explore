import jax
import jax.numpy as jnp
from typing import Tuple

class MountainCarExactValue:
    def __init__(self, pos_bins: int = 100, vel_bins: int = 100, gamma: float = 0.99, absorbing: bool = True, dense: bool = False):
        self.pos_bins = pos_bins
        self.vel_bins = vel_bins
        self.gamma = gamma
        self.absorbing = absorbing
        self.dense = dense
        
        self.num_grid_states = pos_bins * vel_bins
        self.num_actions = 3
        
        # Continuous space bounds
        self.pos_min = -1.2
        self.pos_max = 0.6
        self.vel_min = -0.07
        self.vel_max = 0.07
        
        # Pre-compute spacing for O(1) bin lookups
        self.pos_spacing = (self.pos_max - self.pos_min) / (self.pos_bins - 1)
        self.vel_spacing = (self.vel_max - self.vel_min) / (self.vel_bins - 1)
        
        self.pos_grid = jnp.linspace(self.pos_min, self.pos_max, self.pos_bins)
        self.vel_grid = jnp.linspace(self.vel_min, self.vel_max, self.vel_bins)
        
        # Fully JAX-compiled dynamics generation
        self.next_states, self.R_extrinsic, self.terminals = self._build_env_dynamics()

    def _get_bin_idx(self, val: jax.Array, min_val: float, spacing: float, num_bins: int) -> jax.Array:
        """O(1) calculation of the nearest discrete bin index."""
        idx = jnp.round((val - min_val) / spacing).astype(jnp.int32)
        return jnp.clip(idx, 0, num_bins - 1)

    def _build_env_dynamics(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Uses jax.vmap to compute the exact transition map without Python loops."""
        
        # Create grids of indices
        P_idx, V_idx, A_idx = jnp.meshgrid(
            jnp.arange(self.pos_bins), 
            jnp.arange(self.vel_bins), 
            jnp.arange(self.num_actions), 
            indexing='ij'
        )
        
        # Flatten for vmap
        flat_p_i = P_idx.flatten()
        flat_v_i = V_idx.flatten()
        flat_a = A_idx.flatten()
        
        def step_fn(p_i, v_i, a):
            # 1. Look up physical values
            p = self.pos_grid[p_i]
            v = self.vel_grid[v_i]
            
            # 2. Exact Mountain Car Dynamics
            v_next = v + (a - 1) * 0.001 - jnp.cos(3 * p) * 0.0025
            v_next = jnp.clip(v_next, self.vel_min, self.vel_max)
            
            p_next = p + v_next
            p_next = jnp.clip(p_next, self.pos_min, self.pos_max)
            
            # Left wall crash
            v_next = jnp.where(jnp.logical_and(p_next == self.pos_min, v_next < 0), 0.0, v_next)
            
            # 3. Find target bins arithmetically
            p_next_i = self._get_bin_idx(p_next, self.pos_min, self.pos_spacing, self.pos_bins)
            v_next_i = self._get_bin_idx(v_next, self.vel_min, self.vel_spacing, self.vel_bins)
            
            next_s_idx = p_next_i * self.vel_bins + v_next_i
            curr_s_idx = p_i * self.vel_bins + v_i
            
            reward = -1.0
            terminal = jnp.logical_and(p_next >= 0.5, v_next >= 0.0)
            
            # 4. Handle Terminal & Absorbing logic
            is_absorbing = jnp.logical_and(terminal, self.absorbing)
            next_s_idx = jnp.where(is_absorbing, curr_s_idx, next_s_idx)
            reward = jnp.where(is_absorbing, 0.0, reward)
            
            return next_s_idx, reward, terminal

        # Vectorize the step function and execute in parallel
        vmap_step = jax.vmap(step_fn)
        next_s_flat, r_flat, t_flat = vmap_step(flat_p_i, flat_v_i, flat_a)
        
        # Reshape back to (num_states, num_actions)
        next_states = next_s_flat.reshape((self.num_grid_states, self.num_actions))
        rewards = r_flat.reshape((self.num_grid_states, self.num_actions))
        terminals = t_flat.reshape((self.num_grid_states, self.num_actions))
        
        return next_states, rewards, terminals

    def compute_optimal_values(self, R_int_sa: jax.Array = None) -> Tuple[jax.Array, jax.Array]:
        """Runs Value Iteration inside JAX using lax.while_loop."""
        def solve_v_star(R_sa):
            def body_fn(val):
                V, _, i = val
                
                # Integer indexing for fast transitions: V[next_states]
                V_next = jnp.where(self.terminals & (~self.absorbing), 0.0, V[self.next_states])
                Q = R_sa + self.gamma * V_next
                V_new = jnp.max(Q, axis=1)
                
                delta = jnp.max(jnp.abs(V - V_new))
                return V_new, delta, i + 1

            def cond_fn(val):
                _, delta, i = val
                # Continue until convergence or 2000 iterations
                return jnp.logical_and(delta > 1e-5, i < 2000)

            V_init = jnp.zeros(self.num_grid_states)
            V_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (V_init, 1e6, 0))
            return V_final.reshape((self.pos_bins, self.vel_bins))

        # Extrinsic Logic (+1 density adjustment)
        Re = self.R_extrinsic + 1.0 if self.dense else self.R_extrinsic
        
        v_e_star_grid = solve_v_star(Re)
        v_i_star_grid = solve_v_star(R_int_sa) if R_int_sa is not None else jnp.zeros_like(v_e_star_grid)

        return v_e_star_grid, v_i_star_grid

    def get_observation_stack(self) -> jax.Array:
        """Returns the (x, v) coordinate pair for every state to feed to the network."""
        P_idx, V_idx = jnp.meshgrid(jnp.arange(self.pos_bins), jnp.arange(self.vel_bins), indexing='ij')
        P = self.pos_grid[P_idx]
        V = self.vel_grid[V_idx]
        return jnp.stack([P.flatten(), V.flatten()], axis=-1)
