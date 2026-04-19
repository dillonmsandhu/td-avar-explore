"""JAX implementation of Long Chain environment for Intrinsic Value experiments."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct
from gymnax.environments import environment, spaces
import numpy as np
from typing import Any, Tuple, Callable, Optional
PyTree = Any


@struct.dataclass
class EnvState(environment.EnvState):
    pos: int  
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    fail_prob: float = 0.0
    resample_init_pos: bool = False
    max_steps_in_episode: int = int(1e6) # Ensure this is > chain_length


class LongChain(environment.Environment[EnvState, EnvParams]):
    """
    A 1D Chain environment designed to test Intrinsic Value Functions.
    """

    def __init__(self, chain_length: int = 200):
        super().__init__()
        self.chain_length = chain_length
        self.action_set = jnp.array([-1, 1])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        
        key_random, key_action = jax.random.split(key)
        
        choose_random = jax.random.uniform(key_random, ()) < params.fail_prob
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )

        direction = self.action_set[action]
        next_pos = state.pos + direction
        
        # Use self.chain_length (Static) instead of params.chain_length
        next_pos = jnp.clip(next_pos, 0, self.chain_length - 1)

        reached_goal = (next_pos == self.chain_length - 1)
        reward = reached_goal.astype(jnp.float32)

        state = EnvState(pos=next_pos, time=state.time + 1)
        
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        pos_fixed = 0
        
        # Use self.chain_length (Static)
        pos_random = jax.random.randint(key, (), 0, self.chain_length)
        
        pos = jax.lax.select(params.resample_init_pos, pos_random, pos_fixed)
        
        state = EnvState(pos=pos, time=0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """
        Return One-Hot observation.
        """
        return jax.nn.one_hot(state.pos, num_classes=self.chain_length)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_goal = (state.pos == self.chain_length - 1)
        return done_goal
        

    @property
    def name(self) -> str:
        return "LongChain-v0"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # Use self.chain_length (Static)
        return spaces.Box(0, 1, (self.chain_length,), jnp.float32)
    
    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "pos": spaces.Discrete(self.chain_length),
                "time": spaces.Discrete(1e8),
            }
        )

    def render(self, state: EnvState, params: EnvParams):
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot([0, self.chain_length-1], [0, 0], 'k-', lw=2, alpha=0.5)
        ax.scatter([0], [0], c='blue', s=100, label='Start')
        ax.scatter([self.chain_length-1], [0], c='green', s=100, label='Goal')
        ax.scatter([state.pos], [0], c='red', s=200, marker='*', label='Agent')
        ax.set_yticks([])
        ax.set_xlim(-5, self.chain_length + 5)
        ax.legend()
        ax.set_title(f"Long Chain (L={self.chain_length}) | Step: {state.time}")
        return fig, ax

class LongChainExactValue:
    def __init__(self, chain_length: int, gamma: float = 0.99, episodic: bool = False, absorbing: bool = False):
        self.N = chain_length
        self.num_playable = chain_length - 1  
        self.num_total_states = chain_length  # Explicitly track terminal
        self.terminal_idx = chain_length - 1
        self.gamma = gamma
        self.episodic = episodic
        self.absorbing = absorbing 
        
        # Obs Stack covers ALL states (including terminal) for exact RND evaluation
        self.obs_stack = jax.nn.one_hot(jnp.arange(self.num_total_states), self.N)
        self.reachable_mask = jnp.ones(self.num_playable)
        
        # Dynamics (Square matrices of size N)
        self.P, self.R_extrinsic = self._build_env_dynamics()
        self.P_cont = self._build_env_dynamics_continuing()

    def _build_env_dynamics(self):
        """Episodic/Absorbing Dynamics. End of chain -> Terminal Self-Loop."""
        num_S = self.num_total_states
        P = np.zeros((num_S, 2, num_S))
        R = np.zeros((num_S, 2))
        
        for s in range(self.num_playable):
            P[s, 0, max(0, s - 1)] = 1.0 # Left
            
            next_s = s + 1
            P[s, 1, next_s] = 1.0        # Right
            if next_s == self.terminal_idx:
                R[s, 1] = 1.0            # Extrinsic reward for reaching the end
                
        # Terminal self loop
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        return jnp.array(P), jnp.array(R)

    def _build_env_dynamics_continuing(self):
        """Continuing Dynamics. End of chain -> Reset to 0."""
        num_S = self.num_total_states
        P = np.zeros((num_S, 2, num_S))
        
        for s in range(self.num_playable):
            P[s, 0, max(0, s - 1)] = 1.0
            
            next_s = s + 1
            if next_s == self.terminal_idx:
                P[s, 1, 0] = 1.0  # Teleport to start
            else:
                P[s, 1, next_s] = 1.0
                
        P[self.terminal_idx, :, self.terminal_idx] = 1.0 # Dummy loop for shape stability
        return jnp.array(P)

    def solve_linear_system(self, pi, P_env, R_env):
        P_pi = jnp.einsum('sa, sam -> sm', pi, P_env)
        R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def compute_true_values(self, network, params, get_int_rew_per_state, all=False):
        out = network.apply(params, self.obs_stack)
        
        if len(out) == 3:
            pi_dist, v_net_ext, v_net_int = out
            v_net_tuple = (self.get_value_grid(v_net_ext.squeeze(), all), self.get_value_grid(v_net_int.squeeze(), all))
        else:
            pi_dist, v_net_all = out
            v_net_tuple = self.get_value_grid(v_net_all.squeeze(), all)
            
        pi_matrix = pi_dist.probs # (N, 2)

        # 1. Get Intrinsic Rewards for ALL states
        r_int_s = get_int_rew_per_state(self.obs_stack)
        
        # 2. Unified Episodic vs Absorbing switch
        if self.episodic and not self.absorbing:
            r_int_s = r_int_s.at[self.terminal_idx].set(0.0)
        
        # 3. Choose Dynamics
        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)
        
        # 4. Solve
        v_e = self.solve_linear_system(pi_matrix, self.P, self.R_extrinsic)
        v_i = self.solve_linear_system(pi_matrix, target_P, R_int_sa)

        # Slice off the terminal state before returning
        return self.get_value_grid(v_e, all), self.get_value_grid(v_i, all), v_net_tuple
    
    def get_value_grid(self, x: jax.Array, all=False) -> jax.Array:
        """Slices off the terminal state to protect downstream logging."""
        if not all:
            return x[:self.num_playable]    
        else: return x 

    def compute_optimal_intrinsic_values(self, network, params, get_int_rew_per_state, all=False):
        # 1. Forward Pass (maintain identical output signature)
        out = network.apply(params, self.obs_stack)
        
        if len(out) == 3:
            pi_dist, v_net_ext, v_net_int = out
            v_net_tuple = (
                self.get_value_grid(v_net_ext.squeeze(), all), 
                self.get_value_grid(v_net_int.squeeze(), all)
            )
        else:
            pi_dist, v_net_all = out
            v_net_tuple = self.get_value_grid(v_net_all.squeeze(), all)

        # 2. Extract and Process Intrinsic Rewards
        r_int_s = get_int_rew_per_state(self.obs_stack)
        
        if self.episodic and not self.absorbing:
            r_int_s = r_int_s.at[self.terminal_idx].set(0.0)
        
        # Determine transition dynamics
        target_P = self.P if (self.episodic or self.absorbing) else self.P_cont
        R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)

        # 3. Value Iteration for Optimal Intrinsic Value (V*i)
        def value_iteration(P, R, gamma, eps=1e-8):
            # We define the Bellman Optimality Operator as a pure JAX function
            def bellman_step(v_curr):
                # Q(s, a) = R(s, a) + gamma * sum_{s'} P(s'|s, a) V(s')
                q = R + gamma * jnp.einsum("sam, m -> sa", P, v_curr)
                return jnp.max(q, axis=1)

            # Option A: Fixed-length loop (Highly recommended for JIT stability)
            # 1000 iterations is a safe upper bound for gamma=0.99 convergence
            v_init = jnp.zeros(self.N)
            v_final = jax.lax.fori_loop(0, 1000, lambda i, v: bellman_step(v), v_init)            
            return v_final

        v_i_star = value_iteration(target_P, R_int_sa, self.gamma)

        # 4. Value Iteration for Optimal Extrinsic Value (V*e)
        v_e_star = value_iteration(self.P, self.R_extrinsic, self.gamma)

        return (
            self.get_value_grid(v_e_star, all), 
            self.get_value_grid(v_i_star, all), 
            v_net_tuple
        )

    def plot(self, v_e, v_i, v_pred_tuple):
        """ Visualizes the Value Functions along the 1D Chain.
            v_pred_tuple is assumed to be (v_e, v_i)
        """
        x = np.arange(self.N)
        
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        
        # Plot 1: True Values
        axes[0].plot(x, v_e, label='V_ext (True)', color='black', linestyle='--')
        axes[0].set_title(f"Ground Truths (Episodic={self.episodic})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, v_i, label='V_int (True)', color='blue')
        axes[1].set_title(f"Ground Truths (Episodic={self.episodic})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 2: Network Predictions vs Truth
        if isinstance(v_pred_tuple, tuple):
            v_net_ext, v_net_int = v_pred_tuple
            axes[2].plot(x, v_i, label='V_int (True)', color='blue', alpha=0.5)
            axes[2].plot(x, v_net_int, label='V_int (Pred)', color='red')
            axes[2].set_title("Intrinsic Value: Truth vs Net")
        else:
            axes[2].plot(x, v_pred_tuple, label='V_Net', color='green')
            axes[2].set_title("Value Prediction")
            
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 3: The "Stitching Gap" (Error)
        if isinstance(v_pred_tuple, tuple):
            v_net_ext, v_net_int = v_pred_tuple
            error = v_i - v_net_int
            axes[3].fill_between(x, error, color='red', alpha=0.3)
            axes[3].plot(x, error, color='red')
            axes[3].set_title("Prediction Error (Underestimation)")
            axes[3].set_ylim(bottom=0) # We care mostly about underestimation
            
        plt.tight_layout()
        return fig

