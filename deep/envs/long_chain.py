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
    max_steps_in_episode: int = 1e6


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
        done_steps = state.time >= params.max_steps_in_episode
        done_goal = (state.pos == self.chain_length - 1)
        return jnp.logical_or(done_goal, done_steps)

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
                "time": spaces.Discrete(params.max_steps_in_episode),
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
    def __init__(self, chain_length: int, gamma: float = 0.99, episodic: bool = False):
        """
        Args:
            chain_length (N): The total length defined in the Env. 
                              States are 0..N-1.
                              Playable states are 0..N-2.
                              State N-1 is the Terminal Goal (never seen).
        """
        self.N = chain_length
        self.num_playable = chain_length - 1  # We only solve for 0..N-2
        self.gamma = gamma
        self.episodic = episodic
        
        # 1. Obs Stack: Only for Playable States (0 to N-2)
        # Note: We keep num_classes=N to match the network's input shape, 
        # even though index N-1 is never active.
        self.obs_stack = jax.nn.one_hot(jnp.arange(self.num_playable), self.N)
        self.reachable_mask = jnp.ones(self.num_playable)
        # 2. Dynamics (Square matrices of size N-1)
        self.P, self.R_extrinsic = self._build_env_dynamics()
        self.P_cont = self._build_env_dynamics_continuing()

    def _build_env_dynamics(self):
        """
        Standard Episodic Dynamics.
        Transition off the end of the chain -> Terminal (Exit the system).
        """
        num_S = self.num_playable
        P = np.zeros((num_S, 2, num_S))
        R = np.zeros((num_S, 2))
        
        for s in range(num_S):
            # --- Action 0: Left ---
            P[s, 0, max(0, s - 1)] = 1.0

            # --- Action 1: Right ---
            next_s = s + 1
            
            if next_s == num_S: 
                # We are at s = N-2, moving Right to N-1 (Goal).
                # In Episodic P matrix: This mass leaves the system.
                # Row remains all zeros for this action.
                # Reward is received.
                R[s, 1] = 1.0
            else:
                # Standard transition
                P[s, 1, next_s] = 1.0
                
        return jnp.array(P), jnp.array(R)

    def _build_env_dynamics_continuing(self):
        """
        Continuing Dynamics (LSTD/Optimism).
        Transition off the end of the chain -> Reset to State 0.
        """
        num_S = self.num_playable
        P = np.zeros((num_S, 2, num_S))
        
        for s in range(num_S):
            # --- Action 0: Left ---
            P[s, 0, max(0, s - 1)] = 1.0

            # --- Action 1: Right ---
            next_s = s + 1
            
            if next_s == num_S:
                # We are at s = N-2, moving Right to N-1 (Goal).
                # In Continuing P matrix: We loop back to Start.
                P[s, 1, 0] = 1.0
            else:
                P[s, 1, next_s] = 1.0
            
        return jnp.array(P)

    def solve_linear_system(self, pi, P_env, R_env):
        # pi: (N-1, 2)
        # P_env: (N-1, 2, N-1)
        # R_env: (N-1, 2) (Expected immediate reward)
        
        # P_pi[s, s'] = sum_a pi(a|s) * P(s'|s,a)
        P_pi = jnp.einsum('sa, sam -> sm', pi, P_env)
        
        # R_pi[s] = sum_a pi(a|s) * R(s,a)
        R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        
        I = jnp.eye(self.num_playable)
        
        # Bellman: (I - gamma * P) * V = R
        A = I - self.gamma * P_pi
        
        return jnp.linalg.solve(A, R_pi)

    def compute_true_values(self, network, params, get_int_rew_per_state):
        # 1. Forward Pass on Playable States Only
        out = network.apply(params, self.obs_stack)
        
        if len(out) == 3:
            pi_dist, v_net_ext, v_net_int = out
            v_net_tuple = (v_net_ext.squeeze(), v_net_int.squeeze())
        else:
            pi_dist, v_net_all = out
            v_net_tuple = v_net_all.squeeze()
            
        pi_matrix = pi_dist.probs # (N-1, 2)

        # 2. Intrinsic Reward
        # r(s) for playable states. Shape (N-1,)
        r_int_s = get_int_rew_per_state(self.obs_stack)
        
        # 3. Project to R(s,a)
        # We need to be careful here. 
        # The intrinsic reward vector r_int_s corresponds to states 0..N-2.
        #
        # If we loop (Continuing), we land in 0..N-2.
        # If we terminate (Episodic), we land in Void. 
        #
        # For the "Void" transition, the intrinsic reward is effectively 0 
        # (or whatever reward you assign to the terminal state, usually 0).
        # Since r_int_s doesn't have an entry for "Void", normal multiplication works 
        # provided P columns match r_int_s indices.
        
        target_P = self.P if self.episodic else self.P_cont
        R_int_sa = jnp.einsum('sam, m -> sa', target_P, r_int_s)
        
        # Note: If P has a transition to "Void" (all zeros row), 
        # the einsum sums to 0, correctly implying 0 intrinsic reward for the next state.

        # 4. Solve
        v_e = self.solve_linear_system(pi_matrix, self.P, self.R_extrinsic)
        v_i = self.solve_linear_system(pi_matrix, target_P, R_int_sa)

        return v_e, v_i, v_net_tuple
    
    def get_value_grid(self, x: jax.Array) -> jax.Array:
        """Identity"""
        return x
    

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
