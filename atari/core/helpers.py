import envpool
from core.wrappers import JaxEnvPoolWrapper
from core.imports import *

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: dict
    # --- NEW FIELDS ---
    phi: jnp.ndarray            # LSTD features
    next_phi: jnp.ndarray 
    rho_feat: jnp.ndarray       # Exploration/Intrinsic features
    next_rho_feat: jnp.ndarray  

def make_env(config):
    env = envpool.make(
        config["ENV_NAME"],
        env_type="gym",
        num_envs=config["NUM_ENVS"],
        seed=config["SEED"],
        num_threads=config.get("THREADS", 1),
        **config.get("ENV_KWARGS", {}),
    )
    
    # Expose necessary standard attributes for your agent
    env.num_envs = config["NUM_ENVS"]
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = config["ENV_NAME"]
    
    # Wrap with our unified XLA logic
    env = JaxEnvPoolWrapper(env, config)
    
    return env

def get_scale_free_bonus(S_inv, features):
    """bonus = sqrt(x^T Σ^{-1} x)"""
    bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
    return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

def update_cov(sigma_state, phi):
    S = sigma_state['S']
    S_batch_sum = jnp.einsum("tni, tnj -> ij", phi, phi)
    S_new = S + S_batch_sum
    S_new = 0.5 * (S_new + S_new.T)
    return {'S': S_new, }

def calculate_gae(
    traj_batch, 
    γ, λ, 
    cut_i_trace,  # Precomputed boolean array (T, B)
    absorb_mask,  # Precomputed boolean array (T, B)
    γi=None, λi=None
):
    if γi is None: γi = γ
    if λi is None: λi = λ

    # Extrinsic is always strictly episodic. Cut on death OR the dummy step.
    # (Since you precompute cut_i_trace, we can just quickly grab the extrinsic cuts here)
    done = traj_batch.done
    is_dummy = traj_batch.info.get('is_dummy', jnp.zeros_like(done))
    cut_e_trace = done | is_dummy

    # Convert cut masks to float multipliers for clean math inside the scan
    cut_e_mult = 1.0 - cut_e_trace.astype(jnp.float32)
    cut_i_mult = 1.0 - cut_i_trace.astype(jnp.float32)

    # Package everything into a single tuple for the scan
    scan_inputs = (traj_batch, cut_e_mult, cut_i_mult, absorb_mask)

    def _get_advantages(gae_accs, inputs):
        gae, i_gae = gae_accs
        transition, continue_e, continue_i, absorb = inputs
        
        # --- Extrinsic ---
        delta = transition.reward + γ * transition.next_value * continue_e - transition.value
        gae = delta + (γ * λ * continue_e * gae)
        
        # --- Intrinsic --- 
        # If absorb is True, override the next value with the infinite sum of the intrinsic reward
        target_i_val = jnp.where(absorb, 
                            transition.intrinsic_reward / (1.0 - γi), 
                            transition.next_i_val)

        i_delta = transition.intrinsic_reward + γi * target_i_val * continue_i - transition.i_value 
        i_gae = i_delta + (γi * λi * continue_i * i_gae)
        
        return (gae, i_gae), (gae, i_gae)

    initial_accs = (jnp.zeros_like(traj_batch.value[0]), jnp.zeros_like(traj_batch.i_value[0]))
    
    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages, initial_accs, scan_inputs, reverse=True, unroll=16
    )
    
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

def make_triangle_schedule(total_updates: int, max_beta: float, peak_at: float = 0.05):
    """
    Piecewise linear: 0 -> max_beta (at peak_at) -> 0 (at total_updates).
    """
    def schedule(step):
        progress = step / total_updates
        
        # Use jax.lax.select for JIT-compatible branching
        multiplier = jax.lax.select(
            progress < peak_at,
            # Ramp up: line from (0,0) to (peak_at, 1)
            progress / peak_at,
            # Ramp down: line from (peak_at, 1) to (1, 0)
            (1.0 - progress) / (1.0 - peak_at)
        )
        
        return max_beta * jnp.clip(multiplier, 0.0, 1.0)
        
    return schedule

# FOR LSPI:
def expand_to_sa_features(phi_s, n_actions, taken_actions, dim_kA):
    one_hots = jax.nn.one_hot(taken_actions, n_actions)  
    phi_sa_unflattened = phi_s[..., None, :] * one_hots[..., :, None]
    return phi_sa_unflattened.reshape(*phi_s.shape[:-1], dim_kA)

def expected_next_sa_features(next_phi, Pi, dim_kA):
    expected_next_sa = next_phi[..., None, :] * Pi[..., :, None]
    return expected_next_sa.reshape(*next_phi.shape[:-1], dim_kA)

# For network training:
def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches


def _loss_fn(params, network, traj_batch, gae, targets, config):
    # RERUN NETWORK
    pi, value = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    
    # VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["VF_CLIP"], config["VF_CLIP"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


def calculate_traces(features, cut_trace, γ, λ):
    """
    Unified trace calculation supporting all configurations cleanly.
    Input shapes:
        features: (T, B, k)
        cut_trace: (T, B) - The precomputed boolean mask
    Returns:
        traces: (T, B, k)
    """
    def _step_trace(trace_prev, scan_inputs):
        phi, cut = scan_inputs
        
        # 1. Calculate current trace (has history!)
        trace_current = trace_prev * γ * λ + phi
        
        # 2. Determine history for the NEXT step
        # If cut is True, trace_mult is 0.0, completely severing the link to the next step
        trace_mult = 1.0 - cut.astype(jnp.float32)
        trace_next = trace_current * trace_mult[..., None] 
        
        return trace_next, trace_current

    # Scan over the time dimension (T)
    _, traces = jax.lax.scan(
        _step_trace, 
        jnp.zeros_like(features[0]), 
        (features, cut_trace)
    )
    
    return traces