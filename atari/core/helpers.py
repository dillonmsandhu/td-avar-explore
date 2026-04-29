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
    rho_feats: jnp.ndarray       # Exploration/Intrinsic features
    next_rho_feats: jnp.ndarray  

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

def make_hold_decay_hold_schedule(
    total_updates: int, 
    max_beta: float = 2.0, 
    min_beta: float = 0.05, 
    hold_start_pct: float = 0.25, 
    decay_end_pct: float = 0.75
):
    """
    Piecewise schedule:
    1. Hold at `max_beta` from 0 to `hold_start_pct`.
    2. Linear decay from `max_beta` to `min_beta` between `hold_start_pct` and `decay_end_pct`.
    3. Hold at `min_beta` from `decay_end_pct` to 1.0.
    """
    def schedule(step):
        progress = step / total_updates
        
        # Calculate how far along the decay phase we are (from 0.0 to 1.0)
        # If progress < hold_start_pct, this is negative (gets clipped to 0.0)
        # If progress > decay_end_pct, this is > 1.0 (gets clipped to 1.0)
        decay_progress = (progress - hold_start_pct) / (decay_end_pct - hold_start_pct)
        decay_progress = jnp.clip(decay_progress, 0.0, 1.0)
        
        # Linearly interpolate based on the clipped decay progress
        current_beta = max_beta - decay_progress * (max_beta - min_beta)
        
        return current_beta
        
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
    minibatches = jax.tree_util.tree_map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
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

def _loss_fn_intrinsic_v(params, network, traj_batch, gae, targets, config):
    targets, i_targets = targets
    # RERUN NETWORK
    pi, value, i_val = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    
    # Extrinsic VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["VF_CLIP"], config["VF_CLIP"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    
    # Intrinsic VALUE LOSS
    value_pred_clipped = traj_batch.i_value + (
        i_val - traj_batch.i_value
    ).clip(-config["VF_CLIP"], config["VF_CLIP"])
    value_losses = jnp.square(i_val - i_targets)
    value_losses_clipped = jnp.square(value_pred_clipped - i_targets)
    i_value_loss = (
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
        + config["VF_COEF"] * i_value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (i_value_loss, value_loss, loss_actor, entropy)

# EXTRINSIC:
def calculate_gaeE(traj_batch, γ, λ,):
    # Extrinsic is always strictly episodic. Cut on death OR the dummy step.
    # (Since you precompute cut_i_trace, we can just quickly grab the extrinsic cuts here)
    done = traj_batch.done
    is_dummy = traj_batch.info.get('is_dummy', jnp.zeros_like(done))
    cut_e_trace = done | is_dummy
    cut_e_mult = 1.0 - cut_e_trace.astype(jnp.float32)

    # Package everything into a single tuple for the scan
    scan_inputs = (traj_batch, cut_e_mult)

    def _get_advantages(gae, inputs):
        transition, continue_e = inputs
        
        # --- Extrinsic ---
        delta = transition.reward + γ * transition.next_value * continue_e - transition.value
        gae = delta + (γ * λ * continue_e * gae)
        
        return gae, gae

    initial_acc = jnp.zeros_like(traj_batch.value[0])
    
    _, advantages = jax.lax.scan(
        _get_advantages, initial_acc, scan_inputs, reverse=True, unroll=16
    )
    
    return advantages, advantages + traj_batch.value


def _loss_fn_actor(params, network, traj_batch, gae, targets, config):
    # RERUN NETWORK
    pi = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
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

    total_loss = loss_actor- config["ENT_COEF"] * entropy
    
    return total_loss, (loss_actor, entropy)

# Running batch RMS:
def init_rms(shape):
    return {
        "mean": jnp.zeros(shape, dtype=jnp.float32),
        "var": jnp.ones(shape, dtype=jnp.float32),
        "count": jnp.array(1e-4, dtype=jnp.float32)
    }

def update_rms(rms_state, batch):
    """Batched update of running mean and variance."""
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - rms_state["mean"]
    tot_count = rms_state["count"] + batch_count

    new_mean = rms_state["mean"] + delta * batch_count / tot_count
    
    m_a = rms_state["var"] * rms_state["count"]
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * rms_state["count"] * batch_count / tot_count
    new_var = M2 / tot_count

    return {
        "mean": new_mean,
        "var": new_var,
        "count": tot_count
    }

def normalize_obs(rms_state, obs, clip=5.0):
    """Centers, scales, and clips the observation."""
    std = jnp.sqrt(rms_state["var"] + 1e-8)
    norm_obs = (obs - rms_state["mean"]) / std
    return jnp.clip(norm_obs, -clip, clip)