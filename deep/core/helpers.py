# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
from envs.sparse_mc import SparseMountainCar
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.long_chain import LongChain
from envs.fourrooms_custom import FourRooms
from envs.wrappers import NormalizeObservationWrapper, NormalizeRewardWrapper, AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, TerminalInfoWrapper
from gymnax.environments import spaces

def load_config(args):
    import core.configs as configs
    from core.utils import parse_config_override
    # 1. Look up registry by the --base-config flag
    registry_item = configs.CONFIG_REGISTRY.get(args.base_config)
    
    if registry_item:
        config = registry_item["config_dict"].copy()
    else:
        # Fallback to shared if the name isn't in the registry
        config = configs.shared.copy()

    # 2. Apply command-line JSON overrides
    if args.config:
        config_override = parse_config_override(args.config)
        config.update(config_override)
        
    return config

def make_env(config):
    
    if config['ENV_NAME'] ==  "FourRoomsCustom-v0":
        env = FourRooms(
            N=int(config.get("FOURROOMS_SIZE", 21)),
            use_visual_obs=(config["NETWORK_TYPE"] == "cnn"),
        )
        env_params = env.default_params.replace(
            fail_prob=float(config.get("FOURROOMS_FAIL_PROB", env.default_params.fail_prob)),
            resample_init_pos=bool(config.get("FOURROOMS_RESAMPLE_INIT_POS", env.default_params.resample_init_pos)),
            resample_goal_pos=bool(config.get("FOURROOMS_RESAMPLE_GOAL_POS", env.default_params.resample_goal_pos)),
            max_steps_in_episode=int(config.get("FOURROOMS_MAX_STEPS", env.default_params.max_steps_in_episode)),
        )
    elif config["ENV_NAME"] in {"FourRooms-misc"}:
        env = FourRooms(
            N=int(config.get("FOURROOMS_SIZE", 13)),
            use_visual_obs=(config["NETWORK_TYPE"] == "cnn"),
        )
        env_params = env.default_params.replace(
            fail_prob=float(config.get("FOURROOMS_FAIL_PROB", env.default_params.fail_prob)),
            resample_init_pos=bool(config.get("FOURROOMS_RESAMPLE_INIT_POS", env.default_params.resample_init_pos)),
            resample_goal_pos=bool(config.get("FOURROOMS_RESAMPLE_GOAL_POS", env.default_params.resample_goal_pos)),
            max_steps_in_episode=int(config.get("FOURROOMS_MAX_STEPS", env.default_params.max_steps_in_episode)),
        )
    
    elif config['ENV_NAME'] == "Chain":
        env = LongChain(config.get('CHAIN_LENGTH', 100))
        env_params = env.default_params.replace(
            fail_prob=float(config.get("CHAIN_FAIL_PROB", env.default_params.fail_prob)),
            resample_init_pos=bool(config.get("CHAIN_RESAMPLE_INIT_POS", env.default_params.resample_init_pos)),
            max_steps_in_episode=int(config.get("CHAIN_MAX_STEPS", env.default_params.max_steps_in_episode)),
        )

    elif config['ENV_NAME'] == "SparseMountainCar-v0":
        env = SparseMountainCar()
        env_params = env.default_params
    
    elif config['ENV_NAME'] == 'DeepSea-bsuite':
        env, env_params = gymnax.make(config["ENV_NAME"], size = config.get("DEEPSEA_SIZE", 10))
    
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
        print('Env:', config['ENV_NAME'])
        print('Network:', config['NETWORK_TYPE'])
        print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    env = TerminalInfoWrapper(env) # adds the terminal state to info.
    env = LogWrapper(env)
    
    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env) # Ensures sampled actions are within [low, high]
    
    if config["NETWORK_TYPE"] == "mlp":
        env = FlattenObservationWrapper(env)
    if config["NETWORK_TYPE"] == "cnn":
        if len(env.observation_space(env_params).shape) < 3:
            env = AddChannelWrapper(env) # add an empty channel to the end if 2d input
    if config["NORMALIZE_REWARDS"]:
        env = NormalizeRewardWrapper(env, gamma=config["GAMMA"]) 
    if config["NORMALIZE_OBS"]:
        env = NormalizeObservationWrapper(env) 
    
    
    print('Obs Shape:', env.observation_space(env_params).shape)
    print('Action Shape:', env.action_space(env_params).shape)
    return env, env_params

def cross_cov(traces, current_features, next_features, done, γ):
    "One a sample of the LSTD A matrix - episodic"
    td_features = current_features - γ  * next_features * (1-done)
    A_sample = jnp.outer(traces, td_features)
    return A_sample

def cross_cov_continuing(traces, current_features, next_features, done, γ):
    "One sample of the LSTD A matrix - continuing"
    td_features = current_features - γ  * next_features
    A_sample = jnp.outer(traces, td_features)
    return A_sample

def cosine_similarity(a, b):
    dot = jnp.dot(a,b)
    mag = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return dot/mag

def sigma_update(   sigma_state: Dict,
                    transitions, # Explore_Transition
                    features: jnp.ndarray,
                    α: float,
                    
    ):
    S, t = sigma_state['S'], sigma_state['t']
    batch_axes = tuple(range(transitions.done.ndim))
    N = transitions.done.size + sigma_state['N']  # total number of samples seen so far
    S_update = jax.vmap(jax.vmap(lambda x: jnp.outer(x,x)))(features) # (L, B, k, k)
    S_b = S_update.mean(axis=batch_axes) # Batch average
    S = (1-α) * S + α * S_b # EMA
    S = 0.5 * (S+ S.T) # symmetrize
    return {'S': S, 'N': N, 't': t+1} # new sigma_state

def EMA(coeff, x_old, x_new):
    return (1 - coeff) * x_old + coeff * x_new

def sigma_update_masked(
    sigma_state: Dict,
    features: jnp.ndarray,  # Shape: (..., k)
    mask: jnp.ndarray,      # Shape: (...) matching the batch dimensions of features
    α: float,
):
    """
    Masks out those that are included twice
    Takes a mask that corresponds to what included feature vectors are valid for the update.
    """
    S = sigma_state['S']
    S_update = jnp.einsum("...i, ...j -> ...ij", features, features)
    
    # 2. Apply Mask
    # Expand mask to (..., 1, 1) so it broadcasts over the (k, k) matrix dimensions
    # Zeros out the outer products corresponding to invalid/padding states
    S_masked = S_update * mask[..., None, None]
    
    # 3. Compute Weighted Mean
    batch_axes = tuple(range(mask.ndim))
    total_valid = jnp.sum(mask)
    S_batch_mean = jnp.sum(S_masked, axis=batch_axes) / total_valid
    
    # 4. Update & Force Symmetry
    S_new = EMA(α, S, S_batch_mean)
    S_new = 0.5 * (S_new + S_new.T)
    
    # 5. Update N
    N_new = sigma_state['N'] + jnp.sum(mask)
    
    return {
        'S': S_new, 
        'N': N_new, 
        't': sigma_state['t'] + 1
    }

def _get_all_traces(traj_batch, features, γ, λ):
    """Get all traces for a batch of trajectories.
    Returns: L x B x k
    """
    def get_lambda_traces(phis_s, dones, γ, λ,):
        # We need to manage the carry (prev trace) separate from current trace output
        def _step_trace(trace_prev, inputs):
            phi, done = inputs
            trace_current = trace_prev * γ * λ + phi
            trace_next = trace_current * (1.0 - done)
            # Return: (carry, out)
            return trace_next, trace_current
        init_traces = jnp.zeros_like(phis_s[0]) 
        _, traces = jax.lax.scan(_step_trace, init_traces, (phis_s, dones))
        return traces 
    # Fix: Ensure dones are passed correctly to the inner function
    traces = jax.vmap(get_lambda_traces, in_axes=(1, 1, None, None))(
        features, traj_batch.done, γ, λ
    )
    return traces.transpose(1,0,2)

# def _get_all_traces(traj_batch, features, γ, λ):
#     """Get all traces for a batch of trajectories.
#     Returns: L x B x k, where B is batch size, L is trajectory length, and k is number of features.
#     """
#     def get_lambda_traces(phis_s, traj_batch, γ, λ):
#         def _step_trace(trace, inputs):
#             phi, done = inputs
#             trace = trace * (1-done) * γ * λ + phi
#             return trace, trace
#         # end step_trace
#         init_traces = jnp.zeros_like(phis_s[-1])  # (k,)
#         _, traces = jax.lax.scan(_step_trace, init_traces, (phis_s, traj_batch.done))
#         return traces # L x k
#     # end get_lambda_traces
#     traces = jax.vmap(get_lambda_traces, in_axes=(1, 1, None, None))(features, traj_batch, γ, λ)
#     return traces.transpose(1,0,2) # L x B x k (vmap puts batch axis (B) first)

def _get_all_traces_continuing(traj_batch, features, γ, λ):
    """Get all traces for a batch of trajectories.
    Returns: L x B x k
    """
    def get_lambda_traces(phis_s, γ, λ,):
        # We need to manage the carry (prev trace) separate from current trace output
        def _step_trace(trace_prev, phi):
            trace = trace_prev * γ * λ + phi
            return trace, trace
        
        init_traces = jnp.zeros_like(phis_s[0]) 
        _, traces = jax.lax.scan(_step_trace, init_traces, phis_s)
        return traces 
    # Fix: Ensure dones are passed correctly to the inner function
    traces = jax.vmap(get_lambda_traces, in_axes=(1, None, None))(
        features, γ, λ
    )
    return traces.transpose(1,0,2)

def _calculate_gae(traj_batch, last_val, γ, λ):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + γ * next_value * (1 - done) - value
        gae = (
            delta
            + γ * λ * (1 - done) * gae
        )
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


class Explore_Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    i_value: jnp.ndarray # extra - intrinsic value from second value head
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray # extra - intrinsic reward (RND loss)
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray # extra - to get next features
    embedding: jnp.ndarray # extra - target embedding from target rnd network
    td_error: jnp.ndarray # for OPG (outer product of gradients), the "meat" of the sandwich covariance
    info: jnp.ndarray

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class FullTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    

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

def calculate_gae_intrinsic_and_extrinsic(traj_batch, last_val, last_i_val, γ, λ, γi=None, λi=None):
    """Continuing Intrinsic TD Target"""
    if γi==None:
        γi = γ
    
    if λi==None:
        λi = λ

    def _get_advantages(gae_and_next_value, transition):
        gae, i_gae, next_value, i_next_value = gae_and_next_value
        done, value, reward, i, i_value = (
            transition.done,
            transition.value,
            transition.reward,
            transition.intrinsic_reward,
            transition.i_value,
        )
        
        delta = reward + γ * next_value * (1 - done) - value
        gae = delta + (γ * λ * (1 - done) * gae)
        
        # Intrinsic is non-episodic (no done masking)
        i_delta = i + γi * i_next_value - i_value 
        i_gae = i_delta + (γi * λi * i_gae)
        
        return (gae, i_gae, value, i_value), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

def calculate_gae_intrinsic_and_extrinsic_episodic(traj_batch, last_val, last_i_val, γ, λ, λi=None, γi=None):
    """Episodic Intrinsic TD Target"""
    if λi is None:
        λi = λ 
    if γi is None:
        γi = γ

    def _get_advantages(gae_and_next_value, transition):
        gae, i_gae, next_value, i_next_value = gae_and_next_value
        done, value, reward, i, i_value = (
            transition.done,
            transition.value,
            transition.reward,
            transition.intrinsic_reward,
            transition.i_value,
        )

        not_done = (1-done)
        
        delta = reward + γ * next_value * not_done - value
        gae = delta + (γ * λ * not_done * gae)
        
        # Intrinsic is non-episodic (no done masking)
        i_delta = i + γi * not_done * i_next_value - i_value 
        i_gae = i_delta + (γi * λi * not_done * i_gae )
        
        return (gae, i_gae, value, i_value), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

def calculate_i_and_e_gae_two_critic(traj_batch, last_val, last_i_val_fast, γ, λ, γi = None, λi=None):
    """
    Continuing Intrinsic TD Target using two intrinsic critics: fast (for TD(λ)) and slow (for baseline)
    A = Q_fast - V_slow
    Value Target = Q_fast
    """
    if λi is None:
        λi = λ 

    if γi is None:
        γi = γ

    def _get_advantages(gae_and_next_value, transition):
        gae, i_gae, next_value, i_next_value = gae_and_next_value
        done, value, reward, i, i_value_fast, i_value_slow = (
            transition.done,
            transition.value,
            transition.reward,
            transition.intrinsic_reward,
            transition.i_value_fast,
            transition.i_value_slow,
        )
        
        delta = reward + γ * next_value * (1 - done) - value
        gae = delta + (γ * λ * (1 - done) * gae)
        
        # Intrinsic is non-episodic (no done masking)
        i_delta = i + γi * i_next_value - i_value_slow 
        i_gae = i_delta + (γi * λi * i_gae)
        
        return (gae, i_gae, value, i_value_fast), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val_fast),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value_slow)

def calculate_i_and_e_gae_two_critic_episodic(traj_batch, last_val, last_i_val_fast, γ, λ, γi=None, λi=None):
    """
    Continuing Intrinsic TD Target using two intrinsic critics: fast (for TD(λ)) and slow (for baseline)
    A = Q_fast - V_slow
    Value Target = Q_fast
    """
    if λi is None:
        λi = λ 

    if γi is None:
        γi = γ

    def _get_advantages(gae_and_next_value, transition):
        gae, i_gae, next_value, i_next_value = gae_and_next_value
        done, value, reward, i, i_value_fast, i_value_slow = (
            transition.done,
            transition.value,
            transition.reward,
            transition.intrinsic_reward,
            transition.i_value_fast,
            transition.i_value_slow,
        )

        not_done = (1-done)
        
        delta = reward + γ * next_value * not_done - value
        gae = delta + (γ * λ * not_done * gae)
        
        # Intrinsic is non-episodic (no done masking)
        i_delta = i + γi * i_next_value * not_done- i_value_slow 
        i_gae = i_delta + (γi * λi * not_done * i_gae)
        
        return (gae, i_gae, value, i_value_fast), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val_fast),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value_slow)

def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches

def calculate_gae_intrinsic_and_extrinsic_done_mask(traj_batch, last_val, last_i_val, γ, λ, γi = None, λi = None):

    if λi is None:
        λi = λ 

    if γi is None:
        γi = γ
        
    def _get_advantages(gae_and_next_value, transition):
        gae, i_gae, next_value, i_next_value = gae_and_next_value
        done, value, reward, i, i_value = (
            transition.done,
            transition.value,
            transition.reward,
            transition.intrinsic_reward,
            transition.i_value,
        )
        
        delta = reward + γ * next_value * (1 - done) - value
        gae = delta + (γ * λ * (1 - done) * gae)
        
        i_delta = i + γi * i_next_value * (1 - done) - i_value 
        i_gae = i_delta + (γi * λi * i_gae)
        
        return (gae, i_gae, value, i_value), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

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

def get_alpha_schedule(a_schedule, min_lr=0.1):
    """Returns a function that calculates alpha based on the update step t."""

    if a_schedule == 'inv_t':
        def alpha_fn(t):
            return jnp.maximum(min_lr, 1.0 / (t + 1e-8))
    
    elif a_schedule == 'constant':
        def alpha_fn(t):
            return min_lr
    else:
        assert f'a_schedule={a_schedule} not recognized.'
    return alpha_fn

def schedule_extrinsic_to_intrinsic_ratio(percent, ratio_e_to_i = 1.0):
    # Phase 1: Flat at 1 (0% -> 80%)
    # Phase 3: Decay (80% -> 100%)
    decay = jnp.clip((1.0 - percent) / 0.5, 0.0, 1.0)
    return ratio_e_to_i * decay

def warmup_env(rng, env, env_params, config):
    """
    Runs warmup to populate running statistics, then resets the environment
    to s0 while preserving those statistics.
    
    Assumes Wrapper Hierarchy from make_env:
    Outer -> NormalizeObservationWrapper -> NormalizeRewardWrapper -> Base(Flatten/Clip/etc) -> Inner
    """
    
    # 1. Check which wrappers are actually active
    norm_obs = config.get("NORMALIZE_OBS", False)
    norm_rew = config.get("NORMALIZE_REWARDS", False)
    num_envs = config["NUM_ENVS"]
    
    # 2. Prepare RNGs
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)

    # -------------------------------------------------------------------------
    # CASE A: No Normalization (Skip Warmup)
    # -------------------------------------------------------------------------
    if not norm_obs and not norm_rew:
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
        return env_state, obsv, rng

    # -------------------------------------------------------------------------
    # CASE B: Run Warmup
    # -------------------------------------------------------------------------
    # Initial reset just for the warmup loop
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    def _warmup_step(runner_state, step_idx):
        env_state, last_obs, rng = runner_state
        
        # RNG splitting
        rng, _rng = jax.random.split(rng)
        rng_action = jax.random.split(_rng, num_envs)
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_envs)

        # Action / Step
        action = jax.vmap(env.action_space(env_params).sample)(rng_action)
        obsv, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
            rng_step, env_state, action, env_params
        )

        # Staggered Starts Logic
        if config.get("STAGGERED_STARTS", False):
            rng, _rng = jax.random.split(rng)
            # Create a threshold for each env
            start_thresholds = jax.random.randint(_rng, (num_envs,), 0, config["WARMUP"])
            active_mask = step_idx < start_thresholds
            
            # Mask state and observation
            env_state = jax.tree.map(
                lambda x, y: jnp.where(active_mask.reshape(-1, *([1] * (x.ndim - 1))), x, y),
                next_env_state, env_state
            )
            obsv = jnp.where(active_mask.reshape(-1, *([1] * (obsv.ndim - 1))), obsv, last_obs)
        else:
            env_state = next_env_state

        return (env_state, obsv, rng), None

    # Run Scan
    warmup_runner_state = (env_state, obsv, rng)
    (env_state, obsv, rng), _ = jax.lax.scan(
        _warmup_step, warmup_runner_state, jnp.arange(config["WARMUP"])
    )

    # -------------------------------------------------------------------------
    # CASE C: State Injection (Reset to s0, keep stats)
    # -------------------------------------------------------------------------
    # We generate fresh reset keys for the actual run
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)

    if norm_obs and norm_rew:
        # Hierarchy: ObsWrapper(RewardWrapper(Base))
        obs_stats = env_state.mean_std
        rew_stats = env_state.env_state.mean_std
        base_env = env._env._env 

        # 1. Reset the base environment (clears game state, gives raw s0)
        raw_obs, base_state = jax.vmap(base_env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # 2. Reconstruct Reward State (Keep stats, reset return_val to 0)
        # Note: We create a zeros array of shape (NUM_ENVS,) for the vectorized state
        rew_state = NormalizeRewardEnvState(
            mean_std=rew_stats,
            return_val=jnp.zeros((num_envs,), dtype=jnp.float32), 
            env_state=base_state
        )

        # 3. Reconstruct Obs State (Keep stats, wrap reward state)
        final_state = NormalizeObsEnvState(
            mean_std=obs_stats,
            env_state=rew_state
        )

        # 4. Normalize the raw s0 observation using warmed-up stats
        final_obs = jax.vmap(env._normalize)(raw_obs, obs_stats)

    elif norm_obs and not norm_rew:
        # Hierarchy: ObsWrapper(Base)
        obs_stats = env_state.mean_std
        base_env = env._env 

        raw_obs, base_state = jax.vmap(base_env.reset, in_axes=(0, None))(reset_rngs, env_params)
        
        final_state = NormalizeObsEnvState(
            mean_std=obs_stats,
            env_state=base_state
        )
        final_obs = jax.vmap(env._normalize)(raw_obs, obs_stats)

    elif not norm_obs and norm_rew:
        # Hierarchy: RewardWrapper(Base)
        rew_stats = env_state.mean_std
        base_env = env._env 

        raw_obs, base_state = jax.vmap(base_env.reset, in_axes=(0, None))(reset_rngs, env_params)
        
        final_state = NormalizeRewardEnvState(
            mean_std=rew_stats,
            return_val=jnp.zeros((num_envs,), dtype=jnp.float32),
            env_state=base_state
        )
        final_obs = raw_obs # No obs normalization

    return final_state, final_obs, rng

def update_cov_and_get_rho(traj_batch, sigma_state, get_features_fn, int_rew_from_features_fn, sigma_ema_alpha_fn):
    "Updates traj_batch and sigma_state based on feature visitations."
    # --- 1. Update EMA of Gram Matrix ---
    phi = get_features_fn(traj_batch.obs)          # inference of RND net for features:
    next_phi = get_features_fn(traj_batch.next_obs)  # Contains s_T (Terminal)
    terminal_phi = next_phi * traj_batch.done[..., None]
    # Sigma is updated based on only states visted as s, plus terminal states (Which are only ever visited as s')
    all_phi_sigma = jnp.concatenate([phi, terminal_phi], axis=0)

    # Update Sigma
    mask_sigma = jnp.concatenate([jnp.ones_like(traj_batch.done), traj_batch.done], axis=0)
    
    sigma_state = sigma_update_masked(sigma_state, all_phi_sigma, mask_sigma, sigma_ema_alpha_fn(sigma_state['t']) )
    
    # A. Intrinsic Reward (Future Novelty)
    rho = int_rew_from_features_fn(next_phi)
    rho = rho - rho.min()
    traj_batch = traj_batch._replace(intrinsic_reward=rho) # used by LSTD estimat

    return traj_batch, sigma_state, rho


def add_values_to_metric(config, metric, int_rew_from_state, evaluator, old_beta, network, train_state, traj_batch, get_vi = None, get_ve = None):
    "Uses evaluator to compute the per-state quantities and append them to metric, quantities: ve, vi, ri"
    ri = int_rew_from_state(evaluator.obs_stack)
    ri = evaluator.get_value_grid(ri)
    effective_visits = (old_beta / jnp.maximum(ri, 1e-8))**2
    # 1. Compute Exact Values using the Evaluator
    v_e, v_i, v_pred = evaluator.compute_true_values(network, train_state.params, int_rew_from_state)
    if len(v_pred) ==2:
        v_pred, vi_pred = v_pred # evaluator already returned these in grid form if the network returns them
    else:
        vi_pred = get_vi(evaluator.obs_stack)
        v_pred = evaluator.get_value_grid(v_pred) if get_ve is None else evaluator.get_value_grid(get_ve(evaluator.obs_stack))
        vi_pred = evaluator.get_value_grid(vi_pred)
    
    obs = jnp.asarray(traj_batch.obs)
    if config.get("ENV_NAME") in {"FourRooms-misc", "FourRoomsCustom-v0"}:
        # Visual FourRooms obs: (T, B, H, W, C) where channel 1 is the agent one-hot map.
        if obs.ndim >= 5:
            metric['visitation_count'] = obs[..., 1].sum(axis=(0, 1))
        # Vector FourRooms obs: (T, B, 4) = [agent_y, agent_x, goal_y, goal_x].
        elif obs.ndim >= 3 and obs.shape[-1] >= 2:
            size = int(config.get("FOURROOMS_SIZE", ri.shape[0]))
            pos = obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, size - 1).reshape(-1)
            x = jnp.clip(pos[..., 1], 0, size - 1).reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
        else:
            metric['visitation_count'] = jnp.zeros_like(ri)
    elif config['RND_NETWORK_TYPE'] == 'identity':
        # One-hot tabular features: recover state visitation from observations.
        visitation = obs.sum(0).sum(0)
        metric['visitation_count'] = evaluator.get_value_grid(visitation)
    else:
        metric['visitation_count'] = jnp.zeros_like(ri)
    
    e_sq_err = (v_e - v_pred)**2
    i_sq_err = (v_i - vi_pred)**2
    num_reachable = jnp.sum(evaluator.reachable_mask)
    
    metric.update({
        "ri_grid": ri,
        "vi_pred": vi_pred,
        "v_i_pred": vi_pred,
        "v_i": v_i,
        "v_e": v_e,
        "v_e_pred": v_pred,
        "e_value_error": jnp.sum(evaluator.reachable_mask * e_sq_err) / num_reachable,
        "i_value_error": jnp.sum(evaluator.reachable_mask * i_sq_err) / num_reachable,
        "effective_visits": effective_visits,
    })

    return metric

def update_beta(old_beta, i_values, e_values, progress, update=True):
    "Scales rho. Normalizes the intrinsic value relative to the extrinsic value. "
    "End result: ri is normalized by dividing the mean intinrisic value and multiplying by the mean extrinisc value."
    "Let i = mean(|v_i|) and e = mean(|v_e|), where v_i is unscaled intrinsic value. c_t is on a schedule (usually = 1)"
    "β = c_t * v / i"
    "v_i_scaled = β v_i = (v_i / i) * ( c_t * e )"
    if not update:
        return old_beta
    c_t = schedule_extrinsic_to_intrinsic_ratio(progress) # ratio of i_value to e_value, equal to 1 for most of learnings 
    vi_mag = jnp.mean(jnp.abs(i_values))
    ve_mag = jnp.mean(jnp.abs(e_values))
    target_mag = jnp.maximum(ve_mag, 0.1) # Floor for extrinsic scale
    beta = c_t * target_mag / (vi_mag + 1e-8)
    # will get v_i / vi_mag times c_t times target mag.
    # beta = EMA(0.95, old_beta, beta)
    # beta = jnp.minimum(beta, 1000.0)
    return beta


def calculate_traces(traj_batch, features, γ, λ, is_episodic: bool, is_absorbing: bool):
    """
    Unified trace calculation supporting 'episodic', 'continuing', and 'absorbing'.
    Input shapes:
        features: (T, B, k)
        traj_batch.done: (T, B)
    Returns:
        traces: (T, B, k)
    """
    # The trace is severed if the environment physically resets.
    # This is True for both Episodic and Absorbing states.
    mask_trace = is_episodic or is_absorbing
    
    def _step_trace(trace_prev, scan_inputs):
        phi, done = scan_inputs
        # Calculate current trace: decay the history and add current feature
        trace_current = trace_prev * γ * λ + phi
        # Determine what gets passed as the "history" for the NEXT step.
        # Convert boolean mask flag to float multiplier: True -> 1.0, False -> 0.0
        trace_mult = 1.0 - (done * float(mask_trace))
        trace_next = trace_current * trace_mult[..., None] 
        
        # Return: (carry_state, output_for_this_step)
        return trace_next, trace_current

    # Scan over the time dimension (T)
    _, traces = jax.lax.scan(_step_trace, jnp.zeros_like(features[0]), (features, traj_batch.done))
    
    return traces

def calculate_gae(
    traj_batch, 
    γ, λ, 
    is_episodic: bool, 
    is_absorbing: bool, 
    γi=None, λi=None
):
    """Unified extrinsic and intrinsic GAE, handles episodic, continuing, and abosrbing formulation."""
    if γi is None: γi = γ
    if λi is None: λi = λ

    i_mask_boot = is_episodic and not is_absorbing
    i_mask_gae = is_episodic or is_absorbing

    # Scan natively over traj_batch
    def _get_advantages(gae_accs, transition):
        gae, i_gae = gae_accs
        
        done = transition.done 
        i_boot_mult = 1.0 - (done * i_mask_boot)
        i_gae_mult  = 1.0 - (done * i_mask_gae)

        # Pull next values directly from the transition tuple!
        delta = transition.reward + γ * transition.next_value * (1-done) - transition.value
        gae = delta + (γ * λ * (1-done) * gae)
        
        i_delta = transition.intrinsic_reward + γi * transition.next_i_val * i_boot_mult - transition.i_value 
        i_gae = i_delta + (γi * λi * i_gae_mult * i_gae)
        
        return (gae, i_gae), (gae, i_gae)

    initial_accs = (jnp.zeros_like(traj_batch.value[0]), jnp.zeros_like(traj_batch.i_value[0]))

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

def initialize_evaluator(config):
    from envs.deepsea_v import DeepSeaExactValue
    from envs.fourrooms_custom import FourRoomsExactValue
    from envs.long_chain import LongChainExactValue

    absorbing = config.get('ABSORBING_TERMINAL_STATE', True)
    episodic = config.get('EPISODIC', True)
    
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    
    evaluator = None
    if config['ENV_NAME'] == 'DeepSea-bsuite':
        evaluator = DeepSeaExactValue(
            size=config['DEEPSEA_SIZE'], 
            unscaled_move_cost=0.01, 
            gamma=config['GAMMA'], 
            episodic=episodic,
            absorbing=absorbing
        )
    elif config["ENV_NAME"] in {"FourRooms-misc", "FourRoomsCustom-v0"}:
        goal_pos = config.get("FOURROOMS_GOAL_POS", None)
        if goal_pos is not None:
            goal_pos = tuple(goal_pos)
        evaluator = FourRoomsExactValue(
            size=int(config.get("FOURROOMS_SIZE", 13)),
            fail_prob=float(config.get("FOURROOMS_FAIL_PROB", 1.0 / 3.0)),
            gamma=config["GAMMA"],
            episodic=episodic,
            use_visual_obs=(config["NETWORK_TYPE"] == "cnn"),
            goal_pos=goal_pos,
            absorbing=absorbing
        )
    elif config['ENV_NAME'] == 'Chain':
        evaluator = LongChainExactValue(config.get('CHAIN_LENGTH', 100), config['GAMMA'], episodic, absorbing= absorbing)
    
    return evaluator 