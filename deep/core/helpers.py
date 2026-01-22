# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
from envs.sparse_mc import SparseMountainCar
from envs.fourrooms_custom import FourRooms
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import NormalizeObservationWrapper, NormalizeRewardWrapper, AddChannelWrapper, ClipAction
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
    if config['ENV_NAME'] == "FourRooms-misc" and config['NETWORK_TYPE'] == 'cnn':
        N = config['FOURROOMS_SIZE']
        env = FourRooms(N=N, use_visual_obs=True)
        env_params = env.default_params
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
    S_b = 0.5 * (S_b + S_b.T) # symmetrize
    S = (1-α) * S + α * S_b # EMA
    return {'S': S, 'N': N, 't': t+1} # new sigma_state

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

def calculate_gae_intrinsic_and_extrinsic(traj_batch, last_val, last_i_val, γ, λ):
    """Continuing Intrinsic TD Target"""
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
        i_delta = i + γ * i_next_value - i_value 
        i_gae = i_delta + (γ * λ * i_gae)
        
        return (gae, i_gae, value, i_value), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

def calculate_gae_intrinsic_and_extrinsic_episodic(traj_batch, last_val, last_i_val, γ, λ):
    """Episodic Intrinsic TD Target"""
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
        i_delta = i + γ * not_done * i_next_value - i_value 
        i_gae = i_delta + (γ * λ * not_done * i_gae )
        
        return (gae, i_gae, value, i_value), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)

def calculate_i_and_e_gae_two_critic(traj_batch, last_val, last_i_val_fast, γ, λ):
    """
    Continuing Intrinsic TD Target using two intrinsic critics: fast (for TD(λ)) and slow (for baseline)
    A = Q_fast - V_slow
    Value Target = Q_fast
    """
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
        i_delta = i + γ * i_next_value - i_value_slow 
        i_gae = i_delta + (γ * λ * i_gae)
        
        return (gae, i_gae, value, i_value_fast), (gae, i_gae)

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), jnp.zeros_like(last_val), last_val, last_i_val_fast),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value_slow)

def calculate_i_and_e_gae_two_critic_episodic(traj_batch, last_val, last_i_val_fast, γ, λ):
    """
    Continuing Intrinsic TD Target using two intrinsic critics: fast (for TD(λ)) and slow (for baseline)
    A = Q_fast - V_slow
    Value Target = Q_fast
    """
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
        i_delta = i + γ * i_next_value * not_done- i_value_slow 
        i_gae = i_delta + (γ * λ * not_done * i_gae)
        
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

def calculate_gae_intrinsic_and_extrinsic_done_mask(traj_batch, last_val, last_i_val, γ, λ):
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
        
        i_delta = i + γ * i_next_value * (1 - done) - i_value 
        i_gae = i_delta + (γ * λ * i_gae)
        
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