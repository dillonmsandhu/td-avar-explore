# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
from envs.sparse_mc import SparseMountainCar
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.long_chain import LongChain
from envs.fourrooms_custom import FourRooms
from envs.fourrooms_pixel import FourRoomsPixelExactValue, FourRoomsPixels
from envs.maze import SparseMaze, SparseMazeExactValue
from envs.wrappers import NormalizeObservationWrapper, NormalizeRewardWrapper, AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, TerminalInfoWrapper, SubtractOneRewardWrapper
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
    if config['ENV_NAME'] == 'Maze':
        env = SparseMaze(N= config.get('ENV_SIZE',100))
        env_params = env.default_params
    elif config['ENV_NAME'] == 'FourRoomsPixels':
        env = FourRoomsPixels(N=config.get('ENV_SIZE',21), pixel_size = config.get('FOURROOMS_PIXEL_SIZE', 84))
        env_params = env.default_params
    elif config['ENV_NAME'] in  ("FourRoomsCustom-v0"):
        env = FourRooms(
            N=int(config.get("ENV_SIZE", 21)),
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
            N=int(config.get("ENV_SIZE", 13)),
            use_visual_obs=(config["NETWORK_TYPE"] == "cnn"),
        )
        env_params = env.default_params.replace(
            fail_prob=float(config.get("FOURROOMS_FAIL_PROB", env.default_params.fail_prob)),
            resample_init_pos=bool(config.get("FOURROOMS_RESAMPLE_INIT_POS", env.default_params.resample_init_pos)),
            resample_goal_pos=bool(config.get("FOURROOMS_RESAMPLE_GOAL_POS", env.default_params.resample_goal_pos)),
            max_steps_in_episode=int(config.get("FOURROOMS_MAX_STEPS", env.default_params.max_steps_in_episode)),
        )
    
    elif config['ENV_NAME'] == "Chain":
        env = LongChain(config.get('ENV_SIZE', 100))
        env_params = env.default_params.replace(
            fail_prob=float(config.get("CHAIN_FAIL_PROB", env.default_params.fail_prob)),
            resample_init_pos=bool(config.get("CHAIN_RESAMPLE_INIT_POS", env.default_params.resample_init_pos)),
            max_steps_in_episode=int(config.get("CHAIN_MAX_STEPS", env.default_params.max_steps_in_episode)),
        )

    elif config['ENV_NAME'] == "SparseMountainCar-v0":
        env = SparseMountainCar()
        env_params = env.default_params
    
    elif config['ENV_NAME'] == 'DeepSea-bsuite':
        env, env_params = gymnax.make(config["ENV_NAME"], size = config.get("ENV_SIZE", 10))
    elif config['ENV_NAME'] == 'DeepSea-Dense':
        env, env_params = gymnax.make('DeepSea-bsuite', size = config.get("ENV_SIZE", 10))
        env = SubtractOneRewardWrapper(env)
    
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
        print('Env:', config['ENV_NAME'])
        print('Network:', config['NETWORK_TYPE'])
        print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    # "Goal" environments based on best of knowledge (in Atari we get a goal flag.)
    goal_envs = [
        'DeepSea', 'Chain', 'Maze', 'FourRooms', 'MountainCar', 
        'Freeway-MinAtar', 'Catch', 'DiscountingChain', 'MetaMaze', 'Pong'
    ]
    is_goal_env = any(env in config['ENV_NAME'] for env in goal_envs)
    
    env = TerminalInfoWrapper(env, is_goal_env) # adds the terminal state to info. also adds goal information.
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


def EMA(coeff, x_old, x_new):
    return (1 - coeff) * x_old + coeff * x_new

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


def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches

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


def add_values_to_metric(config, metric, int_rew_from_state, evaluator, old_beta, network, train_state, traj_batch, get_vi = None, get_ve = None, compute_true_vals = True):
    """Uses evaluator to compute the per-state quantities and append them to metric."""
    
    # 1. Intrinsic Reward Grid (Delegated to Evaluator)
    ri = evaluator.get_value_grid(int_rew_from_state(evaluator.obs_stack))
    effective_visits = (old_beta / jnp.maximum(ri, 1e-8))**2

    # 2. Compute Oracle Values & Default Network Predictions
    if compute_true_vals:
        # The evaluator dictates the exact ground truth shapes here
        v_e, v_i, v_net_tuple = evaluator.compute_true_values(network, train_state.params, int_rew_from_state)
        
        # Unpack network predictions
        if isinstance(v_net_tuple, tuple) and len(v_net_tuple) == 2:
            v_pred, vi_pred = v_net_tuple
        else:
            v_pred = v_net_tuple
            vi_pred = jnp.zeros_like(v_i)
    else:
        v_e = jnp.zeros_like(ri)
        v_i = jnp.zeros_like(ri)
        
        # Memory-safe prediction scan if we aren't using the evaluator's exact solver
        def _net_step(unused, x):
            res = network.apply(train_state.params, x[None, ...])
            return None, jax.tree.map(lambda arr: arr.squeeze(0), res)
        _, out = jax.lax.scan(_net_step, None, evaluator.obs_stack)
        
        if len(out) == 3:
            v_pred = evaluator.get_value_grid(out[1].squeeze())
            vi_pred = evaluator.get_value_grid(out[2].squeeze())
        else:
            v_pred = evaluator.get_value_grid(out[1].squeeze())
            vi_pred = jnp.zeros_like(ri)

    # 3. Explicit Overrides (Crucial for LSPI Q-value maximization)
    # We evaluate the whole stack, then trust the evaluator to format it.
    if get_ve is not None:
        v_pred = evaluator.get_value_grid(get_ve(evaluator.obs_stack))
    if get_vi is not None:
        vi_pred = evaluator.get_value_grid(get_vi(evaluator.obs_stack))

    # 4. Visitation Logic
    obs = jnp.asarray(traj_batch.obs)
    next_obs = jnp.asarray(traj_batch.next_obs)
    env_name = config.get("ENV_NAME", "")
    
    if "DeepSea" in env_name:
        metric['visitation_count'] = next_obs.sum(axis=(0, 1))    
        
    elif env_name == "FourRoomsPixels-v0":
        metric['visitation_count'] = traj_batch.info["underlying_grid"][..., 1].sum(axis=(0, 1))
        
    elif env_name in {"FourRooms-misc", "FourRoomsCustom-v0"} or "SparseMaze" in env_name:
        if obs.ndim >= 5:
            metric['visitation_count'] = next_obs[..., 1].sum(axis=(0, 1))
        elif obs.ndim >= 3 and obs.shape[-1] >= 2:
            size = ri.shape[0] 
            pos = next_obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, size - 1).reshape(-1)
            x = jnp.clip(pos[..., 1], 0, size - 1).reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
        else:
            metric['visitation_count'] = jnp.zeros_like(ri)
    
    elif "Chain" in env_name:
        # obs shape: [T, B, Chain_Len]
        if obs.shape[-1] > 1 and obs.ndim == 3:
            # It's One-Hot: Summing gives us the count per state index
            metric['visitation_count'] = next_obs.sum(axis=(0, 1))

    elif config.get('RND_NETWORK_TYPE') == 'identity':
        visitation = next_obs.sum(0).sum(0)
        metric['visitation_count'] = evaluator.get_value_grid(visitation)
    else:
        metric['visitation_count'] = jnp.zeros_like(ri)
    
    # 5. Error Metrics (Perfect shape alignment guaranteed by the evaluator)
    e_sq_err = (v_e - v_pred)**2
    i_sq_err = (v_i - vi_pred)**2
    num_reachable = jnp.maximum(jnp.sum(evaluator.reachable_mask), 1.0)
    
    metric.update({
        "ri_grid": ri,
        "v_i_pred": vi_pred,
        "vi_pred": vi_pred,
        "v_i": v_i,
        "v_e": v_e,
        "v_e_pred": v_pred,
        "e_value_error": jnp.sum(evaluator.reachable_mask * e_sq_err) / num_reachable,
        "i_value_error": jnp.sum(evaluator.reachable_mask * i_sq_err) / num_reachable,
        "effective_visits": effective_visits,
    })
    
    return metric


def calculate_traces(traj_batch, features, γ, λ, is_continuing: bool):
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
    cut_when_done = float(~is_continuing)
    
    def _step_trace(trace_prev, scan_inputs):
        phi, done = scan_inputs
        # Calculate current trace: decay the history and add current feature
        trace_current = trace_prev * γ * λ + phi
        # Determine what gets passed as the "history" for the NEXT step.
        # Convert boolean mask flag to float multiplier: True -> 1.0, False -> 0.0
        trace_mult = 1.0 - (done * cut_when_done)
        trace_next = trace_current * trace_mult[..., None] 
        
        # Return: (carry_state, output_for_this_step)
        return trace_next, trace_current

    # Scan over the time dimension (T)
    _, traces = jax.lax.scan(_step_trace, jnp.zeros_like(features[0]), (features, traj_batch.done))
    
    return traces


def calculate_gae(
    traj_batch, 
    γ, λ, 
    is_continuing: bool, # False for Episodic/Absorbing
    γi=None, λi=None
):
    if γi is None: γi = γ
    if λi is None: λi = λ

    def _get_advantages(gae_accs, transition):
        gae, i_gae = gae_accs
        done = transition.done
        is_goal = transition.goal

        # --- Extrinsic ---
        delta = transition.reward + γ * transition.next_value * (1 - done) - transition.value
        gae = delta + (γ * λ * (1 - done) * gae)
        
        # --- Intrinsic --- 
        # If Continuing, we always bootstrap
        # otherwise kill the bootstrap if it is not done, but not at a goal
        cut_gae = done * (not is_continuing)
        end_bootstrap = (not is_continuing) * done * (1 - is_goal)
        
        i_delta = transition.intrinsic_reward + γi * transition.next_i_val * (1-end_bootstrap) - transition.i_value 
        i_gae = i_delta + (γi * λi * (1-cut_gae) * i_gae)
        
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
    from envs.mountaincar import MountainCarExactValue
    absorbing = config.get('ABSORBING_TERMINAL_STATE', True)
    episodic = config.get('EPISODIC', True)
    
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    
    evaluator = None
    if config['ENV_NAME'] == 'MountainCar-v0':
        evaluator = MountainCarExactValue(gamma=config['GAMMA_i'], absorbing = absorbing, dense=False)
    elif config['ENV_NAME'] == 'SparseMountainCar-v0':
        evaluator = MountainCarExactValue(gamma=config['GAMMA_i'], absorbing = absorbing, dense=False)        
    elif config['ENV_NAME'] == 'Maze':
        evaluator = SparseMazeExactValue(size = config.get('ENV_SIZE', 100), episodic=episodic, absorbing = absorbing, gamma=config['GAMMA_i'], )
    
    elif config['ENV_NAME'] == 'FourRoomsPixels':
        evaluator = FourRoomsPixelExactValue(            
            gamma=config['GAMMA_i'], 
            episodic=episodic,
            absorbing=absorbing
    )
    if config['ENV_NAME'] == 'DeepSea-bsuite':
        evaluator = DeepSeaExactValue(
            size=config['ENV_SIZE'], 
            unscaled_move_cost=0.01, 
            gamma=config['GAMMA_i'], 
            episodic=episodic,
            absorbing=absorbing
        )
    if config['ENV_NAME'] == 'DeepSea-Dense': # all rewards minus 1.
        evaluator = DeepSeaExactValue(
            size=config['ENV_SIZE'], 
            unscaled_move_cost=0.01, 
            gamma=config['GAMMA_i'], 
            episodic=episodic,
            absorbing=absorbing,
            dense=True,
        )
    elif config["ENV_NAME"] in {"FourRooms-misc", "FourRoomsCustom-v0"}:
        goal_pos = config.get("FOURROOMS_GOAL_POS", None)
        if goal_pos is not None:
            goal_pos = tuple(goal_pos)
        evaluator = FourRoomsExactValue(
            size=int(config.get("ENV_SIZE", 13)),
            fail_prob=float(config.get("FOURROOMS_FAIL_PROB", 1.0 / 3.0)),
            gamma=config["GAMMA_i"],
            episodic=episodic,
            use_visual_obs=(config["NETWORK_TYPE"] == "cnn"),
            goal_pos=goal_pos,
            absorbing=absorbing
        )
    elif config['ENV_NAME'] == 'Chain':
        evaluator = LongChainExactValue(config.get('ENV_SIZE', 100), config['GAMMA_i'], episodic, absorbing= absorbing)
    
    return evaluator 


def update_cov(traj_batch, sigma_state, get_features_fn):
    "Updates traj_batch and sigma_state based on feature visitations."

    def cov_update_masked(
        sigma_state: Dict,
        features: jnp.ndarray,  # Shape: (..., k)
        mask: jnp.ndarray,      # Shape: (...) matching the batch dimensions of features
    ):
        """
        Pure summation covariance update.
        Takes a mask that corresponds to what included feature vectors are valid for the update.
        """
        S = sigma_state['S']
        N = sigma_state.get('N', 0)
        
        S_update = jnp.einsum("...i, ...j -> ...ij", features, features)
        
        # 2. Apply Mask
        # Zeros out the outer products corresponding to invalid/padding states
        S_masked = S_update * mask[..., None, None]
        
        # 3. PURE SUMMATION (No division!)
        batch_axes = tuple(range(mask.ndim))
        S_batch_sum = jnp.sum(S_masked, axis=batch_axes)
        
        # 4. Update & Force Symmetry
        S_new = S + S_batch_sum
        S_new = 0.5 * (S_new + S_new.T)

        total_valid = jnp.sum(mask)

        return {
            'S': S_new, 
        }

    # --- 1. Update EMA of Gram Matrix ---
    phi = get_features_fn(traj_batch.obs)          # inference of RND net for features:
    next_phi = get_features_fn(traj_batch.next_obs)  # Contains s_T (Terminal)
    terminal_phi = next_phi * traj_batch.done[..., None] 
    # Sigma is updated based on only states visted as s, plus terminal states (Which are only ever visited as s')
    all_phi_sigma = jnp.concatenate([phi, terminal_phi], axis=0)

    # Update Sigma (include the next state when it ends the episode.)
    mask_sigma = jnp.concatenate([jnp.ones_like(traj_batch.done), traj_batch.done], axis=0)
    
    sigma_state = cov_update_masked(sigma_state, all_phi_sigma, mask_sigma)
    
    return sigma_state

def get_scale_free_bonus(S_inv, features):
    """bonus = sqrt(x^T Σ^{-1} x)"""
    bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
    return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

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