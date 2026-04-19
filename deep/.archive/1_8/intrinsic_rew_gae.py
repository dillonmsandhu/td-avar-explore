from core.utils import *
import core.helpers as helpers
import core.networks as networks

DEFAULT_CONFIG = {
    "ENV_NAME": "SparseMountainCar-v0",
    # "ENV_NAME": "DeepSea-bsuite",
    "LR": 5e-4,
    "LR_END": 5e-4,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 120_000,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.6,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.003,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "NORMALIZE_REWARDS": False,
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,
    "BONUS_SCALE": 10.0,
    "REGULARIZATION": 1e-4,
    "PER_UPDATE_REGULARIZATION": 1e-4,
    "SEED": 42,
    "WARMUP": 200, # warmup steps for running mean/std
    "N_SEEDS": 4,
    "DEEPSEA_SIZE": 20,
}
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray 
    i_value: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray 
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def lstd_i_val(phi_fn, obs, lstd_state):
    """
    phi: (..., k)
    returns: (...)
    """
    features = phi_fn(obs)
    return features @ lstd_state["w_int"]

def get_int_rew(S, features, N):
    Sigma_inv = jnp.linalg.solve(S, jnp.eye(features.shape[-1]))
    bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features) / jnp.maximum(1.0, N)
    rho = config['BONUS_SCALE'] * jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    return rho

def lstd_batch_update( 
                    lstd_state: Dict,
                    transitions, # Explore_Transition
                    features: jnp.ndarray,
                    next_features: jnp.ndarray,
                    traces: jnp.ndarray,
                    config: Dict,
                    α: float,
                    α_b: float
    ):
    
    # Fix 1: Add 'current_features' as argument
    def lstd(traces, current_features, next_features, transition):
        # Now current_features is (k,) thanks to vmap
        td_features = current_features - config['GAMMA'] * (1 - transition.done) * next_features
        # A += z * (φ - γφ')^T
        A_sample = jnp.outer(traces, td_features)
        return A_sample
    
    # Unpack state (Assuming these are RAW uncorrected EMAs)
    A, S, t = lstd_state['A'], lstd_state['S'], lstd_state['t']
    batch_axes = tuple(range(transitions.done.ndim))
    N = transitions.done.size + lstd_state['N']  # total number of samples seen so far
    # Fix 1: Pass features to the vmap
    # Vmap structure: (L, B, k) -> we map over axis 0 (L) then axis 1 (B)
    batch_lstd = jax.vmap(jax.vmap(lstd))
    
    # A_update will be (L, B, k, k)
    A_update = batch_lstd(traces, features, next_features, transitions)
    S_update = jax.vmap(jax.vmap(lambda x: jnp.outer(x,x)))(features)

    # Batch average
    A_b, S_b = jax.tree.map(lambda x: x.mean(axis=batch_axes), (A_update, S_update))
    # regularize
    reg_eye = config['PER_UPDATE_REGULARIZATION'] * jnp.eye(A.shape[0])
    A_b += reg_eye
    S_b += reg_eye
    # symmetrize
    S_b = 0.5 * (S_b + S_b.T)  # symmetrize
    
    # EMA
    A = (1-α) * A + α * A_b
    S = (1-α) * S + α * S_b
    
    # bias correction
    # bc = 1.0 - (1.0 - α)**t
    # bc = jnp.maximum(bc, 1e-6)
    bc = 1.0
    A_view, S_view = jax.tree.map(lambda x: x / bc, (A, S))
    
    rho = get_int_rew(S_view, next_features, N)
    # batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    # rho = get_int_rew(S_view, next_features, 2*batch_size/α)

    # solve LSTD for intrinsic system A^{-1} x = b
    b_int_sample = traces * rho[..., None] # Expand rho for broadcast
    b_b = b_int_sample.mean(axis=batch_axes)
    b_new = (1-α_b) * lstd_state['b_int'] + α_b * b_b
    # bc = 1.0 - (1.0 - α_b)**t
    # bc = jnp.maximum(bc, 1e-6)
    b_view = b_new / bc
    w_int = jnp.linalg.solve(A_view, b_view)
    
    return {'A': A, 'S': S, 'b_int': b_new, 'w_int': w_int, 'N': N, 't': t+1, 'int_rew': rho}

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    
    GET_ALPHA_FN = lambda t: jnp.maximum(1/100, 1/t)
    GET_ALPHA_FN_b = lambda t: jnp.maximum(1/10, 1/t)
    
    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = initialize_rnd_network(rnd_rng, obs_shape, config)
        _, target_params = initialize_rnd_network(target_rng, obs_shape, config)
            
        # initialize value and policy network
        network, network_params = initialize_actor_critic(rng, obs_shape, n_actions, config, n_heads=2)
        dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
        dummy_phi = rnd_net.apply(target_params, dummy_obs)
        k = dummy_phi.shape[-1]
        initial_lstd_state = {
            'A': jnp.eye(k) * config['REGULARIZATION'],  # Regularization for numerical stability
            'b_int': jnp.zeros(k), 
            'w_int': jnp.zeros(k),
            'S': jnp.eye(k) * config['REGULARIZATION'],
            'N': 0, # number of samples
            't': 1, # number of updates
            'int_rew': jnp.zeros((config['NUM_STEPS'], config['NUM_ENVS'], ))
        }
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # WARMUP:
        def _warmup_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            
            # Select RANDOM actions (no need for network here, just exploring state space)
            rng, _rng = jax.random.split(rng)
            rng_action = jax.random.split(_rng, config["NUM_ENVS"])
            action = jax.vmap(env.action_space(env_params).sample)(rng_action)
            
            # Step env (wrappers will update their internal mean/std stats automatically)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                rng_step, env_state, action, env_params
            )
            
            return (env_state, obsv, rng), None

        # Run the warmup
        warmup_runner_state = (env_state, obsv, rng)
        (env_state, obsv, rng), _ = jax.lax.scan(
            _warmup_step, warmup_runner_state, None, config["WARMUP"]
        )
        # -------------------------

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng= env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # get i_val (dot product with LSTD state)
                i_val = lstd_i_val(get_features_fn, last_obs, lstd_state)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                # Record
                intrinsic_reward = jnp.zeros_like(reward)  # placeholder, will be filled later
                transition = Transition(
                    done, action, value, i_val, reward, intrinsic_reward, log_prob, last_obs, obsv, info, 
                )

                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            # COMPUTE intrinsic reward:            
            next_phi = batch_get_features(traj_batch.next_obs)
            # TODO: Zero out intrinsic reward when done...
            rho = get_int_rew(lstd_state['S'], next_phi, lstd_state['N'])
            # rho = get_int_rew(lstd_state['S'], next_phi, 2 * batch_size / GET_ALPHA_FN(lstd_state['t']))
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            # Advantage
            _, last_val = network.apply(train_state.params, last_obs)
            last_i_val = lstd_i_val(get_features_fn, last_obs, lstd_state)
            gaes, targets = helpers.calculate_gae_intrinsic_and_extrinsic(traj_batch, last_val, last_i_val, config["GAMMA"], config["GAE_LAMBDA"])
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss
                # end update_minibatch

                train_state, rnd_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # end update_epoch

            # --------- Train the network ---------
            initial_update_state = (train_state, rnd_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, rnd_state, _, _, _, rng = update_state
            # -------------------------------
            # --------- Update LSTD ---------
            new_phi = batch_get_features(traj_batch.obs)
            new_phi_prime = batch_get_features(traj_batch.next_obs)
            traces = helpers._get_all_traces(traj_batch, new_phi, config['GAMMA'], config['GAE_LAMBDA'])
            lstd_state = lstd_batch_update(
                lstd_state,
                traj_batch,
                new_phi,
                new_phi_prime,
                traces,
                config,
                α=GET_ALPHA_FN(lstd_state['t']),
                α_b=GET_ALPHA_FN_b(lstd_state['t']),
            )
            # -------------------------------
            # --------- Update metrics ------
            metric = {k: v.mean() for k, v in traj_batch.info.items()} # performance
            metric.update({
                "ppo_loss": loss_info[0], 
                "rnd_loss": loss_info[1],
                "feat_norm": jnp.linalg.norm(next_phi, axis=-1).mean(),
                "bonus_mean": gaes[1].mean(),
                "bonus_std": gaes[1].std(),
                "bonus_max": gaes[1].max(),
                "lambda_ret_mean": targets[0].mean(),
                "lambda_ret_std": targets[0].std(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                "intrinsic_v_mean": traj_batch.i_value.mean(),
                "intrinsic_v_std": traj_batch.i_value.std(),
                "i_val_const_obs": lstd_i_val(get_features_fn, jnp.zeros_like(traj_batch.obs), lstd_state).mean(),
                "mean_rew": traj_batch.reward.mean(),
            })
            runner_state = (train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric
            # end update_step

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
    
def main():
    import warnings; warnings.simplefilter('ignore')
    import os
    from core.utils import save_results, save_plot, parse_config_override
    import datetime
    import argparse
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD Explore experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values, e.g. \'{"LR": 0.001, "LAMBDA": 0.0}\'')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help='saves to count_rew_prop/{args.run_suffix}' )
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')

    
    args = parser.parse_args()
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Override with command line config
    config_override = parse_config_override(args.config)
    config.update(config_override)
    # update the network type and learning rate based on the env.
    config = resolve_env_config(config)
    rng = jax.random.PRNGKey(config['SEED'])
        
    def evaluate(config, rng):
        steps_per_pi = config["NUM_ENVS"]*config["NUM_STEPS"]
        run_fn = jax.jit(jax.vmap(make_train(config)))
        rngs = jax.random.split(rng, config['N_SEEDS'])
        out = run_fn(rngs)
        metrics = out["metrics"]

        print("Mean return is " , jnp.mean(metrics['returned_episode_returns']))
        print("(Mean) Max return is " , jnp.max(metrics['returned_episode_returns']))

        run_dir = os.path.join("results", f"count_rew_prop/{args.run_suffix}")
        env_dir = os.path.join(run_dir, config['ENV_NAME'])
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(env_dir, exist_ok=True)
        print(f"Saving {config['ENV_NAME']} results to {run_dir}")

        if args.save_checkpoint:
            save_results(out, config, config['ENV_NAME'], env_dir)
        else:
            save_results(metrics, config, config['ENV_NAME'], env_dir)
        
        mean_rets = metrics['returned_episode_returns'].mean(0) if config['N_SEEDS'] > 1 else metrics['returned_episode_returns']
        if config['ENV_NAME'] == "SparseMountainCar-v0":
            mean_rets = metrics['returned_discounted_episode_returns'].mean(0) if config['N_SEEDS'] > 1 else metrics['returned_discounted_episode_returns']
        
        bonus_mean = metrics['bonus_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['bonus_mean']
        intrinsic_v_mean = metrics['intrinsic_v_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['intrinsic_v_mean']
        intrinsic_v_constant_obs = metrics['i_val_const_obs'].mean(0) if config['N_SEEDS'] > 1 else metrics['i_val_const_obs']
        intrinsic_rew_mean = metrics['intrinsic_rew_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['intrinsic_rew_mean']
        
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, bonus_mean[1:], 'i_advantage')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsic_v_mean[1:], 'i_val')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsic_v_constant_obs[1:], 'i_val_zero_obs')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsic_v_constant_obs[1:], 'i_val_zero_obs')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsic_rew_mean[1:], 'intrinsic_rew_mean')
    
    evaluate(config, rng)

if __name__ == '__main__':
    main()