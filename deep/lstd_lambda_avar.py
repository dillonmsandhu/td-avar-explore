from utils import *
from helpers import _calculate_gae, _get_all_traces, Explore_Transition, _loss_fn
import helpers
from envs.sparse_mc import SparseMountainCar

DEFAULT_CONFIG = {
    # "ENV_NAME": "SparseMountainCar-v0",
    "ENV_NAME": "DeepSea-bsuite",
    "LR": 5e-4,
    "LR_END": 5e-4,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 250_000,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.6,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.003,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "NORMALIZE_REWARDS": False,
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,
    "BONUS_SCALE": 1.0,
    "REGULARIZATION": 1e-4,
    "PER_UPDATE_REGULARIZATION": 1e-4,
    "SEED": 42,
    "WARMUP": 200, # warmup steps for running mean/std
    "N_SEEDS": 4,
    "PRIOR_N": 1_000, # strength of prior: number of transitions where the "prior" (max) td error was "observed".
    "DEEPSEA_SIZE":15,
}

def compute_bonus(features, lstd_state, config=None):
    """Computes the state bonus using the sandwich covariance"""
    Sandwich = lstd_state['Sandwich']
    variance_per_state = jnp.einsum('...i,ij,...j->...', features, Sandwich, features)
    B = config['BONUS_SCALE'] * jnp.sqrt(jnp.maximum(variance_per_state, 1e-6))
    return B

def lstd_batch_update( 
                    lstd_state: Dict,
                    transitions, # Explore_Transition
                    features: jnp.ndarray,
                    next_features: jnp.ndarray,
                    traces: jnp.ndarray,
                    config: Dict,
                    α_A: float, 
                    α_S: float,
    ):
    
    # Fix 1: Add 'current_features' as argument
    def lstd(traces, current_features, next_features, transition):
        # Now current_features is (k,) thanks to vmap
        td_features = current_features - config['GAMMA'] * (1 - transition.done) * next_features
        # A += z * (φ - γφ')^T
        A_sample = jnp.outer(traces, td_features)
        return A_sample
        
    def per_sample_OPG(z, td_error):
        return jnp.outer(z, z) * (td_error**2)

    # Unpack state (Assuming these are RAW uncorrected EMAs)
    A, S, t = lstd_state['A'], lstd_state['S'], lstd_state['t']
    batch_axes = tuple(range(transitions.done.ndim))
    N = transitions.done.size
    # Fix 1: Pass features to the vmap
    # Vmap structure: (L, B, k) -> we map over axis 0 (L) then axis 1 (B)
    batch_lstd = jax.vmap(jax.vmap(lstd))
    
    # A_update will be (L, B, k, k)
    A_update = batch_lstd(traces, features, next_features, transitions)
    S_update = jax.vmap(jax.vmap(per_sample_OPG))(traces, transitions.td_error)

    # Batch averages
    # batch averages
    A_b, S_b = jax.tree.map(lambda x: x.mean(axis=batch_axes), (A_update, S_update))
    S_b = 0.5 * (S_b + S_b.T)  # immediately symmetrize for numerical stability
    # Add regularization
    A_b = A_b + config['PER_UPDATE_REGULARIZATION'] * jnp.eye(A.shape[0])  # Regularization for numerical stability

    # EMA
    A = (1-α_A) * A + α_A * A_b
    S = (1-α_S) * S + α_S * S_b
    
    # bias correction
    # bc = 1.0 - (1.0 - α_A)**t
    # bc = jnp.maximum(bc, 1e-6)
    bc = 1.0
    A_view = A / bc

    # effective sample size of EMA is 1/alpha
    A_inv = jnp.linalg.solve(A_view, jnp.eye(A.shape[0]))
    N_eff = 2.0 / α_A
    cov_w = (1 / N_eff) * (A_inv @ S @ A_inv.T)

    # return {'A': A_view, 'S': S, 'N': N, 't': t+1, 'Sandwich': cov_w, 'A_update': A_b, 'S_update': S_b}
    return {'A': A, 'S': S, 'N': N + lstd_state['N'], 't': t+1, 'Sandwich': cov_w, 'A_update': A_b, 'S_update': S_b}

def compute_sandwich(lstd_state: Dict, α = 1):
    "Computes the sandwich covariance from LSTD state"
    A, S = lstd_state['A'], lstd_state['S']
    A_inv = jnp.linalg.pinv(A)
    cov_w = (α/2) * (A_inv @ S @ A_inv.T)
    return cov_w

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    total_grad_steps = config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    # EMA set up:
    GET_ALPHA_FN = lambda t: jnp.maximum(1/10, 1/t)
    prior_t = config["PRIOR_N"] // batch_size

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
        max_r = 1.0
        
        initial_lstd_state = {
            'A': jnp.eye(k) * config['REGULARIZATION'],  # Regularization for numerical stability
            'S': jnp.eye(k) * max_r**2,
            'N': 0, # number of samples
            't': 1, # number of updates
            'A_update': jnp.zeros((k, k)),
            'S_update': jnp.zeros((k, k)),
        }
        initial_lstd_state['Sandwich']= compute_sandwich(initial_lstd_state)

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=config["LR_END"],
            transition_steps=total_grad_steps
        )
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(lr_scheduler, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        rnd_state = RNDTrainState.create(
            apply_fn=rnd_net.apply,
            params=rnd_params,
            tx=tx,
            target_params=target_params,
        )
        get_features_fn = lambda params, obs: rnd_net.apply(params, obs)
        batch_get_features = jax.vmap(get_features_fn, in_axes=(None, 0))
        
        get_v_features_fn = lambda params, obs: train_state.apply_fn(params, obs, method="get_value_features")[0]
        batch_get_v_features = jax.vmap(get_v_features_fn, in_axes=(None, 0))

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

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                # Record
                next_pi, next_value = network.apply(train_state.params, obsv)
                δ = reward + config['GAMMA'] * (1-done) * next_value - value
                # δ = jnp.ones_like(reward)

                intrinsic_reward = jnp.zeros_like(reward)
                target_embedding = jnp.zeros_like(reward)
                transition = Explore_Transition(
                    done, action, value, value, reward, intrinsic_reward, log_prob, last_obs, obsv, target_embedding, δ, info
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            # COMPUTE BONUS:
            next_phi = batch_get_features(rnd_state.target_params, traj_batch.next_obs)
            U_prime = compute_bonus(next_phi, lstd_state, config)
            # CALCULATE ADVANTAGE
            _, last_val = network.apply(train_state.params, last_obs)

            gae, targets = _calculate_gae(traj_batch, last_val, config['GAMMA'], config['GAE_LAMBDA'])
            # optimistic advantage! the next state value is uncertainty - so we add a bonus proportional to its stddev
            advantages = gae + config['GAMMA'] * U_prime * (1 - traj_batch.done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                
                def _update_minbatch(minibatch_input, batch_info):
                    train_state, rnd_state, mask_rng = minibatch_input
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    rnd_loss = jnp.zeros_like(total_loss)                    
                    return (train_state, rnd_state, mask_rng), (total_loss, rnd_loss)
                # end update_minibatch

                train_state, rnd_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                rng, mask_rng = jax.random.split(rng)
                (train_state, rnd_state, mask_rng), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, rnd_state, mask_rng), minibatches
                )
                update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # end update_epoch

            # --------- Train the network ---------
            update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["NUM_EPOCHS"]
            )
            train_state, rnd_state, _, _, _, rng = update_state
            # -------------------------------
            # --------- Update LSTD ---------
            new_phi = batch_get_features(rnd_state.target_params, traj_batch.obs)
            new_phi_prime = batch_get_features(rnd_state.target_params, traj_batch.next_obs)
            traces = _get_all_traces(traj_batch, new_phi, config['GAMMA'], config['GAE_LAMBDA'])
            lstd_state = lstd_batch_update(
                lstd_state,
                traj_batch,
                new_phi,
                new_phi_prime,
                traces,
                config,
                α_A=GET_ALPHA_FN(lstd_state['t']),
                α_S=GET_ALPHA_FN(lstd_state['t'] + prior_t),
            )
            # -------------------------------
            # --------- Update metrics ------
            metric = {k: v.mean() for k, v in traj_batch.info.items()} # performance
            
            v_features = batch_get_v_features(train_state.params, traj_batch.next_obs)
            v_feat_norm = jnp.linalg.norm(v_features, axis=-1)
            feat_norm = jnp.linalg.norm(next_phi, axis=-1)

            # constant obs:
            constant_obs = jnp.zeros_like(traj_batch.obs) + 0.1
            target_features_const_obs = rnd_net.apply(rnd_state.target_params, constant_obs)
            avg_targ_feat_const_obs = jnp.linalg.norm(target_features_const_obs,axis=-1).mean()

            # 2. Bonus Statistics
            mean_state = traj_batch.obs.mean(0).mean(0) # shape: (obs_shape)
            mean_rew = traj_batch.reward.mean()
            metric.update({
                "ppo_loss": loss_info[0], 
                "rnd_loss": loss_info[1],
                "mean_x": mean_state[0],
                "mean_v": mean_state[1],
                "v_feat_norm": v_feat_norm.mean(),
                "feat_norm_std": feat_norm.std(),
                "feat_norm": feat_norm.mean(),
                "feat_norm_std": feat_norm.std(),
                "bonus_mean": U_prime.mean(),
                "bonus_std": U_prime.std(),
                "bonus_max": U_prime.max(),
                "implied_count": 1.0 / (jnp.square(U_prime).mean() + 1e-8),
                "gae_mean": gae.mean(),
                "gae_std":  gae.std(),
                "gae_max":  gae.max(),
                "lambda_ret_mean": targets.mean(),
                "lambda_ret_std": targets.std(),
                "td_error_mean":traj_batch.td_error.mean(),
                "td_error_std": traj_batch.td_error.std(),
                "avg_targ_feat_const_obs": avg_targ_feat_const_obs,
                "mean_rew": mean_rew,
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
    from utils import save_results, save_plot, parse_config_override
    import datetime
    import argparse
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD Explore experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values, e.g. \'{"LR": 0.001, "LAMBDA": 0.0}\'')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help='saves to lstd_lambda_avar/{args.run_suffix}' )
    parser.add_argument('--n-seeds', type=int, default=0)
    
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()

    # Override with command line config
    config_override = parse_config_override(args.config)
    config.update(config_override)
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

        run_dir = os.path.join("results", f"lstd_lambda_avar/{args.run_suffix}")
        env_dir = os.path.join(run_dir, config['ENV_NAME'])
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(env_dir, exist_ok=True)
        print(f"Saving {config['ENV_NAME']} results to {run_dir}")

        save_results(metrics, config, config['ENV_NAME'], env_dir)
        mean_rets = metrics['returned_episode_returns'].mean(0) if config['N_SEEDS'] > 1 else metrics['returned_episode_returns']
        if config['ENV_NAME'] == "SparseMountainCar-v0":
            mean_rets = metrics['returned_discounted_episode_returns'].mean(0) if config['N_SEEDS'] > 1 else metrics['returned_discounted_episode_returns']
        
        bonus_mean = metrics['bonus_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['bonus_mean']
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, bonus_mean , 'Bonus')
    
    evaluate(config, rng)

if __name__ == '__main__':
    main()