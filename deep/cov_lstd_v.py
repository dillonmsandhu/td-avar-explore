# Covariance-Based Intrinsic Reward, propegated by LSTD.
# For deepsea only, solves for the value function for debugging.
from utils import *
import helpers
import networks
from envs.deepsea_v import DeepSeaExactValue
SAVE_DIR = 'cov_lstd_v'

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

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    
    # GET_ALPHA_FN = lambda t: jnp.maximum(1/(2*t), 1/10)
    GET_ALPHA_FN = lambda t: 1/10
    # GET_ALPHA_FN_b = lambda t: jnp.maximum(1/5, 1/t)
    # GET_ALPHA_FN_b = lambda t: jnp.maximum(1/(2*t), 1/10)
    GET_ALPHA_FN_b = lambda t: 1/10
    GET_ALPHA_FN_cov = lambda t: jnp.maximum(1/10, 1/t)

    evaluator = DeepSeaExactValue(size=config['DEEPSEA_SIZE'], unscaled_move_cost=0.01)

    def sigma_update(   sigma_state: Dict,
                        transitions, # Explore_Transition
                        features: jnp.ndarray,
                        α: float
        ):
        
        # Unpack state (Assuming these are RAW uncorrected EMAs)
        S, t = sigma_state['S'], sigma_state['t']
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + sigma_state['N']  # total number of samples seen so far
        # S_update (L, B, k, k)
        S_update = jax.vmap(jax.vmap(lambda x: jnp.outer(x,x)))(features)
        # Batch average
        S_b = S_update.mean(axis=batch_axes)
        # symmetrize
        S_b = 0.5 * (S_b + S_b.T)
        # EMA
        S = (1-α) * S + α * S_b
        return {'S': S, 'N': N, 't': t+1} # new sigma_state
    
    def get_int_rew(S, features, N):
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features)
        bonus_sq /= jnp.maximum(1.0, N)
        rho = config['BONUS_SCALE'] * jnp.sqrt(bonus_sq)
        return rho
    
    def lstd_batch_update( 
                        lstd_state: Dict,
                        transitions, # Explore_Transition
                        features: jnp.ndarray,
                        next_features: jnp.ndarray,
                        traces: jnp.ndarray,
                        α: float,
                        α_b: float,
        ):
        def lstd(traces, current_features, next_features, transition):
            td_features = current_features - config['GAMMA'] * next_features
            A_sample = jnp.outer(traces, td_features)
            return A_sample
        
        # Unpack state (Assuming these are RAW uncorrected EMAs)
        A, t = lstd_state['A'], lstd_state['t']
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']  # total number of samples seen so far
        batch_lstd = jax.vmap(jax.vmap(lstd))
        A_update = batch_lstd(traces, features, next_features, transitions) # (L, B, k, k)
        A_b = A_update.mean(axis=batch_axes)
        # EMA
        β = (1-α)
        bc = 1-β**t
        A = β * A + α * A_b
        A_view = A + config['A_REGULARIZATION_PER_STEP'] * jnp.eye((A.shape[0]))
        # A_view /=  (1-β**t)
        rho = transitions.intrinsic_reward
        # solve LSTD for intrinsic system A^{-1} x = b
        b_int_sample = traces * rho[..., None] # Expand rho for broadcast
        b_b = b_int_sample.mean(axis=batch_axes)
        b = (1-α_b) * lstd_state['b_int'] + α_b * b_b
        βb = (1-α_b)
        # b_view = b / (1-βb**t)
        b_view = b
        w_int = jnp.linalg.solve(A_view, b_view)
        return {'A': A, 'b_int': b, 'w_int': w_int, 'N': N, 't': t+1}

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
            # 'A': jnp.eye(k) * config['A_REGULARIZATION'],  # Regularization for numerical stability
            'A': jnp.eye(k),  # Regularization for numerical stability
            'b_int': jnp.zeros(k), 
            'w_int': jnp.zeros(k),
            'N': 0, # number of samples
            't': 1, # number of updates
        }
        initial_sigma_state = {
            'S': jnp.eye(k),
            'N': 0, # number of samples
            't': 1, # number of updates
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
            
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
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
            
            # Intrinsic reward 
            next_phi = batch_get_features(traj_batch.next_obs)
            phi = batch_get_features(traj_batch.obs)
            sigma_state = sigma_update(sigma_state, traj_batch, phi, GET_ALPHA_FN_cov(sigma_state['t']))
            int_rew_from_features = lambda features: get_int_rew(sigma_state['S'], features, sigma_state['N'])
            # alpha = GET_ALPHA_FN_cov(lstd_state['t'])
            # N_eff = (batch_size / alpha )
            # rho = get_int_rew(lstd_state['S'], next_phi, N_eff) * (1-traj_batch.done)
            rho = int_rew_from_features(next_phi)
            traj_batch = traj_batch._replace(intrinsic_reward=rho)

            # Intrinsic Critic:
            traces = helpers._get_all_traces_continuing(phi, config['GAMMA'], config['GAE_LAMBDA'])
            lstd_state = lstd_batch_update(
                lstd_state,
                traj_batch,
                phi,
                next_phi,
                traces,
                α=GET_ALPHA_FN(lstd_state['t']),
                α_b=GET_ALPHA_FN_b(lstd_state['t']),
            )

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

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # end update_epoch

            # --------- Train the network ---------
            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state
            
            # --------- Update metrics ------
            metric = {k: v.mean() for k, v in traj_batch.info.items()} # performance

            # def compute_true_values(self, network: Any, params: PyTree,lstd_state: Dict, get_features: Callable, get_int_rew: Callable
            v_e, v_i, v_pred = evaluator.compute_true_values(network, train_state.params, batch_get_features, int_rew_from_features)
            v_i_pred = evaluator.get_value_grid(lstd_i_val(get_features_fn, evaluator.obs_stack, lstd_state))

            e_value_error = jnp.mean(evaluator.reachable_mask * (v_e - v_pred)**2)
            i_value_error = jnp.mean(evaluator.reachable_mask * (v_i - v_i_pred)**2)

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
                "mean_rew": traj_batch.reward.mean(),
                "v_i": v_i,
                "v_e": v_e,
                "v_e_pred": v_pred,
                "v_i_pred": v_i_pred,
                "e_value_error": e_value_error,
                "i_value_error": i_value_error
            })
            runner_state = (train_state, lstd_state,sigma_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric
            # end update_step

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, 0)
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
    import configs
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD Explore experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values, e.g. \'{"LR": 0.001, "LAMBDA": 0.0}\'')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help=f'saves to {SAVE_DIR}/args.run_suffix/' )
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--base-config', type = str, default = 'mc', choices = ['mc', 'ds'])
    args = parser.parse_args()
    
    if args.base_config == 'mc':
        config = configs.mc_config.copy()
        raise AssertionError('conv_net_v.py only has value solver implemented for DeepSea')
    elif args.base_config == 'ds':
        config = configs.ds_config.copy()

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

        run_dir = os.path.join("results", f"{SAVE_DIR}/{args.run_suffix}")
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
        bonus_std = metrics['bonus_std'].mean(0) if config['N_SEEDS'] > 1 else metrics['bonus_std']
        intrinsic_rew_mean = metrics['intrinsic_rew_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['intrinsic_rew_mean']
        i_value_error = metrics['i_value_error'].mean(0) if config['N_SEEDS'] > 1 else metrics['i_value_error']
        e_value_error = metrics['e_value_error'].mean(0) if config['N_SEEDS'] > 1 else metrics['e_value_error']
        
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, bonus_mean[1:], 'i_advantage_mean')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, bonus_std[1:], 'i_advantage_std')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsic_rew_mean[1:], 'intrinsic_rew_mean')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, i_value_error[1:], 'i_val_mse')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, e_value_error[1:], 'e_val_mse')

        # List of new value metrics to plot
        value_metrics = ["v_i", "v_e", "v_e_pred", "v_i_pred"]

        for key in value_metrics:
            if key in metrics:
                # 1. Handle Multi-Seed Averaging
                data = metrics[key]
                mean_data = data.mean(0) if config['N_SEEDS'] > 1 else data
                initial_state_data = mean_data[:, 0, 0] # initial state is 0 0 on grid.
                
                # 2. Save Plot
                # We slice [1:] to skip the initial untrained step, matching your other plots
                save_plot(env_dir, config['ENV_NAME'], steps_per_pi, initial_state_data[1:], key)
    
    evaluate(config, rng)

if __name__ == '__main__':
    main()