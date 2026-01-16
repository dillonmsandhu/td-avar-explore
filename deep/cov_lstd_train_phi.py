# Covariance-Based Intrinsic Reward, propegated by LSTD.
# For deepsea only, solves for the value function for debugging.
# Learns two intrisic value weights
from utils import *
import helpers
import networks
from envs.deepsea_v import DeepSeaExactValue
SAVE_DIR = 'cov_lstd_train_phi'

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray 
    i_value_fast: jnp.ndarray
    i_value_slow: jnp.ndarray
    phi_train: jnp.ndarray # Keep this! We need the OLD value for PPO clipping.
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray 
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def lstd_i_val_slow(phi_fn, obs, lstd_state):
    """
    phi: (..., k)
    returns: (...)
    """
    features = phi_fn(obs)
    return features @ lstd_state["w_pessimistic"]

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    alpha_fn = lambda t: 1/10
    GET_ALPHA_FN_cov = lambda t: jnp.maximum(1/10, 1/t)

    evaluator = DeepSeaExactValue(size=config['DEEPSEA_SIZE'], unscaled_move_cost=0.01, gamma = config['GAMMA'], episodic = config['EPISODIC'])

    if config['EPISODIC']: 
        gae_fn = helpers.calculate_i_and_e_gae_two_critic_episodic
        trace_fn = helpers._get_all_traces
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov(z, phi, phi_prime, done, config['GAMMA'])
    else:
        gae_fn = helpers.calculate_i_and_e_gae_two_critic
        trace_fn = helpers._get_all_traces_continuing
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov_continuing(z, phi, phi_prime, done, config['GAMMA'])

    k = config.get('RND_FEATURES', 128)
    if config.get('OPTIMISTIC_INIT', False):
        V_target = 1/(1-config['GAMMA'])
        b_init = jnp.zeros(k).at[-1].set(V_target * config['A_REGULARIZATION'])
    else: 
        b_init = jnp.zeros(k)

    def get_int_rew(S, features, N):
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features)
        bonus_sq /= jnp.maximum(1.0, N)
        rho = config['BONUS_SCALE'] * jnp.sqrt(bonus_sq)
        return rho
    
    def interpolate_lstd_val(lstd_state, ri, phi_fn=None, obs=None, phi=None):
        """
        Returns a convex combination of the LSTD solution and a maximal possible intrinsic value.
        Math: V = weight_lstd * V_lstd + (1 - weight_lstd) * V_max
        """
        # --- 1. Common Setup ---
        if phi is not None:
            features = phi
        elif phi_fn is not None and obs is not None:
            features = phi_fn(obs)
        else:
            assert False, 'Must provide either phi function and obs OR phi'

        v_lstd = features @ lstd_state["w"]
        ri_unscaled = ri / config['BONUS_SCALE']
        
        # Calculate V_max
        ri_min = jnp.minimum(1.0, jnp.max(ri))
        default_vmax = ri_min / (1 - config['GAMMA'])
        v_max = config.get('V_MAX', default_vmax)

        if config.get('VMAX_INTERPOLATE_LINEAR', False):
            N0 = config.get('EFFECTIVE_VISITS_TO_REMAIN_OPT', 10)
            N_eff = 1.0 / (ri_unscaled ** 2 + 1e-8)
            weight_lstd = jnp.clip(N_eff / N0, 0.0, 1.0)
        else:
            eps = jnp.clip(ri_unscaled, 0.0, 1.0)
            weight_lstd = 1.0 - eps

        # --- 3. Convex Combination ---
        V = weight_lstd * v_lstd + (1.0 - weight_lstd) * v_max
        return V
    
    def lstd_batch_update(lstd_state: Dict,
                          transitions, 
                          features: jnp.ndarray,
                          next_features: jnp.ndarray,
                          traces: jnp.ndarray):        
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']
        t = lstd_state['t']
        α = alpha_fn(lstd_state['t']) 
        reward_scale = 1.0 / jnp.sqrt(N)
        rho = transitions.intrinsic_reward / reward_scale
        
        # LSTD updates using the TRAINABLE features and their traces
        A_update = jax.vmap(jax.vmap(cross_cov))(traces, features, next_features, transitions.done) 
        A_b = A_update.mean(axis=batch_axes)
        
        A, b_optimistic, b_pessimistic = lstd_state['A'], lstd_state['b_optimistic'], lstd_state['b_pessimistic']
        b_int_sample = traces * rho[..., None] 
        b_b = b_int_sample.mean(axis=batch_axes)
        
        # EMAs
        def EMA(α, x_start, x_sample):
            return (1-α) * x_start + α * x_sample
        
        A = EMA(α, A, A_b)
        b_optimistic = EMA(α, b_optimistic, b_b)
        b_pessimistic = EMA(α, b_pessimistic, b_b)

        εI = config['A_REGULARIZATION_PER_STEP'] * jnp.eye(A.shape[0])

        w_optimistic = jnp.linalg.solve(A + εI, b_optimistic) * reward_scale
        w_pessimistic = jnp.linalg.solve(A + εI, b_pessimistic) * reward_scale
        
        return {'A': A, 'b_optimistic': b_optimistic, 'b_pessimistic': b_pessimistic, 'w_optimistic': w_optimistic, 'w_pessimistic': w_pessimistic, 'N': N, 't': t+1}
    
    # --- Feature Regression Loss ---
    def rnd_feature_loss(params, rnd_net, obs, target_trace, old_features):
            """
            Pure Cosine Similarity Loss.
            Minimizes 1 - CosineSimilarity.
            """
            # 1. Predict features (Normalized by RND_Net)
            predicted_features = rnd_net.apply(params, obs)

            cos1 = jax.vmap(helpers.cosine_similarity)(predicted_features, target_trace)
            # 4. Calculate Clipped Prediction
            # Note: This vector is technically slightly off the unit sphere, 
            # but for small eps (0.1-0.2) it is close enough.
            predicted_features_clipped = old_features + jnp.clip(
                predicted_features - old_features, 
                -config["CLIP_EPS"], 
                config["CLIP_EPS"]
            )
            cos2 = jax.vmap(helpers.cosine_similarity)(predicted_features_clipped, target_trace)

            further = jnp.minimum(cos1, cos2) # whichever one is a worse predictor - the predicted phi or predicted phi within a region.
            loss = 2 - 2 * further # cosine similarity = 1 -> 0 loss. Cosine similarity = 0, loss = 2
            return (1/2) * loss.mean()
    
    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = initialize_rnd_network(rnd_rng, obs_shape, config, k)
        _, target_params = initialize_rnd_network(target_rng, obs_shape, config, k)
            
        # initialize value and policy network
        network, network_params = initialize_actor_critic(rng, obs_shape, n_actions, config, n_heads=2)
        
        # Initialize Train States (including updated RND state with optimizer)
        train_state, rnd_state = initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        # Helper wrappers
        get_target_features_fn = lambda obs: rnd_net.apply(rnd_state.target_params, obs)
        batch_get_target_features = jax.vmap(get_target_features_fn)
        
        # Important: This uses the TRAINABLE params
        get_train_features_fn = lambda params, obs: rnd_net.apply(params, obs)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        
        # Initialize LSTD State
        initial_lstd_state = {
            'A': jnp.eye(k) * config['A_REGULARIZATION'], 
            'b_optimistic': b_init,
            'b_pessimistic': jnp.zeros(k), 
            'w_optimistic': jnp.zeros(k),
            'w_pessimistic': jnp.zeros(k),
            'N': 0, 't': 1,
        }
        # Initialize Sigma State (for reward bonus calculation)
        initial_sigma_state = {
            'S': jnp.eye(k),
            'N': 0, 't': 1,
        }

        # WARMUP
        def _warmup_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            rng, _rng = jax.random.split(rng)
            rng_action = jax.random.split(_rng, config["NUM_ENVS"])
            action = jax.vmap(env.action_space(env_params).sample)(rng_action)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                rng_step, env_state, action, env_params
            )
            return (env_state, obsv, rng), None

        warmup_runner_state = (env_state, obsv, rng)
        (env_state, obsv, rng), _ = jax.lax.scan(
            _warmup_step, warmup_runner_state, None, config["WARMUP"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # --- COLLECT TRAJECTORIES ---
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Get LSTD features using CURRENT TRAINABLE PARAMS
                current_train_features_fn = lambda obs: get_train_features_fn(rnd_state.params, obs)
                
                phi_target = get_target_features_fn(last_obs)
                # get i_val (dot product with LSTD state and Trainable features)
                i_val_fast = interpolate_lstd_val(current_train_features_fn, last_obs, lstd_state)
                i_val_slow = lstd_i_val_slow(current_train_features_fn, last_obs, lstd_state)


                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                intrinsic_reward = jnp.zeros_like(reward) 
                transition = Transition(
                    done, action, value, i_val_fast, i_val_slow, phi_target, reward, intrinsic_reward, 
                    log_prob, last_obs, obsv, info 
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # --- Process Features for Reward & LSTD ---
            
            # 1. Target Features (Fixed) -> Used for Intrinsic Reward & Regression Target
            # phi_target is already in traj_batch, next_phi_target needed for reward bonus
            next_phi_target = batch_get_target_features(traj_batch.next_obs)
            phi_fixed = batch_get_target_features(traj_batch.obs)
            
            # Update Sigma (Reward Covariance) using Fixed Features
            sigma_state = helpers.sigma_update(sigma_state, traj_batch, phi_fixed, GET_ALPHA_FN_cov(sigma_state['t']))
            
            # Calculate Intrinsic Reward using Fixed Features
            int_rew_from_features = lambda features: get_int_rew(sigma_state['S'], features, sigma_state['N'])
            rho = int_rew_from_features(next_phi_target)
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            
            # Calculate Traces of Fixed Features (The Regression Target)
            target_traces = trace_fn(traj_batch, phi_fixed, config['GAMMA'], config['GAE_LAMBDA'])

            # 2. Trainable Features (Learned) -> Used for LSTD System
            current_train_features_fn = lambda obs: get_train_features_fn(rnd_state.params, obs)
            batch_get_train_features = jax.vmap(current_train_features_fn)
            
            next_phi_train = batch_get_train_features(traj_batch.next_obs)
            
            # Traces of Trainable Features (For LSTD Update)
            train_traces = trace_fn(traj_batch, traj_batch.phi_train, config['GAMMA'], config['GAE_LAMBDA'])

            # Update LSTD using Trainable Features
            lstd_state = lstd_batch_update(lstd_state, traj_batch, traj_batch.phi_train, next_phi_train, train_traces)

            # --- Advantage Calculation ---
            _, last_val = network.apply(train_state.params, last_obs)
            last_i_val_fast = interpolate_lstd_val(current_train_features_fn, last_obs, lstd_state)
            gaes, targets = gae_fn(traj_batch, last_val, last_i_val_fast, config["GAMMA"], config["GAE_LAMBDA"])
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            # --- UPDATE NETWORKS ---        
            def _update_epoch(update_state, unused):
                
                def _update_minbatch(scan_state, batch_info):
                    train_state, rnd_state = scan_state
                    # Unpack: batch_info contains PPO data, Trace Targets, AND Old Features
                    traj_batch_mini, advantages_mini, targets_mini, target_traces_mini, old_features_mini = batch_info
                    
                    # 1. Update PPO Agent
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch_mini, advantages_mini, targets_mini, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    # 2. Update RND Features (Feature Learning) with CLIPPING
                    rnd_grad_fn = jax.value_and_grad(rnd_feature_loss)
                    rnd_loss, rnd_grads = rnd_grad_fn(
                        rnd_state.params, 
                        rnd_net, 
                        traj_batch_mini.obs, 
                        target_traces_mini,
                        old_features_mini # <--- old features for clipping
                    )
                    rnd_state = rnd_state.apply_gradients(grads=rnd_grads)

                    return (train_state, rnd_state), (total_loss, rnd_loss)
                # end _update_minibatch
                
                train_state, rnd_state, traj_batch, advantages, targets, target_traces, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Bundle the 'old_features' (traj_batch.phi_train) into the batch
                batch = (traj_batch, advantages, targets, target_traces, traj_batch.phi_train)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                (train_state, rnd_state), (ppo_loss, rnd_feat_loss) = jax.lax.scan(_update_minbatch, (train_state, rnd_state), minibatches)
                
                update_state = (train_state, rnd_state, traj_batch, advantages, targets, target_traces, rng)
                return update_state, (ppo_loss, rnd_feat_loss)
            # end _update_epoch

            initial_update_state = (train_state, rnd_state, traj_batch, advantages, extrinsic_target, target_traces, rng)
            update_state, (ppo_losses, rnd_losses) = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, rnd_state, _, _, _, _, rng = update_state
            
            # --- METRICS ---
            metric = {k: v.mean() for k, v in traj_batch.info.items()}
            
            # Re-bind for metric evaluation
            eval_train_features_fn = lambda obs: get_train_features_fn(rnd_state.params, obs)

            v_e, v_i, v_pred = evaluator.compute_true_values(network, train_state.params, batch_get_target_features, int_rew_from_features)
            
            # Evaluator uses LSTD on TRAINABLE features
            v_i_pred_fast = evaluator.get_value_grid(interpolate_lstd_val(eval_train_features_fn, evaluator.obs_stack, lstd_state))
            v_i_pred_slow = evaluator.get_value_grid(lstd_i_val_slow(eval_train_features_fn, evaluator.obs_stack, lstd_state))
            
            ri_grid = evaluator.get_value_grid(int_rew_from_features(batch_get_target_features(evaluator.obs_stack)))

            metric.update({
                "ppo_loss": ppo_losses.mean(), 
                "rnd_feature_loss": rnd_losses.mean(), # Loss for learning the trace
                "feat_norm": jnp.linalg.norm(next_phi_target, axis=-1).mean(),
                "bonus_mean": gaes[1].mean(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "ri_grid": ri_grid,
                "v_i": v_i,
                "v_e": v_e,
                "v_e_pred": v_pred,
                "fast_v_i_pred": v_i_pred_fast,
                "slow_v_i_pred": v_i_pred_slow,
                "e_value_error": jnp.mean(evaluator.reachable_mask * (v_e - v_pred)**2),
                "fast_i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - v_i_pred_fast)**2),
                "slow_i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - v_i_pred_slow)**2)
            })
            runner_state = (train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric

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
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--base-config', type = str, default = 'mc', choices = ['mc', 'ds', 'min'])
    args = parser.parse_args()
    
    if args.base_config == 'mc':
        config = configs.mc_config.copy()
        raise AssertionError('conv_net_v.py only has value solver implemented for DeepSea')
    elif args.base_config == 'ds':
        config = configs.ds_config.copy()
    elif args.base_config  == 'min':
        config = configs.min_config.copy()
    
    # Override with command line config
    config_override = parse_config_override(args.config)
    config.update(config_override)
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
        intrinsic_rew_mean = metrics['intrinsic_rew_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['intrinsic_rew_mean']
        fast_i_value_error = metrics['fast_i_value_error'].mean(0) if config['N_SEEDS'] > 1 else metrics['fast_i_value_error']
        slow_i_value_error = metrics['slow_i_value_error'].mean(0) if config['N_SEEDS'] > 1 else metrics['slow_i_value_error']
        e_value_error = metrics['e_value_error'].mean(0) if config['N_SEEDS'] > 1 else metrics['e_value_error']
        
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, bonus_mean[1:], 'i_advantage_mean')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsic_rew_mean[1:], 'intrinsic_rew_mean')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, fast_i_value_error[1:], 'fast_i_val_mse')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, slow_i_value_error[1:], 'slow_i_val_mse')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, e_value_error[1:], 'e_val_mse')

        # List of new value metrics to plot
        value_metrics = ["v_i", "v_e", "v_e_pred", "fast_v_i_pred", "slow_v_i_pred"]

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