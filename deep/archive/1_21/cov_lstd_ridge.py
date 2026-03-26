# Covariance-Based Intrinsic Reward, propegated by LSTD.
# Consolidated version: Handles both standard training and ExactValue logging via config.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
SAVE_DIR = 'cov_lstd'

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

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    REG_DECAY_PCT = config.get("REG_DECAY_PCT", 1.0)
    REG_INIT = config.get("A_REG_INIT", 1e-1)
    REG_FINAL = config.get("A_REGULARIZATION_PER_STEP", 1e-4)
    
    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    
    def eps_fn(t):
        timescale = REG_DECAY_PCT * config['NUM_UPDATES']
        progress = jnp.clip(t / timescale, 0, 1)
        eps = progress * REG_FINAL + (1-progress) * REG_INIT
        return eps
    
    if config.get('DECAY_BETA', False): 
        beta_fn = helpers.make_beta_schedule(config)
    else:
        beta_fn = lambda n: config['BONUS_SCALE']
    
    # --- Flag to enable heavy exact value calculation ---
    calc_true_values = config.get('CALC_TRUE_VALUES', False)

    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    if calc_true_values:
        evaluator = DeepSeaExactValue(
            size=config['DEEPSEA_SIZE'], 
            unscaled_move_cost=0.01, 
            gamma=config['GAMMA'], 
            episodic=config['EPISODIC']
        )

    if config['EPISODIC']: 
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
        trace_fn = helpers._get_all_traces
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov(z, phi, phi_prime, done, config['GAMMA'])
    else:
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic
        trace_fn = helpers._get_all_traces_continuing
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov_continuing(z, phi, phi_prime, done, config['GAMMA'])

    k = config.get('RND_FEATURES', 128)

    def get_int_rew(S, features, N):
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features)
        bonus_sq /= jnp.maximum(1.0, N)
        rho = beta_fn(N) * jnp.sqrt(bonus_sq)
        return rho
    
    
    def lstd_batch_update(lstd_state: Dict, transitions, features, next_features, traces):        
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']
        t = lstd_state['t']
        α = alpha_fn(lstd_state['t']) 
        reward_scale = 1.0 / jnp.sqrt(N)
        rho = transitions.intrinsic_reward / reward_scale
        
        A_update = jax.vmap(jax.vmap(cross_cov))(traces, features, next_features, transitions.done)
        A_b = A_update.mean(axis=batch_axes)
        
        A, b = lstd_state['A'], lstd_state['b']
        b_int_sample = traces * rho[..., None]
        b_b = b_int_sample.mean(axis=batch_axes)
        
        def EMA(α, x_start, x_sample):
            return (1-α) * x_start + α * x_sample
        
        A = EMA(α, A, A_b)
        b = EMA(α, b, b_b)
        εI = eps_fn(lstd_state['t']) * jnp.eye(A.shape[0])
        w = jnp.linalg.solve(A + εI, b) * reward_scale
        w_unreg = jnp.linalg.solve(A, b) * reward_scale
        
        return {'A': A, 'b': b, 'w': w, 'w_unreg': w_unreg ,'N': N, 't': t+1}

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config, k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config, k)
            
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        initial_lstd_state = {
            'A': jnp.eye(k), 
            'b': jnp.zeros(k), 
            'w': jnp.zeros(k),
            'w_unreg': jnp.zeros(k),
            'N': 0, 't': 1,
        }
        initial_sigma_state = {
            # 'S_long': jnp.eye(k) * config['GRAM_REG'],
            'S': jnp.eye(k) * config['GRAM_REG'],
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
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
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
            sigma_state = helpers.sigma_update(sigma_state, traj_batch, phi, alpha_fn(sigma_state['t']))
            int_rew_from_features = lambda features: get_int_rew(sigma_state['S'], features, sigma_state['N'])
            rho = int_rew_from_features(phi)
            traj_batch = traj_batch._replace(intrinsic_reward=rho)

            # Intrinsic Critic
            traces = trace_fn(traj_batch, phi, config['GAMMA'], config['GAE_LAMBDA'])
            lstd_state = lstd_batch_update(lstd_state, traj_batch, phi, next_phi, traces)

            # Intrinsic value (optimistic)
            vi = phi @ lstd_state["w"]
            traj_batch = traj_batch._replace(i_value=vi)

            # Advantage
            _, last_val = network.apply(train_state.params, last_obs)
            last_vi =  batch_get_features(last_obs) @ lstd_state['w']
            gaes, targets = gae_fn(traj_batch, last_val, last_vi, config["GAMMA"], config["GAE_LAMBDA"])
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

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state
            
            # --------- Metrics ---------
            metric = {k: v.mean() for k, v in traj_batch.info.items()}
            
            # Common Metrics
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
                "mean_rew": traj_batch.reward.mean(),
            })

            # Branch: Expensive True Values vs. Cheap Proxies
            if calc_true_values:
                # 1. Compute Exact Values using the Evaluator
                v_e, v_i, v_pred = evaluator.compute_true_values(
                    network, train_state.params, batch_get_features, int_rew_from_features
                )
                
                # 2. Compute Optimistic Grid
                all_phi = get_features_fn(evaluator.obs_stack)
                ri_grid_vals = int_rew_from_features(batch_get_features(evaluator.obs_stack))
                vi_pred_opt = evaluator.get_value_grid(all_phi @ lstd_state['w_unreg'])
                vi_pred = evaluator.get_value_grid(all_phi @ lstd_state['w'])
                ri_grid = evaluator.get_value_grid(ri_grid_vals)

                metric.update({
                    "ri_grid": ri_grid,
                    "vi_pred": vi_pred,
                    "v_i_pred_opt": vi_pred_opt,
                    "v_i": v_i,
                    "v_e": v_e,
                    "v_e_pred": v_pred,
                    "e_value_error": jnp.mean(evaluator.reachable_mask * (v_e - v_pred)**2),
                    "i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - vi_pred)**2),
                    "opt_i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - vi_pred_opt)**2),
                })
            else:
                # Use batch means as fast proxies
                vi = phi @ lstd_state["w"]
                vi_unreg = phi @ lstd_state["w_unreg"]
                metric.update({
                    "vi_pred": vi.mean(),
                    "v_i_pred_opt": vi_unreg.mean(),
                    "v_e_pred": traj_batch.value.mean()
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
    from core.utils import parse_config_override, evaluate
    import datetime
    import argparse
    import core.configs as configs
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD Explore experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values, e.g. \'{"LR": 0.001, "LAMBDA": 0.0}\'')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help=f'saves to {SAVE_DIR}/args.run_suffix/' )
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--base-config', type = str, default = 'mc', choices = ['mc', 'ds', 'min'])
    parser.add_argument('--env_ids', nargs='+', default=[], 
                       help='Optional list of envs to run sequentially. If provided, overrides the config ENV_NAME.')

    args = parser.parse_args()
    
    # 1. Load Base Config
    if args.base_config == 'mc':
        config = configs.mc_config.copy()
    elif args.base_config == 'ds':
        config = configs.ds_config.copy()
    elif args.base_config  == 'min':
        config = configs.min_config.copy()

    # 2. Apply Overrides (Global overrides applied to all envs)
    config_override = parse_config_override(args.config)
    config.update(config_override)

    # 3. Determine List of Environments to Run
    # If --env_ids is passed, use that list. Otherwise use the single one from config.
    env_list = args.env_ids if args.env_ids else [config['ENV_NAME']]

    # 4. Sequential Execution Loop
    for i, env_name in enumerate(env_list):
        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"{'='*50}")
        
        # Create a fresh config copy for this environment
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        # Generate RNG (fresh seed based on config to ensure reproducibility)
        rng = jax.random.PRNGKey(run_config['SEED'])
        
        try:
            evaluate(run_config, make_train, SAVE_DIR, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            print(e)
            import traceback
            traceback.print_exc()
            print("Continuing to next environment...")

if __name__ == '__main__':
    main()