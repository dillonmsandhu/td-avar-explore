# Covariance-Based Intrinsic Reward, propegated by an intrinsic value net
from imports import *
import helpers
import networks
from envs.deepsea_v import DeepSeaExactValue
SAVE_DIR = 'cov_net'


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
    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    
    alpha_fn_cov = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    if config.get('DECAY_BONUS', True):
        beta_fn = lambda n: config['BONUS_SCALE'] * (1 - (n / config['TOTAL_TIMESTEPS']))
    else:
        beta_fn = lambda n: config['BONUS_SCALE']

    if config['EPISODIC']: 
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
    else:
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic
    # --- Setup Evaluator (Only if requested) ---
    calc_true_values = config.get('CALC_TRUE_VALUES', False)
    if calc_true_values:
        evaluator = DeepSeaExactValue(
            size=config['DEEPSEA_SIZE'], 
            unscaled_move_cost=0.01, 
            gamma=config['GAMMA'], 
            episodic=config['EPISODIC']
        )
    def get_int_rew(S, features, N):
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features) / jnp.maximum(1.0, N)
        rho = beta_fn(N) * jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
        return rho
    
    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config)
            
        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, config, n_heads=3)
        dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
        dummy_phi = rnd_net.apply(target_params, dummy_obs)
        k = dummy_phi.shape[-1]
        initial_sigma_state = {
            'S_long': jnp.eye(k) * config['GRAM_REG'],
            'S': jnp.eye(k) * config['GRAM_REG'],
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
            
            train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng= env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value, i_val = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

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

            # -------------------------------------------------------------
            # --------- Update Sigma and compute intrinsic reward ---------
            phis = batch_get_features(traj_batch.obs)
            sigma_state = helpers.sigma_update(sigma_state, traj_batch, phis, α=alpha_fn_cov(sigma_state['t']),)
            
            # COMPUTE intrinsic reward:            
            int_rew_from_features = lambda features: get_int_rew(sigma_state['S'], features, sigma_state['N'])
            next_phi = batch_get_features(traj_batch.next_obs)
            rho = int_rew_from_features(batch_get_features(traj_batch.obs))
            traj_batch = traj_batch._replace(intrinsic_reward=rho)     
            
            # Advantage
            _, last_val, last_i_val = network.apply(train_state.params, last_obs)
            gaes, targets = gae_fn(traj_batch, last_val, last_i_val, config["GAMMA"], config["GAE_LAMBDA"])
            advantages = gaes[0] + gaes[1]

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
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

                    # --- UPDATE PPO ---
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (i_value_loss, value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    return train_state, (total_loss, i_value_loss, value_loss, loss_actor, entropy)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses
            # --------- Train the network ---------
            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state
            # -------------------------------
            # --------- Update metrics ------
            metric = {k: v.mean() for k, v in traj_batch.info.items()} # performance
            metric.update({
                "ppo_loss": loss_info[0].mean(), 
                "i_value_loss": loss_info[1].mean(),
                "e_value_loss": loss_info[2].mean(),
                "pi_loss": loss_info[3].mean(),
                "entropy": loss_info[4].mean(),
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
            # Branch: Expensive True Values vs. Cheap Proxies
            if calc_true_values:
                # def compute_true_values(self, network: Any, params: PyTree,lstd_state: Dict, get_features: Callable, get_int_rew: Callable
                v_e, v_i, v_pred = evaluator.compute_true_values(network, train_state.params, batch_get_features, int_rew_from_features)
                v_pred, v_i_pred = v_pred
                ri_grid_vals = int_rew_from_features(batch_get_features(evaluator.obs_stack))
                ri_grid = evaluator.get_value_grid(ri_grid_vals)

                metric.update({
                    "ri_grid": ri_grid,
                    "vi_pred": v_i_pred,
                    "v_i": v_i,
                    "v_e": v_e,
                    "v_e_pred": v_pred,
                    "e_value_error": jnp.mean(evaluator.reachable_mask * (v_e - v_pred)**2),
                    "i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - v_i_pred)**2),
                })
            else:
                metric.update({
                    "vi_pred": traj_batch.i_value.mean(),
                    "v_e_pred": traj_batch.value.mean()
                })
            runner_state = (train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric
            # end update_step

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
    
def main():
    import warnings; warnings.simplefilter('ignore')
    from utils import evaluate, parse_config_override
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
    # NEW: Argument to take a list of environments
    parser.add_argument('--env_ids', nargs='+', default=[], 
                       help='Optional list of envs to run sequentially. If provided, overrides the config ENV_NAME.')

    args = parser.parse_args()
    
    # 1. Load Base Config
    if args.base_config == 'mc':
        config = configs.mc_config.copy()
        # raise AssertionError('conv_net_v.py only has value solver implemented for DeepSea') 
        # (Commented out assertion just in case you want to try others)
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