# Covariance-Based Intrinsic Reward, propegated by an intrinsic value net
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
SAVE_DIR = 'cov_grpo'

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
    obs_shape = env.observation_space(env_params).shape
    k = config.get('RND_FEATURES', 128)
    GET_ALPHA_FN = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)

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
        rho = config['BONUS_SCALE'] * jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
        return rho
    
    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], k)
            
        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

        initial_sigma_state = {
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
                intrinsic_reward = jnp.zeros_like(reward)  # placeholder, will be filled later
                i_val = jnp.zeros_like(reward)  # placeholder, will be filled later
                transition = Transition(
                    done,
                    action,
                    value,
                    i_val,
                    reward,
                    intrinsic_reward,
                    log_prob,
                    last_obs,
                    info["real_next_obs"],
                    info,
                )

                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            # GRPO: all envs start at the same state. A "group" is a timestep across all parallel envs.
            rng, _rng = jax.random.split(rng)
            random_idx = jax.random.randint(_rng, (), 0, config["NUM_ENVS"])
            env_state = jax.tree.map(lambda x : x[random_idx], env_state)
            env_state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (config["NUM_ENVS"], *x.shape)), 
                env_state
            )
            last_obs = jnp.broadcast_to(last_obs[random_idx], last_obs.shape)
            initial_obs = last_obs
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # -------------------------------------------------------------
            # --------- Update Sigma and compute intrinsic reward ---------
            initial_obs_expanded = jnp.expand_dims(initial_obs, axis=0)
            all_encountered_obs = jnp.concatenate([initial_obs_expanded, traj_batch.next_obs], axis=0)
            all_phi = batch_get_features(all_encountered_obs)
            sigma_state = helpers.sigma_update( sigma_state, traj_batch, all_phi, α=GET_ALPHA_FN(sigma_state['t']),)
            # COMPUTE intrinsic reward:            
            int_rew_from_features = lambda features: get_int_rew(sigma_state['S'], features, sigma_state['N'] + config['NUM_ENVS'])
            
            next_phi = all_phi[1:]
            rho = int_rew_from_features(next_phi)

            def standardize(x):
                return (x - x.mean()) / (1e-8 + x.std())
            
            rho = jax.lax.cond(config['STANDARDIZE_RHO'],
                               lambda x: standardize(x),
                               lambda x: x,
                               operand = rho,
                               )
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            
            # Advantage
            _, last_val = network.apply(train_state.params, last_obs)
            gaes, targets = gae_fn(traj_batch, last_val, jnp.zeros_like(last_val), config["GAMMA"], config['GAE_LAMBDA'], λi=1.0, γi = config["GAMMA_i"])
            e_gae, i_gae = gaes
            # Average Return across this timestep across all batches ~ V
            # i_gae = (i_gae - i_gae.mean(1, keepdims=True)) / (i_gae.std(1, keepdims=True) + 1e-8) # shape is (num_steps, num_envs, 1)
            i_gae = (i_gae - i_gae.mean(1, keepdims=True))  # shape is (num_steps, num_envs, 1)
            
            i_gae = jax.lax.cond(config["STANDARDIZE_I_GAE"], lambda x: x / x.std(1, keepdims=True), lambda x: x, i_gae)
            
            e_gae = jax.lax.cond(config["STANDARDIZE_E_GAE"], 
                                 lambda x: (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True), 
                                 lambda x: x, 
                                 e_gae
            )
            gaes = (e_gae, i_gae)
            advantages = i_gae + e_gae
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
            
            # -------------------------------
            # --------- Update metrics ------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }

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

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
