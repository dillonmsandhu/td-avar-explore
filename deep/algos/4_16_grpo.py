# REINFORCE / GRPO-style policy gradient (for intrinsic value)
# uses a timestep dependent variant, based on batch index i.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "4_16_grpo"

class Transition(NamedTuple):
    done: jnp.ndarray
    goal: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    # Episodic / Continuing / Absorbing
    is_episodic = config.get("EPISODIC", True)
    is_continuing = (not is_episodic)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    k_lstd = config.get("RND_FEATURES", 128)
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    # evaluator = helpers.initialize_evaluator(config)
    evaluator = None 

    if config.get('SCHEDULE_BETA', False):
        beta_sch = helpers.make_triangle_schedule(total_updates = config['NUM_UPDATES'], max_beta=config['BONUS_SCALE'], peak_at=0.01) 
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    def train(rng):
        
        initial_sigma_state = {"S": jnp.eye(k_lstd, dtype=jnp.float64)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_lstd
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_lstd
        )
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)
        def get_phi(obs_batch):
            def _step(unused, x):
                x_input = jnp.expand_dims(x, 0) if x.ndim == 3 else x
                features = rnd_net.apply(target_params, x_input)
                return None, jnp.squeeze(features, 0) if x.ndim == 3 else features
            
            _, phi = jax.lax.scan(_step, None, obs_batch)
            return phi

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                target_next_obs = info["real_next_obs"].reshape(last_obs.shape)
                is_goal = info['is_goal']
                dummy = jnp.zeros_like(reward)

                transition = Transition(
                    done, is_goal, action, value, dummy, dummy, dummy, reward, dummy, log_prob, last_obs, target_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # Feature Extraction for Current Batch
            phi = batch_get_features(traj_batch.obs)
            next_phi = batch_get_features(traj_batch.next_obs)
            terminals = jnp.where(not is_continuing, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.goal, 0)

            # --- GLOBAL COVARIANCE UPDATE (Pure Accumulation) --
            sigma_state = helpers.update_cov(traj_batch, sigma_state, phi, next_phi)            
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_lstd))

            rho = helpers.get_scale_free_bonus(Sigma_inv, next_phi)
            traj_batch = traj_batch._replace(intrinsic_reward = rho) # GRPO uses pure rho.
            
            # --- Absorbing overwrite ---
            # exact_terminal_i_val = rho / (1.0 - config["GAMMA_i"])
            # overwrite_val = jnp.logical_and(traj_batch.goal, is_absorbing)
            # fixed_next_i_val = jnp.where(overwrite_val, exact_terminal_i_val, 0.0)

            # # --- Final traj_batch update for GAE ---
            # traj_batch = traj_batch._replace(
            #     intrinsic_reward=rho, 
            #     next_i_val=fixed_next_i_val
            # )

            # --- ADVANTAGE CALCULATION ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"],
                is_continuing,
                γi=config["GAMMA_i"], 
                λi=1.0  # Forces pure Monte Carlo returns
            )
            
            extrinsic_adv = gaes[0]
            extrinsic_target = targets[0]
            
            # raw_intrinsic_returns has shape (NUM_STEPS, NUM_ENVS)
            raw_intrinsic_returns = gaes[1] 
            
            # --- THE TIME-DEPENDENT BASELINE ---
            # Compute the mean and std across the NUM_ENVS axis (axis=1).
            # This creates a unique baseline for EVERY timestep in the rollout chunk.
            mean_int_ret_t = jnp.mean(raw_intrinsic_returns, axis=1, keepdims=True)
            
            # Normalize conditionally based on the timestep index
            baselined_int_adv = (raw_intrinsic_returns - mean_int_ret_t)
            
            # Final combined advantage for the Actor
            rho_scale = beta_sch(idx)
            advantages = extrinsic_adv + (rho_scale * baselined_int_adv)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    
                    (total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    return train_state, (total_loss, value_loss, loss_actor, entropy)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                # loss_info is now a tuple of 4 arrays: (total_loss, value_loss, loss_actor, entropy)
                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), loss_info

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            # Shared Metrics
            metric.update(
                {
                    "total_loss": loss_info[0].mean(),
                    "value_loss": loss_info[1].mean(),
                    "actor_loss": loss_info[2].mean(),
                    "entropy": loss_info[3].mean(),
                    "feat_norm": jnp.linalg.norm(next_phi, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                    "mean_rew": traj_batch.reward.mean(),
                    "rho_scale": rho_scale,
                    "vi_pred": traj_batch.i_value.mean(),
                    "v_e_pred": traj_batch.value.mean(),
                    "num_goals": jnp.sum(traj_batch.goal)
                }
            )

            runner_state = (train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
