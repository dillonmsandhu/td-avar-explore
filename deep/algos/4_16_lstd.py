from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.buffer import LSTDBufferState, FeatureTraceBufferManager
from core.lstd import solve_lstd_lambda_from_buffer
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "4_16_lstd" # performs the update to the covariance matrix FIRST.

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
    k_lstd = config.get("LSTD_FEATURES", 128)
    k_rho = config.get("RHO_FEATURES", 128)

    # Episodic / Continuing / Absorbing
    is_episodic = config.get("EPISODIC", True)
    is_continuing = not is_episodic
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    overwrite_absorbing_gae = config.get("USE_ABSORBING_OVERWRITE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'
    
    # Replay Buffer
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    config['CHUNK_SIZE'] =  100_000 + batch_size# chunking for LSTD solver
    buffer_manager = FeatureTraceBufferManager(config, k_lstd, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE'], k_rho) # stateless buffer manager.
    config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    config['PADDED_CAPACITY'] = buffer_manager.padded_capacity
    
    # Env
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)
    
    if config.get('SCHEDULE_BETA', False):
        # goes up until peak and then linearly decays to 0.
        beta_sch = helpers.make_triangle_schedule(total_updates = config['NUM_UPDATES'], max_beta=config['BONUS_SCALE'], peak_at=0.0) 
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    print(f'LSTD Net has {k_lstd} features, Normalization is {config["NORMALIZE_LSTD_FEATURES"]} network type is {config["RND_NETWORK_TYPE"]}, and bias is {config["LSTD_BIAS"]}')
    print(f'Rho Net has {k_rho} features, Normalization is {config["NORMALIZE_RHO_FEATURES"]} network type is {config["RND_NETWORK_TYPE"]}, and bias is {config["BIAS"]}')

    # Metrics Function
    def _compile_metrics(network, batch_get_rho_features, batch_get_lstd_features,  traj_batch, next_phi, loss_info, gaes, targets, rho_scale, Sigma_inv, lstd_state, train_state):
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
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
                "rho_scale": rho_scale,
                "num_goals": jnp.sum(traj_batch.goal)
            })

            if evaluator is None:
                metric.update({
                    "vi_pred": traj_batch.i_value.mean(),
                    "v_e_pred": traj_batch.value.mean(),
                })
            else:
                int_rew_from_state = lambda s: helpers.get_scale_free_bonus(Sigma_inv, batch_get_rho_features(s)) * rho_scale
                get_vi = lambda obs: batch_get_lstd_features(obs) @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(
                    config, metric, int_rew_from_state, evaluator, rho_scale,
                    network, train_state, traj_batch, get_vi,
                )
            return metric

    def train(rng):
        initial_lstd_state = {"w": jnp.zeros(k_lstd), }
        initial_buffer_state = buffer_manager.init_state()
        initial_sigma_state = {"S": config.get('GRAM_REG', 1.0) * jnp.eye(k_rho, dtype=jnp.float64)} # global accumulation

        # Initialize Feature Maps
        rnd_rng, rng = jax.random.split(rng)
        # Normalized keeps rho between 0 and 1, bias ensures sigma keeps track of total count.
        rho_net, rho_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config["NORMALIZE_RHO_FEATURES"], bias=config['BIAS'], k=k_rho 
        )
        lstd_net, lstd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config["NORMALIZE_LSTD_FEATURES"], bias=config['LSTD_BIAS'], k=k_lstd 
        )

        def get_rho_feats(obs): return rho_net.apply(rho_params, obs)
        def get_lstd_feats(obs): return lstd_net.apply(lstd_params, obs)
        batch_get_rho_features = jax.vmap(get_rho_feats)
        batch_get_lstd_features = jax.vmap(get_lstd_feats)
        
        # Policy and extrinsic value networks
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rho_net, network_params, rho_params
        )
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, rng, idx = runner_state

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
                is_goal = info['is_goal']
                target_next_obs = jnp.where(is_continuing, obsv, info["real_next_obs"].reshape(last_obs.shape))  #because Gymnax has no transition S_T -> S_0, the continuing formulation requires this.
                next_val = network.apply(train_state.params, target_next_obs, method=network.value)

                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                next_i_val = jnp.zeros_like(reward)

                transition = Transition(
                    done, is_goal, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, target_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # Post-Process batch
            phi = batch_get_lstd_features(traj_batch.obs)
            next_phi = batch_get_lstd_features(traj_batch.next_obs)
            terminals = jnp.where(not is_continuing, traj_batch.done, 0) # indicates a transition INTO S_T.
            absorb_masks = jnp.where(is_absorbing, traj_batch.goal, 0) # indicates a transition into S_T where S_T is a goal.
            traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["LSTD_LAMBDA_i"], is_continuing)
            
            # --- 0. UPDATE COVARIANCE SUM MATRIX ---
            rho_feats = batch_get_rho_features(traj_batch.obs)
            next_rho_feats = batch_get_rho_features(traj_batch.next_obs)
            sigma_state = helpers.update_cov(traj_batch, sigma_state, rho_feats, next_rho_feats)          

            # --- 1. UPDATE EXTENDED BUFFER ---f
            buffer_batch = LSTDBufferState(traces, phi, next_phi, next_rho_feats, terminals, absorb_masks, size=jnp.array(batch_size))
            buffer_state = buffer_manager.update_buffer(buffer_state, buffer_batch)
            
            # --- 2. GLOBAL COVARIANCE UPDATE (Pure Accumulation) --
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_rho))

            # --- 3. SOLVE LSTD ON BUFFER ---
            lstd_state = solve_lstd_lambda_from_buffer(buffer_state, Sigma_inv, lstd_state, config)

            # --- 4. EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, prb_rng)
            
            # --- 5. COMPUTE TARGETS ---
            rho = helpers.get_scale_free_bonus(Sigma_inv, next_rho_feats)
            
            # --- LSTD PREDICTIONS ---
            v_i = phi @ lstd_state["w"] 
            next_v_i = next_phi @ lstd_state["w"] 
            
            # --- Clip ---
            V_max_raw = 1.0 / (1.0 - config['GAMMA_i'])
            v_i, next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, V_max_raw), (v_i, next_v_i))
            
            exact_terminal_i_val = rho / (1.0 - config["GAMMA_i"])
            should_apply_mask = traj_batch.goal & is_absorbing & overwrite_absorbing_gae
            # Seamlessly select between the network prediction and the analytical value
            fixed_next_i_val = jnp.where(should_apply_mask, exact_terminal_i_val, next_v_i)
            traj_batch = traj_batch._replace(
                i_value=v_i, 
                intrinsic_reward=rho, 
                next_i_val=fixed_next_i_val
            )

            gaes, targets = helpers.calculate_gae(
                traj_batch, 
                config["GAMMA"], 
                config["GAE_LAMBDA"], 
                is_continuing, 
                γi=config["GAMMA_i"], 
                λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            
            # --- 6. INTRINSIC vs. EXTRINSIC SCALING ---
            rho_scale = beta_sch(idx) # triangle schedule
            advantages = gae_e + (rho_scale * gae_i)
            extrinsic_target = targets[0]

            # 7. UPDATE NETWORK
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
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = _compile_metrics(
                network, batch_get_rho_features, batch_get_lstd_features, traj_batch, next_phi, loss_info, gaes, targets, rho_scale, Sigma_inv, lstd_state, train_state
            )

            runner_state = (train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
