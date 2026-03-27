# PURE ORACLE: Exact Value Policy Gradient

from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "3_26_oracle"

def add_oracle_values_to_metric(
    config, metric, ri_grid, v_e_grid, v_i_grid, traj_batch, evaluator
):
    """
    Minimalist Oracle Metrics:
    Logs the pre-calculated Bellman grids and rollout statistics.
    """
    # 1. Visitation Logging
    obs = jnp.asarray(traj_batch.obs)
    if config.get("ENV_NAME") in {"FourRooms-misc", "FourRoomsCustom-v0"}:
        if obs.ndim >= 5: # CNN
            metric['visitation_count'] = obs[..., 1].sum(axis=(0, 1))
        elif obs.ndim >= 3: # Vector
            size = int(config.get("FOURROOMS_SIZE", ri_grid.shape[0]))
            pos = obs[..., :2].astype(jnp.int32)
            y, x = pos[..., 0].reshape(-1), pos[..., 1].reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
            
    elif "DeepSea" in config.get("ENV_NAME", ""):
        # For DeepSea CNN obs, channel 0 is the agent position
        metric['visitation_count'] = obs[..., 0].sum(axis=(0, 1))
        
    elif "Chain" in config.get('ENV_NAME', '') or config.get('RND_NETWORK_TYPE') == 'identity':
        visitation = obs.sum(axis=(0, 1))
        metric['visitation_count'] = evaluator.get_value_grid(visitation)
    else:
        metric['visitation_count'] = jnp.zeros_like(ri_grid)

    # 2. Map Pre-Calculated Oracle Grids to Metrics
    metric.update({
        "ri_grid": ri_grid,
        "v_i": v_i_grid,
        "v_e": v_e_grid,
        "v_i_pred": v_i_grid,
        "v_e_pred": v_e_grid,
        "effective_visits": (config["BONUS_SCALE"] / jnp.maximum(ri_grid, 1e-8))**2,
    })

    return metric

class Transition(NamedTuple):
    done: jnp.ndarray
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

    config["VF_COEF"] = 0.0 
    
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // batch_size)

    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    alpha_fn = lambda t: jnp.maximum(config.get("MIN_COV_LR", 1 / 10), 1 / t)
    evaluator = helpers.initialize_evaluator(config)
    
    # Safety Check: Oracle requires an evaluator
    assert evaluator is not None, "Oracle requires a valid exact evaluator for the environment!"

    def get_scale_free_bonus(S_inv, features):
        """bonus = x^T Σ^{-1} X, where Σ^{-1} is the empirical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    def train(rng):
        # 1. Initialize Networks
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k,
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k,
        )
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        
        # 2. Initialize Environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        # 3. Initialize Covariance State (We only need this to compute novelty)
        initial_sigma_state = {
            "S": jnp.zeros((k,k)),
            "N": 1,
            "t": 1,
        }

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx) = runner_state

            # --- PHASE 1: COLLECT TRAJECTORIES ---
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, _ = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]

                # Dummy values to satisfy the NamedTuple. They will be overwritten by the Oracle.
                dummy_val = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, dummy_val, dummy_val, dummy_val, dummy_val, reward, dummy_val,
                    log_prob, last_obs, true_next_obs, info,
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # --- PHASE 2: COMPUTE NOVELTY REWARDS ---
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + config['GRAM_REG'] * jnp.eye(k), jnp.eye(k))
            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            
            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state["N"]))
            rho_scale = config["BONUS_SCALE"] / sqrt_n
            
            # Unscaled rho for the batch
            rho_batch = int_rew_from_features(batch_get_features(traj_batch.next_obs))

            # --- PHASE 3: ORACLE VALUE INJECTION ---
            def int_rew_from_state(s):
                """Helper for the evaluator to generate the exact intrinsic reward grid."""
                return int_rew_from_features(batch_get_features(s)) * rho_scale

            # Get Exact Values for the Entire MDP
            v_e_true_grid, v_i_true_grid, _ = evaluator.compute_true_values(
                network, train_state.params, int_rew_from_state, all = True
            )

            # Fast JAX mapping from spatial observation to grid value
            def get_oracle_values(obs_batch):
                env_name = config["ENV_NAME"]
                if "DeepSea" in env_name:
                    v_e = jnp.sum(obs_batch[..., 0] * v_e_true_grid, axis=(-2, -1))
                    v_i = jnp.sum(obs_batch[..., 0] * v_i_true_grid, axis=(-2, -1))
                elif "FourRooms" in env_name:
                    if obs_batch.ndim >= 4: # Visual Obs
                        v_e = jnp.sum(obs_batch[..., 1] * v_e_true_grid, axis=(-2, -1))
                        v_i = jnp.sum(obs_batch[..., 1] * v_i_true_grid, axis=(-2, -1))
                    else: # Vector Obs
                        y, x = obs_batch[..., 0].astype(jnp.int32), obs_batch[..., 1].astype(jnp.int32)
                        v_e, v_i = v_e_true_grid[y, x], v_i_true_grid[y, x]
                elif "Chain" in env_name:
                    v_e = jnp.sum(obs_batch * v_e_true_grid, axis=-1)
                    v_i = jnp.sum(obs_batch * v_i_true_grid, axis=-1)
                else:
                    v_e, v_i = jnp.zeros(obs_batch.shape[:2]), jnp.zeros(obs_batch.shape[:2])
                return v_e, v_i

            true_v_e, true_v_i = get_oracle_values(traj_batch.obs)
            true_next_v_e, true_next_v_i = get_oracle_values(traj_batch.next_obs)

            # Overwrite with the oracle
            traj_batch = traj_batch._replace(
                value=true_v_e,
                next_value=true_next_v_e,
                i_value=true_v_i,
                next_i_val=true_next_v_i,
                intrinsic_reward=rho_batch * rho_scale
            )

            # --- PHASE 4: GAE CALCULATION ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
                is_episodic=is_episodic, is_absorbing=is_absorbing,
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            # --- PHASE 5: UPDATE POLICY NETWORK ---
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    # Because config["VF_COEF"] is 0.0, the value loss gradients are entirely ignored
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config,
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
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            
            # --- PHASE 6: UPDATE COVARIANCE ---
            _, sigma_state, _ = helpers.update_cov_and_get_rho(
                traj_batch, sigma_state, batch_get_features, int_rew_from_features, alpha_fn,
            )

            # --- Metrics ---
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            metric.update({
                "ppo_loss": loss_info[0],
                "bonus_mean": gaes[1].mean(),
                "bonus_std": gaes[1].std(),
                "bonus_max": gaes[1].max(),
                "lambda_ret_mean": targets[0].mean(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "mean_rew": traj_batch.reward.mean(),
                "rho_scale": rho_scale,
                "vi_pred": v_i_true_grid,
                "v_e_pred": v_e_true_grid
            })

            ri_grid = evaluator.get_value_grid(int_rew_from_state(evaluator.obs_stack)) * rho_scale
            metric = add_oracle_values_to_metric(
                config=config,
                metric=metric,
                ri_grid=ri_grid,
                v_e_grid=v_e_true_grid,
                v_i_grid=v_i_true_grid,
                traj_batch=traj_batch,
                evaluator=evaluator
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