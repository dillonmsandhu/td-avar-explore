# PURE ORACLE: Exact Value Policy Gradient
# corrects the intrinsic reward to not scale with N
# UPDATE SIGMA FIRST
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_14_oracle"

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

    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // batch_size)

    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config) # solves env exactly.
    
    if config.get('SCHEDULE_BETA', False):
        beta_sch = helpers.make_triangle_schedule(config['NUM_UPDATES'] * 0.9, max_beta=config['BONUS_SCALE'])
    else:
        beta_sch = lambda x: config['BONUS_SCALE']
    
    # Safety Check: Oracle requires an evaluator
    assert evaluator is not None, "Oracle requires a valid exact evaluator for the environment!"

    def get_scale_free_bonus(S_inv, features):
        """bonus = x^T Σ^{-1} X, where Σ^{-1} is the empirical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

    def add_oracle_values_to_metric(
        config, metric, ri_grid, v_e_grid, v_i_grid, traj_batch, evaluator, v_net_ext_grid
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
            next_obs = jnp.asarray(traj_batch.next_obs)
            metric['visitation_count'] = next_obs[..., 0].sum(axis=(0, 1))
            
        elif "Chain" in config.get('ENV_NAME', '') or config.get('RND_NETWORK_TYPE') == 'identity':
            next_obs = jnp.asarray(traj_batch.next_obs)  # already true_next_obs for absorbing
            visitation = next_obs.sum(axis=(0, 1))
            metric['visitation_count'] = evaluator.get_value_grid(visitation)
        else:
            metric['visitation_count'] = jnp.zeros_like(ri_grid)

        # 2. Map Pre-Calculated Oracle Grids to Metrics
        metric.update({
            "ri_grid": ri_grid,
            "v_i": v_i_grid,
            "v_e": v_e_grid,
            "v_i_pred": v_i_grid,
            "v_e_pred": v_net_ext_grid,
            "effective_visits": (config["BONUS_SCALE"] / jnp.maximum(ri_grid, 1e-8))**2,
        })

        return metric

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
        
        # 2. Initialize Environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        # 3. Initialize Covariance State (We only need this to compute novelty)
        initial_sigma_state = {"S": jnp.eye(k, dtype = jnp.float64),}

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx) = runner_state

            # --- PHASE 1: COLLECT TRAJECTORIES ---
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, ve = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]
                _, next_ve = network.apply(train_state.params, true_next_obs)

                # Dummy values to satisfy the NamedTuple. They will be overwritten by the Oracle.
                dummy_val = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, ve, next_ve, dummy_val, dummy_val, reward, dummy_val,
                    log_prob, last_obs, true_next_obs, info,
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )
            # --- UPDATE COVARIANCE ---
            sigma_state = helpers.update_cov(
                traj_batch, sigma_state, get_phi
            )
            # --- COMPUTE NOVELTY REWARDS ---
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k))
            
            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            rho_scale = beta_sch(idx)
            
            # Unscaled rho for the batch
            next_phi = get_phi(traj_batch.next_obs)
            rho_batch = int_rew_from_features(next_phi)

            # Compute the intrinsic rewards for all states
            all_phi = get_phi(evaluator.obs_stack) 
            all_r_int = get_scale_free_bonus(Sigma_inv, all_phi) * rho_scale
            ri_grid = evaluator.get_value_grid(all_r_int)
            
            def lookup_int_rew(unused_obs_input):
                return all_r_int
            
            # --- PHASE 3: ORACLE VALUE INJECTION ---
            v_e_true_grid, v_i_true_grid, v_net_ext_grid = evaluator.compute_optimal_intrinsic_values(
                network, train_state.params, lookup_int_rew, all = True
            )

            # --- PHASE 3: ORACLE VALUE INJECTION ---
            def int_rew_from_state(s):
                """Helper for the evaluator to generate the exact intrinsic reward grid."""
                return int_rew_from_features(get_phi(s)) * rho_scale

            # Get Exact Values for the Entire MDP
            v_e_true_grid, v_i_true_grid, v_net_ext_grid = evaluator.compute_true_values(
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
                i_value=true_v_i,
                next_i_val=true_next_v_i,
                intrinsic_reward=rho_batch * rho_scale
            )

            # --- PHASE 4: GAE CALCULATION ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
                is_episodic=is_episodic, is_absorbing=is_absorbing,
                γi=config["GAMMA_i"], λi=0.0
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
                        
            done_mask = traj_batch.done > 0.5
            total_done = done_mask.sum()

            metric["true_next_v_i_on_done"] = (true_next_v_i * done_mask).sum() / jnp.maximum(total_done, 1)
            metric["rho_on_done"] = (traj_batch.intrinsic_reward * done_mask).sum() / jnp.maximum(total_done, 1)
            metric["true_v_i_on_done"] = (true_v_i * done_mask).sum() / jnp.maximum(total_done, 1)

            non_done_mask = 1.0 - done_mask
            total_non_done = non_done_mask.sum()
            residual = traj_batch.intrinsic_reward + config['GAMMA_i'] * true_next_v_i - true_v_i
            metric["bellman_residual_non_done"] = (residual * non_done_mask).sum() / jnp.maximum(total_non_done, 1)
            metric["bellman_residual_non_done_std"] = jnp.sqrt(
                ((residual - metric["bellman_residual_non_done"])**2 * non_done_mask).sum() / jnp.maximum(total_non_done, 1)
            )
            metric["rho_eval_mean"] = all_r_int.mean()        # evaluator's rho over all states
            metric["rho_batch_mean"] = rho_batch.mean()        # rollout rho
            metric["rho_eval_max"] = all_r_int.max()
            metric["rho_batch_max"] = rho_batch.max()
            
            def _pi_step(unused, x):
                # network.apply returns (pi_dist, v_ext, v_int) or (pi_dist, v_net)
                out = network.apply(train_state.params, x[None, ...])
                # Remove dummy batch dim from every element in the output tree
                return None, jax.tree.map(lambda arr: arr.squeeze(0), out)

            # 2. Scan over the observation stack
            _, out_all = jax.lax.scan(_pi_step, None, evaluator.obs_stack)

            # 3. Access pi_all (assuming pi_dist is the first element of the return tuple)
            pi_all = out_all[0] 
            num_actions = pi_all.probs.shape[-1]
            
            # Create a list of grids, one per action
            policy_grids = []
            for a in range(num_actions):
                prob_a = pi_all.probs[:, a]
                grid_a = evaluator.get_value_grid(prob_a)
                policy_grids.append(grid_a)
            
            # Stack into shape (num_actions, height, width)
            full_policy_grid = jnp.stack(policy_grids, axis=0)
            
            # Extract the probability of taking Action 1. 
            # pi_all.probs has shape (num_states, 2). We slice the second column.
            prob_a1 = pi_all.probs[:, 1]
            policy_grid_a1 = evaluator.get_value_grid(prob_a1)
            # ------------------------------

            metric = add_oracle_values_to_metric(
                config=config,
                metric=metric,
                ri_grid=ri_grid,
                v_e_grid=v_e_true_grid,
                v_i_grid=v_i_true_grid,
                traj_batch=traj_batch,
                evaluator=evaluator, 
                v_net_ext_grid = v_net_ext_grid
            )
            
            # Inject the policy grid into the final metrics dict
            metric["policy_grid_a1"] = policy_grid_a1
            metric["policy_dist_grid"] = full_policy_grid
            
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