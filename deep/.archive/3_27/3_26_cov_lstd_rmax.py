# designed to correctly handle terminal state
# uses r-max optimism
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "3_26_lstd_rmax"

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


def append_frontier_transition(traj_batch: Transition, last_obs: jnp.ndarray) -> Transition:
    """
    Appends a dummy terminal transition from the last observation to the batch.
    This acts as the frontier for LSTD-RMAX optimistic initialization.

    Args:
        traj_batch: The original Transition NamedTuple of shape (T, B, ...)
        last_obs: The final observation tensor of shape (B, *obs_shape)

    Returns:
        padded_traj_batch: A new Transition NamedTuple of shape (T+1, B, ...)
    """
    # 1. Create a zeroed-out dummy transition of shape (1, B, ...)
    dummy_transition = jax.tree_util.tree_map(lambda x: jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), traj_batch)

    # 2. Override specific fields to represent a terminal step from last_obs
    num_envs = last_obs.shape[0]
    dummy_transition = dummy_transition._replace(
        obs=last_obs[None, ...],  # The frontier state is the current obs
        done=jnp.ones((1, num_envs), dtype=traj_batch.done.dtype),  # Force terminal
    )

    # 3. Concatenate along the time axis (axis=0) to create the padded batch
    padded_traj_batch = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0), traj_batch, dummy_transition
    )

    return padded_traj_batch


def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]  # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    # --- Flag to enable heavy exact value calculation ---
    calc_true_values = config.get("CALC_TRUE_VALUES", False)
    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    alpha_fn = lambda t: jnp.maximum(config.get("MIN_COV_LR", 1 / 10), 1 / t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config["ALPHA_SCHEDULE"], config["MIN_LSTD_LR"])
    alpha_fn_lstd_b = helpers.get_alpha_schedule(config["ALPHA_SCHEDULE"], config["MIN_LSTD_LR_RI"])
    evaluator = helpers.initialize_evaluator(config)

    if config["EPISODIC"]:
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
        trace_fn = helpers._get_all_traces  # continuing due to setting phi' = 0 when done = True.
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov(z, phi, phi_prime, done, config["GAMMA_i"])
    else:
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic
        trace_fn = helpers._get_all_traces_continuing
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov_continuing(
            z, phi, phi_prime, done, config["GAMMA_i"]
        )

    def get_scale_free_bonus(Sigma_inv, features):
        """bonus = x^T Sigma^{-1} X, where Sigma^{-1} is the empriical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, Sigma_inv, features)
        return jnp.sqrt(bonus_sq)

    def lstd_batch_update(lstd_state: Dict, transitions, features, next_features, traces, lambda_s):
        """
        LSTD update with:
        - intrinsic reward based on next-state uncertainty
        - soft LSPI-RMAX prior on intrinsic value using state-dependent lambda
        """

        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state["N"]
        t = lstd_state["t"]
        rho = transitions.intrinsic_reward

        # ------------------------------------------------------------
        # 2. Standard LSTD A update (policy evaluation)
        # ------------------------------------------------------------
        A_update = jax.vmap(jax.vmap(cross_cov))(traces, features, next_features, transitions.done)
        lambda_s = jnp.clip(lambda_s, 0.0, 1.0)  # lambda_s denotes
        A_i_update = A_update * (1 - lambda_s)[..., None, None]  # downweight each contribution as (1-lambda)
        # A_i_update = A_update

        A_batch = A_update.mean(axis=batch_axes)
        A_i_batch = A_i_update.mean(axis=batch_axes)
        weighted_gram_batch = jnp.einsum("bt, bti, btj->ij", lambda_s, features, features) / transitions.done.size
        A_batch_rmax = A_i_batch + weighted_gram_batch
        A_i_view = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], A_batch_rmax)

        # for storage (exclude novelty-based optimsim):
        A_i = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], A_batch)

        ## feature-reward vector
        b_i_sample = traces * rho[..., None]
        b_i_sample_opt = (1 - lambda_s)[..., None] * b_i_sample + lambda_s[..., None] * lstd_state["V_max"] * features
        b_i_opt = b_i_sample_opt.mean(axis=batch_axes)
        # ema of stored (non-optimistic) b and optimistic b.
        b_i_view = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_i_opt)

        # for storage:
        b_i = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_i_sample.mean(axis=batch_axes))

        # ------------------------------------------------------------
        # 6. Solve linear systems
        # ------------------------------------------------------------
        reg = jnp.eye(A_batch.shape[0]) * config["A_REGULARIZATION_PER_STEP"]
        w_i = jnp.linalg.solve(A_i_view + reg, b_i_view)

        return {
            "A": A_i,
            "b": b_i,
            "w": w_i,
            "N": N,
            "t": t + 1,
            "V_max": lstd_state["V_max"],
            "Beta": lstd_state["Beta"],
        }
        # end lstd_batch_update

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng,
            obs_shape,
            config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"],
            config["BIAS"],
            k,
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng,
            obs_shape,
            config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"],
            config["BIAS"],
            k,
        )

        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )

        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        initial_lstd_state = {
            "A": jnp.eye(k) * config["A_REGULARIZATION"],
            "b": jnp.zeros(k),
            "w": jnp.zeros(k),
            "N": 0,
            "t": 1,
            "Beta": config["BONUS_SCALE"],
            "V_max": 1 / (1 - config["GAMMA_i"]),
        }
        initial_sigma_state = {
            "S": jnp.eye(k) * config["GRAM_REG"],
            "N": 1,
            "t": 1,
        }

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            (
                train_state,
                lstd_state,
                sigma_state,
                rnd_state,
                env_state,
                last_obs,
                rng,
                idx,
            ) = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
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

            initial_obs = last_obs  # shape: (NUM_ENVS, *obs_shape)
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )
            # Done gathering environment steps
            # Intrinsic Rewards and optimism

            k = sigma_state["S"].shape[0]
            S_inv_local = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))

            int_rew_from_features = lambda phi: get_scale_free_bonus(S_inv_local, phi)
            rho = int_rew_from_features(batch_get_features(traj_batch.next_obs))

            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            traj_batch_padded = append_frontier_transition(traj_batch, last_obs)

            # --- 4. Optimistic Initialization (De-biased Frontier Prior) ---
            phi = batch_get_features(traj_batch_padded.obs)

            # 1. Variance in the direction of each state's feature vector
            # int_rew_from_features is sqrt(phi^T S_inv phi), so square it for variance
            variance_s = int_rew_from_features(phi) ** 2

            # 2. Total precision (data counts + fake regularization counts)
            total_precision_s = sigma_state["N"] / (variance_s + 1e-8)

            # 3. Calculate hallucinated counts from the regularizer for this specific direction
            # (If features are L2 normalized, phi_sq_norm is just 1.0, but this makes it universally safe)
            phi_sq_norm = jnp.sum(phi**2, axis=-1)
            counts_from_reg = sigma_state["N"] * config["GRAM_REG"] * phi_sq_norm

            # 4. Subtract to reveal the true empirical counts for this state
            pseudo_counts_s = jnp.maximum(0.0, total_precision_s - counts_from_reg)

            # 5. Apply the Bayesian ratio
            PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 50.0)
            lambda_s = PRIOR_SAMPLES / (PRIOR_SAMPLES + pseudo_counts_s)

            # Drops to zero when this specific state/direction is explored
            lambda_s = jnp.where(lambda_s >= 0.1, lambda_s, 0.0)

            # Update LSTD
            traces = trace_fn(traj_batch_padded, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"])
            next_phi = batch_get_features(traj_batch_padded.next_obs)
            lstd_state = lstd_batch_update(lstd_state, traj_batch_padded, phi, next_phi, traces, lambda_s)

            # Compute Values for GAE.
            _, last_val = network.apply(train_state.params, last_obs)
            # last_phi = batch_get_features(last_obs)
            last_phi = phi[-1]
            v_i = phi[:-1] @ lstd_state["w"]
            last_i_val = last_phi @ lstd_state["w"]
            lstd_state["V_max"] = jnp.maximum(jnp.max(v_i), jnp.max(last_i_val))  # unscaled maximum value.

            # Adaptive beta (scaling of ri and vi)
            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state["N"]))

            lstd_state["Beta"] = helpers.schedule_extrinsic_to_intrinsic_ratio(
                sigma_state["N"] / config["TOTAL_TIMESTEPS"], config["BONUS_SCALE"]
            )
            # lstd_state['Beta'] = helpers.update_beta(lstd_state['Beta'], v_i / sqrt_n, traj_batch.value, progress = sigma_state['N'] / config['TOTAL_TIMESTEPS'], update=config['ADAPTIVE_BETA'])
            # Final scale of r_i is unscaled rho times 1/sqrt(N) times beta
            rho_scale = lstd_state["Beta"] / sqrt_n

            # Scale vi and ri in traj_batch for GAE.
            v_i *= rho_scale
            last_i_val *= rho_scale
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho * rho_scale)

            # --- 4. ADVANTAGE CALCULATION (Scaled) ---
            gaes, targets = gae_fn(
                traj_batch,
                last_val,
                last_i_val,
                config["GAMMA"],
                config["GAE_LAMBDA"],
                config["GAE_LAMBDA_i"],
                config["GAMMA_i"],
            )
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params,
                        network,
                        traj_batch,
                        advantages,
                        targets,
                        config,
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

            initial_update_state = (
                train_state,
                traj_batch,
                advantages,
                extrinsic_target,
                rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            # UPDATE Covariance
            _, sigma_state, _ = helpers.update_cov_and_get_rho(
                traj_batch,
                sigma_state,
                batch_get_features,
                int_rew_from_features,
                alpha_fn,
            )

            # --------- Metrics ---------

            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }

            # Shared Metrics
            metric.update(
                {
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
                    "lambda_s": jnp.mean(lambda_s),
                    "lambda_s_min": jnp.min(lambda_s),
                    "lambda_s_max": jnp.max(lambda_s),
                    "beta": lstd_state["Beta"],
                    "rho_scale": rho_scale,
                }
            )

            if evaluator is None:  # No way to compute true values, just record the batch average prediction.
                metric.update(
                    {
                        "vi_pred": traj_batch.i_value.mean(),
                        "v_e_pred": traj_batch.value.mean(),
                    }
                )
            else:

                def int_rew_from_state(
                    s,
                ):  # for computing the intrinsic reward given an arbitrary state
                    phi = batch_get_features(s)
                    rho = int_rew_from_features(phi) * rho_scale
                    return rho

                def get_vi(obs):
                    return batch_get_features(obs) @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(
                    config,
                    metric,
                    int_rew_from_state,
                    evaluator,
                    lstd_state["Beta"],
                    network,
                    train_state,
                    traj_batch,
                    get_vi,
                )

            runner_state = (
                train_state,
                lstd_state,
                sigma_state,
                rnd_state,
                env_state,
                last_obs,
                rng,
                idx + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            initial_lstd_state,
            initial_sigma_state,
            rnd_state,
            env_state,
            obsv,
            _rng,
            0,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from core.utils import run_experiment_main

    run_experiment_main(make_train, SAVE_DIR)
