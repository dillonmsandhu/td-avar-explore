# a different approach to optimism.
# updates LSTD to have a prior towards V-max on the identity matrix
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue

SAVE_DIR = "3_18_cov_lstd"


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
    evaluator = None
    if calc_true_values:
        if config["ENV_NAME"] == "DeepSea-bsuite":
            evaluator = DeepSeaExactValue(
                size=config["DEEPSEA_SIZE"],
                unscaled_move_cost=0.01,
                gamma=config["GAMMA"],
                episodic=config["EPISODIC"],
            )
        if config["ENV_NAME"] == "Chain":
            evaluator = LongChainExactValue(config.get("CHAIN_LENGTH", 100), config["GAMMA"], config["EPISODIC"])

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

    def get_scale_free_bonus(S_inv, features):
        """bonus = x^T Sigma^{-1} X, where Sigma^{-1} is the empriical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    def lstd_batch_update(lstd_state: Dict, transitions, features, next_features, traces, Lambda_mat, batch_phi_precision):
        """
        LSTD update with:
        - intrinsic reward based on next-state uncertainty
        - Diagonal prior optimism applied purely at solve-time via Lambda_mat
        """
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state["N"]
        t = lstd_state["t"]
        rho = transitions.intrinsic_reward

        # 1. Empirical LSTD Updates
        A_update = jax.vmap(jax.vmap(cross_cov))(traces, features, next_features, transitions.done)
        A_batch = A_update.mean(axis=batch_axes)

        b_i_sample = traces * rho[..., None]
        b_batch = b_i_sample.mean(axis=batch_axes)

        # 2. Update EMA
        A_i = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], A_batch)
        b_i = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_batch)
        
        # Update the explicit feature counts (Sum of squares)
        phi_diag_counts = lstd_state["phi_diag_counts"] + batch_phi_precision

        # 3. Apply Diagonal Prior and Solve
        k = A_batch.shape[0]
        reg = jnp.eye(k) * config["A_REGULARIZATION_PER_STEP"]

        A_view = A_i + Lambda_mat + reg
        prior_b = jnp.diag(Lambda_mat) * lstd_state["V_max"]
        b_view = b_i + prior_b

        w_i = jnp.linalg.solve(A_view, b_view)

        return {
            "A": A_i,
            "b": b_i,
            "w": w_i,
            "N": N,
            "t": t + 1,
            "V_max": lstd_state["V_max"],
            "Beta": lstd_state["Beta"],
            "phi_diag_counts": phi_diag_counts, # Persistent tracker
        }

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

        V_max = (jnp.sqrt(1.0 / config["GRAM_REG"])) / (1 - config["GAMMA_i"]) # maximum intrinsic values
        if config["NORMALIZE_FEATURES"]:
            V_max /= jnp.sqrt(k)

        initial_lstd_state = {
            "A": jnp.eye(k) * config["A_REGULARIZATION"],
            "b": jnp.zeros(k),
            "w": jnp.zeros(k),
            "N": 0,
            "t": 1,
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            'phi_diag_counts': jnp.zeros(k)
        }
        initial_sigma_state = {
            "S": jnp.zeros((k,k)),
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
                    obsv,
                    info,
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition

            # end _env_step

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # --- Intrinsic Rewards ---
            # invert the covariance matrix
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + config['GRAM_REG'] * jnp.eye(k) ,jnp.eye(k))

            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            # traj_batch, sigma_state, rho = helpers.update_cov_and_get_rho(traj_batch, sigma_state, batch_get_features, int_rew_from_features, alpha_fn)
            rho = int_rew_from_features(batch_get_features(traj_batch.next_obs))
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            phi = batch_get_features(traj_batch.obs)

            # --- 4. Optimistic Initialization (Diagonal Prior) ---
            PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)

            # Calculate how much this batch contributes to feature "energy"
            # phi shape is (T, B, k), so we sum over T and B
            batch_phi_precision = (phi**2).sum(axis=(0, 1)) 
            current_total_counts = lstd_state["phi_diag_counts"] + batch_phi_precision

            # Bayesian ratio: If counts are low, lambda is high (prior dominates)
            lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + current_total_counts)
            lambda_k = jnp.where(lambda_k >= 0.1, lambda_k, 0.0)
            Lambda_mat = jnp.diag(lambda_k)

            # --- LSTD ---
            traces = trace_fn(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"])
            next_phi = batch_get_features(traj_batch.next_obs)
            lstd_state = lstd_batch_update(lstd_state, traj_batch, phi, next_phi, traces,  Lambda_mat, batch_phi_precision)

            # --- GAE ---
            _, last_val = network.apply(train_state.params, last_obs)
            v_i = phi @ lstd_state["w"]
            last_i_val = batch_get_features(last_obs) @ lstd_state["w"]

            # Adaptive beta (scaling of ri and vi)
            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state["N"]))

            lstd_state["Beta"] = helpers.schedule_extrinsic_to_intrinsic_ratio(
                sigma_state["N"] / config["TOTAL_TIMESTEPS"], config["BONUS_SCALE"]
            )
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
            metric = {k: v.mean() for k, v in traj_batch.info.items()}

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
                    "lambda_k": lambda_k,
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
