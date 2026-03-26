# LSTD-Q evaluating the current policy
# Independent State Action Featuresn (sparse block LSTD system)
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue

SAVE_DIR = "3_23_cov_lstd_q_on_policy"


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
    n_actions = env.action_space(env_params).n

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
    else:
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic
        trace_fn = helpers._get_all_traces_continuing

    def get_scale_free_bonus(S_inv, features):
        """bonus = x^T Sigma^{-1} X, where Sigma^{-1} is the empriical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    def expand_to_sa_features(phi_s, n_actions, taken_actions):
        "Takes phi_s (batched length k vectors), and actions taken, and returns phi_action_taken, with a block-structure [ ...0..., phi(s), ... 0 ... ]"
        # 1. Construct the block-sparse Phi(s, a) for the taken actions
        one_hots = jax.nn.one_hot(taken_actions, n_actions)  # Shape: (T, B, n_actions)
        # Broadcast multiply: (T, B, 1, k) * (T, B, n_actions, 1) -> (T, B, n_actions, k)
        phi_sa_unflattened = phi_s[..., None, :] * one_hots[..., :, None]
        # Flatten the last two dims to get the block-diagonal structure: (T, B, n_actions * k)
        phi_taken_action = phi_sa_unflattened.reshape(*phi_s.shape[:-1], n_actions * k)
        return phi_taken_action

    def expected_next_sa_features(next_phi, Pi):
        "Assumes Pi is size (..., n_actions), and next_phi is size (..., k). Contracts over the policy"
        expected_next_sa = next_phi[..., None, :] * Pi[..., :, None]
        return expected_next_sa.reshape(*next_phi.shape[:-1], -1)

    def LSTDQ(lstd_state: Dict, transitions, features, next_features, traces, target_pi):
        """
        LSTDQ for ON-POLICY target with:
        - intrinsic reward based on next-state uncertainty
        - Diagonal prior optimism applied purely at solve-time via Lambda_mat
        """
        batch_axes = tuple(range(transitions.done.ndim))
        batch_size = transitions.done.size
        N = batch_size + lstd_state["N"]
        t = lstd_state["t"]
        rho = transitions.intrinsic_reward
        Z = traces  # Shape: (T, B, k * n_actions)
        Φ = features  # Shape: (T, B, k * n_actions)
        γ = config["GAMMA"]
        # terminal masking whenever episodic is true
        is_episodic = config.get("EPISODIC", True)
        terminal = jnp.where(is_episodic, transitions.done, 0)[..., None]

        # Policy: On-policy target. Shape: (T, B, n_actions)
        Pi = target_pi
        # ------------------------------------------------------------
        # 1. Purely Empirical LSTD Updates
        # ------------------------------------------------------------
        S = jnp.einsum("nmi, nmj -> ij", Z, Φ)
        PΠφ = expected_next_sa_features(next_features, Pi)  # (T, B, n_actions * k)
        γPΠφ = γ * (1 - terminal) * PΠφ
        γPΠΦ = jnp.einsum("nmi, nmj -> ij", Z, γPΠφ)
        A_batch = S - γPΠΦ  # A = Ζ^Τ (Φ-γPΠΦ)
        A_batch /= batch_size  # mean over batch

        b_i_sample = traces * rho[..., None]
        b_batch = b_i_sample.mean(axis=batch_axes)
        # ------------------------------------------------------------
        # 2. Update EMA (Strictly empirical, NO optimism baked in)
        # ------------------------------------------------------------
        A_i = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], A_batch)
        b_i = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_batch)
        # ------------------------------------------------------------
        # 3. Apply Diagonal Prior and Solve (Optimism View)
        # ------------------------------------------------------------
        batch_sa_precision = (Φ**2).sum(axis=batch_axes)
        sa_diag_counts = lstd_state["sa_diag_counts"] + batch_sa_precision

        # Bayesian Prior per State-Action
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_kA = PRIOR_SAMPLES / (PRIOR_SAMPLES + sa_diag_counts)
        lambda_kA = jnp.where(lambda_kA >= 0.1, lambda_kA, 0.0)
        Lambda_mat = jnp.diag(lambda_kA)  # Shape: (k*|A|, k*|A|)

        dim_kA = A_batch.shape[0]  # This is now k * n_actions
        reg = jnp.eye(dim_kA) * config["A_REGULARIZATION_PER_STEP"]

        # Inject optimism directly into the current view of A
        A_view = A_i + Lambda_mat + reg

        prior_b = jnp.diag(Lambda_mat) * lstd_state["V_max"]
        b_view = b_i + prior_b

        # Final Solve
        w_i = jnp.linalg.solve(A_view, b_view)

        return {
            "A": A_i,
            "b": b_i,
            "w": w_i,
            "N": N,
            "t": t + 1,
            "V_max": lstd_state["V_max"],
            "Beta": lstd_state["Beta"],
            "sa_diag_counts": sa_diag_counts,
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
        get_phi = jax.vmap(get_features_fn)
        # get_phi = get_features_fn

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        V_max = (jnp.sqrt(1.0 / config["GRAM_REG"])) / (1 - config["GAMMA_i"]) # maximum intrinsic values
        if config["NORMALIZE_FEATURES"]:
            V_max /= jnp.sqrt(k)

        # A = (Z^ΤΦ)^{-1} - γZ^ΤPπΦ
        dim_kA = k * n_actions
        initial_lstd_state = {
            "A": jnp.eye(dim_kA) * config["A_REGULARIZATION"],
            "b": jnp.zeros(dim_kA),
            "w": jnp.zeros(dim_kA),
            "N": 0,
            "t": 1,
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            "sa_diag_counts": jnp.zeros(dim_kA),
        }
        initial_sigma_state = {
            "S": jnp.zeros((k, k)),
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
            # --- Intrinsic Reward (due to Precision) ---
            # Precision matrix
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + config['GRAM_REG'] * jnp.eye(k) ,jnp.eye(k),)
            
            ρ_from_phi = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            phi_next_s = get_phi(traj_batch.next_obs)
            rho = ρ_from_phi(phi_next_s)  # scale-free

            # scale of the intrinsic reward:
            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state["N"]))
            ri_scale = lstd_state["Beta"] / sqrt_n

            # --- LSTD ---
            phi_s = get_phi(traj_batch.obs)  # Pure state features: (T, B, k)
            # Forward pass the next states through the actor to get pi(a | s')
            pi_next, _ = network.apply(train_state.params, traj_batch.next_obs)
            target_pi_next = pi_next.probs

            phi_sa = expand_to_sa_features(phi_s, n_actions, traj_batch.action)
            # 2. Compute traces using the block-sparse features
            traces = trace_fn(traj_batch, phi_sa, config["GAMMA_i"], config["GAE_LAMBDA_i"])
            # 3. Call LSTDQ (passing the expanded Phi_taken, but the raw next_phi_s)
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            lstd_state = LSTDQ(lstd_state, traj_batch, phi_sa, phi_next_s, traces, target_pi_next)

            # Intrinsic values (scaled):
            def get_vi(obs):
                "Gets intrinsic value from LSTDQ (uniform average policy)"
                phi = get_phi(obs)  # Shape: (..., k)
                pi_obs, _ = network.apply(train_state.params, obs)
                Pi = pi_obs.probs

                # Expand to expected state-action features
                phi_policy = expected_next_sa_features(phi, Pi)
                return phi_policy @ lstd_state["w"] * ri_scale

            # intrinsic value computation on the batch (random policy):
            v_i = get_vi(traj_batch.obs)
            last_i_val = get_vi(last_obs)
            # Scale vi and ri in traj_batch for GAE.
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho * ri_scale)

            # --- GAE ---
            _, last_val = network.apply(train_state.params, last_obs)

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
                get_phi,
                ρ_from_phi,
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
                    "feat_norm": jnp.linalg.norm(phi_next_s, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                    "mean_rew": traj_batch.reward.mean(),
                    "lambda_k": lstd_state["sa_diag_counts"],
                    "beta": lstd_state["Beta"],
                    "rho_scale": ri_scale,
                }
            )

            if evaluator is None:  # No way to compute true values, just record the batch average prediction.
                metric.update(
                    {
                        "vi_pred": traj_batch.i_value.mean(),
                        "v_e_pred": traj_batch.value.mean(),
                    }
                )
            else:  # Compute the true intrinsic value using the evaluator

                def int_rew_from_state(
                    s,
                ):
                    phi = get_phi(s)
                    rho = ρ_from_phi(phi) * ri_scale
                    return rho

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
