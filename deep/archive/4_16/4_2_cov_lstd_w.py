# Full accumulation (no EMAs)
# scales the bonus by ||w||_infty
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_2_cov_lstd_w"
LEAK_FACTOR = (1-1e-3)

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
    # terminate bootstrap in LSTD?
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]  # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        """bonus = x^T Σ^{-1} X, where Σ^{-1} is the empriical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    def lstd_batch_update(lstd_state: Dict, transitions, features, next_features, traces):
            """
            LSTD update with:
            - intrinsic reward based on next-state uncertainty
            - Diagonal prior optimism applied purely at solve-time via Lambda_mat
            """
            batch_axes = tuple(range(transitions.done.ndim))
            batch_size = transitions.done.size # Extracted for easy division
            N = batch_size + lstd_state["N"]
            t = lstd_state["t"]
            
            rho = transitions.intrinsic_reward
            ext_rew = transitions.reward # <-- ADDED: Track extrinsic reward
            γ = config["GAMMA_i"]
            terminal = jnp.where(terminate_lstd_bootstrap, transitions.done, 0)[..., None]
            
            # ------------------------------------------------------------
            # 1. Empirical LSTD Updates
            # ------------------------------------------------------------
            # (Φ - γ * Φ')
            delta_Phi = features - γ * (1 - terminal) * next_features
            
            A_batch = jnp.einsum("nmi, nmj -> ij", traces, delta_Phi)
            b_batch = (traces * rho[..., None]).sum(axis=batch_axes)
            b_batch_ext = (traces * ext_rew[..., None]).sum(axis=batch_axes) # <-- ADDED
            
            # ------------------------------------------------------------
            # 2. Absorbing Terminal State: add fake data to the system
            # ------------------------------------------------------------
            absorb_mask = jnp.where(is_absorbing, transitions.done, 0)[..., None]
            
            absorbing_features = next_features * absorb_mask 
            absorbing_traces = absorbing_features 

            A_absorb = (1-γ) * jnp.einsum("nmi, nmj -> ij", absorbing_traces, absorbing_features)
            b_absorb = (absorbing_traces * rho[..., None]).sum(axis=batch_axes)
            # b_absorb_ext = (absorbing_traces * ext_rew[..., None]).sum(axis=batch_axes) # <-- ADDED
            b_absorb_ext = jnp.zeros_like(b_batch_ext)

            # ------------------------------------------------------------
            # 3. Update EMA, then add optimism and solve
            # ------------------------------------------------------------
            A_i = lstd_state["A"] * LEAK_FACTOR + A_batch + A_absorb
            b_i = lstd_state["b"] * LEAK_FACTOR + b_batch + b_absorb
            b_e = lstd_state["b_e"] * LEAK_FACTOR + b_batch_ext + b_absorb_ext # <-- ADDED

            # Optimistic Initialization (Diagonal Prior) 
            PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)

            # Calculate how much this batch contributes to feature "energy"
            batch_phi_precision = (features**2).sum(axis=(0, 1)) 
            absorbing_phi_precision = (absorbing_features**2).sum(axis=(0, 1)) 
            new_phi_diag_counts = lstd_state["phi_diag_counts"] + batch_phi_precision + absorbing_phi_precision
            
            # Bayesian ratio: If counts are low, lambda is high (prior dominates)
            lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts)
            lambda_k = jnp.where(lambda_k >= 0.1, lambda_k, 0.0)
            Lambda_mat = jnp.diag(lambda_k)

            k = A_batch.shape[0]
            reg = jnp.eye(k) * config["A_REGULARIZATION_PER_STEP"]

            A_view = A_i + Lambda_mat + reg
            prior_b = jnp.diag(Lambda_mat) * lstd_state["V_max"]
            b_view = b_i + prior_b

            # <-- MODIFIED: Stack target vectors to solve simultaneously for intrinsic and extrinsic weights -->
            B_mat = jnp.stack([b_view, b_e], axis=1) 
            W_mat = jnp.linalg.solve(A_view, B_mat)
            
            w_i = W_mat[:, 0]
            w_e = W_mat[:, 1]

            return {
                "A": A_i,
                "b": b_i,
                "b_e": b_e,  # <-- ADDED
                "w": w_i,
                "w_e": w_e,  # <-- ADDED
                "N": N,
                "t": t + 1,
                "V_max": lstd_state["V_max"],
                "Beta": lstd_state["Beta"],
                "phi_diag_counts": new_phi_diag_counts, 
            }
    
    V_max = (config['BONUS_SCALE']) / (1 - config["GAMMA_i"]) # maximum intrinsic values
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k)

    def train(rng):
        # <-- MODIFIED: Add tracking variables for extrinsic rewards -->
        initial_lstd_state = {
            "A": jnp.eye(k) * config["A_REGULARIZATION"],
            "b": jnp.zeros(k),
            "b_e": jnp.zeros(k), # <-- ADDED
            "w": jnp.zeros(k),
            "w_e": jnp.zeros(k), # <-- ADDED
            "N": 0,
            "t": 1,
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            'phi_diag_counts': jnp.zeros(k)
        }
        initial_sigma_state = {"S": jnp.eye(k),} 

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
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

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
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                next_i_val = jnp.zeros_like(reward)

                transition = Transition(
                    done,
                    action,
                    value,
                    next_val,
                    i_val,
                    next_i_val,
                    reward,
                    intrinsic_reward,
                    log_prob,
                    last_obs,
                    true_next_obs,
                    info,
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            Sigma_inv = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))
            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            rho = int_rew_from_features(batch_get_features(traj_batch.next_obs)) 
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            phi = batch_get_features(traj_batch.obs)

            # --- LSTD ---
            traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            next_phi = batch_get_features(traj_batch.next_obs)
            lstd_state = lstd_batch_update(lstd_state, traj_batch, phi, next_phi, traces)



            # --- MODIFIED: Dynamic Beta Calculation using ||w_e|| ---

            # Max norm:
            w_e_norm = jnp.max(jnp.abs(lstd_state["w_e"]))
            scaling_factor = jnp.minimum(lstd_state['V_max'], w_e_norm)
            dynamic_beta = config["BONUS_SCALE"] + config["BONUS_SCALE"] * scaling_factor
            # two scales:
            # dynamic_beta = .01 + config["BONUS_SCALE"] * scaling_factor
            progress = idx / config["NUM_UPDATES"]
            # 2. Linear decay multiplier (1.0 -> 0.0)
            # decay_mult = jnp.maximum(0.0, 1.0 - progress)
            decay_mult=1
            rho_scale = dynamic_beta * decay_mult
            
            # Record it back in the state for logging
            lstd_state["Beta"] = dynamic_beta 

            v_i = phi @ lstd_state["w"] * rho_scale
            next_v_i = next_phi @ lstd_state["w"] * rho_scale

            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho * rho_scale, next_i_val = next_v_i)

            # --- 4. ADVANTAGE CALCULATION (Scaled) ---
            gaes, targets = helpers.calculate_gae(
                traj_batch,
                config["GAMMA"],
                config["GAE_LAMBDA"],
                is_episodic=is_episodic,
                is_absorbing=is_absorbing,
                γi=config["GAMMA_i"],     
                λi=config["GAE_LAMBDA_i"]
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
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)

            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
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
                    "lambda_k": lstd_state['phi_diag_counts'],
                    "beta": lstd_state["Beta"],
                    "rho_scale": rho_scale,
                    "w_e_norm": w_e_norm, # <-- ADDED for monitoring
                }
            )

            if evaluator is None: 
                metric.update(
                    {
                        "vi_pred": traj_batch.i_value.mean(),
                        "v_e_pred": traj_batch.value.mean(),
                    }
                )
            else:
                def int_rew_from_state(s,): 
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