# LSPI for intrinsic value
# Independent State Action Features (sparse block LSTD system)
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "3_26_cov_lspi" # Updated date

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray # Added
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray # Added
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    # Extract formulation flags directly from config
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]  
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    calc_true_values = config.get("CALC_TRUE_VALUES", False)
    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    alpha_fn = lambda t: jnp.maximum(config.get("MIN_COV_LR", 1 / 10), 1 / t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config["ALPHA_SCHEDULE"], config["MIN_LSTD_LR"])
    alpha_fn_lstd_b = helpers.get_alpha_schedule(config["ALPHA_SCHEDULE"], config["MIN_LSTD_LR_RI"])
    
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    def expand_to_sa_features(phi_s, n_actions, taken_actions):
        one_hots = jax.nn.one_hot(taken_actions, n_actions)  
        phi_sa_unflattened = phi_s[..., None, :] * one_hots[..., :, None]
        phi_taken_action = phi_sa_unflattened.reshape(*phi_s.shape[:-1], n_actions * k)
        return phi_taken_action

    def expected_next_sa_features(next_phi, Pi):
        expected_next_sa = next_phi[..., None, :] * Pi[..., :, None]
        return expected_next_sa.reshape(*next_phi.shape[:-1], -1)

    def LSPI(lstd_state: Dict, transitions, features, next_features, traces, num_iters=3):
        """
        Least-Squares Policy Iteration with Absorbing State Transitions.
        """
        batch_axes = tuple(range(transitions.done.ndim))
        batch_size = transitions.done.size
        N = batch_size + lstd_state["N"]
        t = lstd_state["t"]
        rho = transitions.intrinsic_reward
        Z = traces
        Φ = features
        γ = config["GAMMA_i"] # Fixed to intrinsic discount
        
        terminal = jnp.where(terminate_bootstrap, transitions.done, 0)[..., None]
        absorb_mask = jnp.where(is_absorbing, transitions.done, 0)[..., None]
        
        num_absorbs = absorb_mask.sum()
        total_samples = batch_size + num_absorbs
        # absorbing_traces = Z * absorb_mask
        absorbing_traces =  Φ * absorb_mask # for now: use abosrbing features since we don't have a trace for the fake data.

        # ------------------------------------------------------------
        # 1. Constants for the Batch (S and b) computed ONCE
        # ------------------------------------------------------------
        S = jnp.einsum("nmi, nmj -> ij", Z, Φ)
        
        # IV for Right Side (b) is static. 
        b_std = (Z * rho[..., None]).sum(axis=batch_axes)
        b_absorb = (absorbing_traces * rho[..., None]).sum(axis=batch_axes)
        b_batch = (b_std + b_absorb) / total_samples

        # ------------------------------------------------------------
        # 2. Stable Optimism Prior
        # ------------------------------------------------------------
        batch_sa_precision = (Φ**2).sum(axis=batch_axes)
        
        # We compute ghost precision using Uniform Policy to keep prior stable during LSPI loop
        Pi_uniform = jnp.ones((*transitions.done.shape, n_actions)) * (1.0 / n_actions)
        uniform_absorbing_features = expected_next_sa_features(next_features, Pi_uniform) * absorb_mask
        absorbing_sa_precision = (uniform_absorbing_features**2).sum(axis=batch_axes)
        
        new_sa_diag_counts = lstd_state["sa_diag_counts"] + batch_sa_precision + absorbing_sa_precision

        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_kA = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_sa_diag_counts)
        lambda_kA = jnp.where(lambda_kA >= 0.1, lambda_kA, 0.0)
        Lambda_mat = jnp.diag(lambda_kA)

        dim_kA = k * n_actions
        reg = jnp.eye(dim_kA) * config["A_REGULARIZATION_PER_STEP"]
        prior_b = jnp.diag(Lambda_mat) * lstd_state["V_max"]

        # ------------------------------------------------------------
        # 3. LSPI Inner Loop
        # ------------------------------------------------------------
        def lspi_step(w_current, _):
            # 1. Extract greedy policy
            w_reshaped = w_current.reshape(n_actions, k)
            Q_next = jnp.einsum("...k, ak -> ...a", next_features, w_reshaped)
            greedy_actions = jnp.argmax(Q_next, axis=-1)
            Pi_greedy = jax.nn.one_hot(greedy_actions, n_actions)

            # 2. Policy Evaluation on just this batch!
            PΠφ = expected_next_sa_features(next_features, Pi_greedy)
            γPΠφ = γ * (1 - terminal) * PΠφ
            γPΠΦ = jnp.einsum("nmi, nmj -> ij", Z, γPΠφ)

            A_std = S - γPΠΦ 
            absorbing_features = PΠφ * absorb_mask
            A_absorb = (1 - γ) * jnp.einsum("nmi, nmj -> ij", absorbing_traces, absorbing_features)

            A_batch = (A_std + A_absorb) / total_samples

            # FIX: Do not blend with historical EMA during the inner loop!
            # Solve purely on the current batch to find the true greedy policy
            A_view = A_batch + Lambda_mat + reg
            b_view = b_batch + prior_b

            w_new = jnp.linalg.solve(A_view, b_view)
            return w_new, A_batch

        w_init = lstd_state["w"]
        w_final, A_batch_history = jax.lax.scan(lspi_step, w_init, None, length=num_iters)
        final_A_batch = A_batch_history[-1]

        # ------------------------------------------------------------
        # 4. Final State Updates
        # ------------------------------------------------------------
        A_i = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], final_A_batch)
        b_i = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_batch)

        return {
            "A": A_i,
            "b": b_i,
            "w": w_final,
            "N": N,
            "t": t + 1,
            "V_max": lstd_state["V_max"],
            "Beta": lstd_state["Beta"],
            "sa_diag_counts": new_sa_diag_counts,
        }

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"], config["BIAS"], k,
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"], config["BIAS"], k,
        )

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )

        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        get_phi = jax.vmap(get_features_fn)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        V_max = (jnp.sqrt(1.0 / config["GRAM_REG"])) / (1 - config["GAMMA_i"]) 
        if config["NORMALIZE_FEATURES"]:
            V_max /= jnp.sqrt(k)

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

            (train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx) = runner_state

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
                
                # Pre-compute true next values
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)
                
                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                next_i_val = jnp.zeros_like(reward)
                
                transition = Transition(
                    done, action, value, next_val, i_val, next_i_val, 
                    reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info,
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )
            
            # --- Intrinsic Reward ---
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + config['GRAM_REG'] * jnp.eye(k) ,jnp.eye(k))

            ρ_from_phi = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            phi_next_s = get_phi(traj_batch.next_obs)
            rho = ρ_from_phi(phi_next_s)  

            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state["N"]))
            ri_scale = lstd_state["Beta"] / sqrt_n

            # --- LSTD ---
            phi_s = get_phi(traj_batch.obs)  
            phi_sa = expand_to_sa_features(phi_s, n_actions, traj_batch.action)
            
            # Unified Trace function from helpers
            traces = helpers.calculate_traces(
                traj_batch, phi_sa, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing
            )
            
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            lstd_state = LSPI(lstd_state, traj_batch, phi_sa, phi_next_s, traces, num_iters=config.get("LSPI_NUM_ITERS", 3))

            # Intrinsic values (scaled):
            def get_vi(obs):
                "Gets greedy intrinsic value V(s) from LSPI"
                phi = get_phi(obs)  
                w_reshaped = lstd_state["w"].reshape(n_actions, k)
                Q_vals = jnp.einsum("...k, ak -> ...a", phi, w_reshaped)  
                return jnp.max(Q_vals, axis=-1) * ri_scale  

            v_i = get_vi(traj_batch.obs)
            next_v_i = get_vi(traj_batch.next_obs)
            
            traj_batch = traj_batch._replace(i_value=v_i, next_i_val=next_v_i, intrinsic_reward=rho * ri_scale)

            # --- GAE ---
            # Using the unified GAE function directly from helpers
            gaes, targets = helpers.calculate_gae(
                traj_batch,
                config["GAMMA"],
                config["GAE_LAMBDA"],
                is_episodic=is_episodic,
                is_absorbing=is_absorbing,
                γi=config["GAMMA_i"],
                λi=config["GAE_LAMBDA_i"],
            )
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
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

            # UPDATE Covariance
            _, sigma_state, _ = helpers.update_cov_and_get_rho(
                traj_batch, sigma_state, get_phi, ρ_from_phi, alpha_fn,
            )

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
                    "feat_norm": jnp.linalg.norm(phi_next_s, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                    "mean_rew": traj_batch.reward.mean(),
                    "beta": lstd_state["Beta"],
                    "rho_scale": ri_scale,
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
                def int_rew_from_state(s):
                    phi = get_phi(s)
                    rho = ρ_from_phi(phi) * ri_scale
                    return rho

                metric = helpers.add_values_to_metric(
                    config, metric, int_rew_from_state, evaluator,
                    lstd_state["Beta"], network, train_state, traj_batch, get_vi,
                )

            runner_state = (train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
