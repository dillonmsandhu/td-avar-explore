# Full accumulation
# corrected intrinsic reward
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_2_cov_lstd_q" # Updated to reflect new run

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray  # Added
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray  # Added
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

    def LSTDQ(lstd_state: Dict, transitions, features, next_features, traces):
        """
        LSTDQ for random target policy with exact Absorbing State tracking.
        """
        batch_axes = tuple(range(transitions.done.ndim))
        batch_size = transitions.done.size
        N = batch_size + lstd_state["N"]
        t = lstd_state["t"]
        rho = transitions.intrinsic_reward
        Z = traces 
        Φ = features 
        
        γ = config["GAMMA_i"] 

        # Policy: Uniform random.
        Pi = jnp.ones((*transitions.done.shape, n_actions)) * (1.0 / n_actions)
        
        # ------------------------------------------------------------
        # 1. Standard LSTD Updates
        # ------------------------------------------------------------
        terminal = jnp.where(terminate_bootstrap, transitions.done, 0)[..., None]
        PΠφ = expected_next_sa_features(next_features, Pi)  
        
        delta_Phi = Φ - γ * (1 - terminal) * PΠφ
        A_batch = jnp.einsum("nmi, nmj -> ij", Z, delta_Phi)
        b_batch = (Z * rho[..., None]).sum(axis=batch_axes)
        
        # ------------------------------------------------------------
        # 2. Absorbing Terminal State (LSTD-Q)
        # ------------------------------------------------------------
        absorb_mask = jnp.where(is_absorbing, transitions.done, 0)[..., None]
        
        # The expected feature of the absorbing state under policy Pi
        absorbing_features = PΠφ * absorb_mask
        absorbing_traces = absorbing_features # for now: use abosrbing features since we don't have a trace for the fake data.
        
        # IV formulation for state-action
        A_absorb = (1 - γ) * jnp.einsum("nmi, nmj -> ij", absorbing_traces, absorbing_features)
        b_absorb = (absorbing_traces * rho[..., None]).sum(axis=batch_axes)

        # ------------------------------------------------------------
        # 3. Accumulate Memory
        # ------------------------------------------------------------
        A_i = lstd_state["A"] + A_batch + A_absorb
        b_i = lstd_state["b"] + b_batch + b_absorb

        # ------------------------------------------------------------
        # 4. Diagonal Prior
        # ------------------------------------------------------------
        batch_sa_precision = (Φ**2).sum(axis=batch_axes)
        absorbing_sa_precision = (absorbing_features**2).sum(axis=batch_axes)
        
        # Safe local immutable update
        new_sa_diag_counts = lstd_state["sa_diag_counts"] + batch_sa_precision + absorbing_sa_precision

        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_kA = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_sa_diag_counts)
        lambda_kA = jnp.where(lambda_kA >= 0.1, lambda_kA, 0.0)
        Lambda_mat = jnp.diag(lambda_kA) 

        dim_kA = A_batch.shape[0] 
        reg = jnp.eye(dim_kA) * config["A_REGULARIZATION_PER_STEP"]
        
        # ------------------------------------------------------------
        # 4. Final Solve
        # ------------------------------------------------------------
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
            "S": jnp.eye(k),
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
                
                # Extract pre-computed true next values for exact episodic boundaries
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
            
            # --- Intrinsic Reward (due to Precision) ---
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + 1e-8 * jnp.eye(k) ,jnp.eye(k),)

            ρ_from_phi = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            phi_next_s = get_phi(traj_batch.next_obs)
            rho = ρ_from_phi(phi_next_s)  

            ri_scale = lstd_state["Beta"]

            # --- LSTD ---
            phi_s = get_phi(traj_batch.obs) 
            phi_sa = expand_to_sa_features(phi_s, n_actions, traj_batch.action)
            
            # Use unified trace masking
            traces = helpers.calculate_traces(
                traj_batch, phi_sa, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing
            )
            
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            lstd_state = LSTDQ(lstd_state, traj_batch, phi_sa, phi_next_s, traces)

            # Intrinsic values (scaled):
            def get_vi(obs):
                phi = get_phi(obs) 
                Pi = jnp.ones((*phi.shape[:-1], n_actions)) * (1.0 / n_actions)
                phi_policy = expected_next_sa_features(phi, Pi)
                return phi_policy @ lstd_state["w"] * ri_scale

            v_i = get_vi(traj_batch.obs)
            next_v_i = get_vi(traj_batch.next_obs)

            # Scale vi and ri in traj_batch for GAE.
            traj_batch = traj_batch._replace(i_value=v_i, next_i_val=next_v_i, intrinsic_reward=rho * ri_scale)

            # --- GAE ---
            # Using the exact same unified GAE function from standard LSTD
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
            sigma_state= helpers.update_cov(
                traj_batch, sigma_state, get_phi,
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
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
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