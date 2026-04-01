# Model Based Evaluation of ON-POLICY behavior
# State Features (LSTD-V system) + Optimistic Norm Scaling
# ABSORBING
from logging import config

from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "3_31_linear_model_v_optimistic" 

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

    alpha_fn = lambda t: jnp.maximum(config.get("MIN_COV_LR", 1 / 10), 1 / t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config["ALPHA_SCHEDULE"], config["MIN_LSTD_LR"])
    alpha_fn_lstd_b = helpers.get_alpha_schedule(config["ALPHA_SCHEDULE"], config["MIN_LSTD_LR_RI"])
    
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    # def LinearModelVI(model_state: Dict, transitions, features, next_features, ri_scale):
    #     """
    #     Learns M (k -> k), w_e (k), w_i (k), and w_u (k).
    #     Performs on-policy weight-space VI using the joint norm ||w_e + w_i||.
    #     """
    #     batch_axes = tuple(range(transitions.done.ndim))
    #     batch_size = transitions.done.size
    #     t = model_state["t"]
        
    #     r_e = transitions.reward
    #     rho_i = transitions.intrinsic_reward
        
    #     Φ = features             # phi(s)   -> Shape: (Batch, Envs, k)
    #     next_phi = next_features # phi(s')  -> Shape: (Batch, Envs, k)
        
    #     γ_e = config.get("GAMMA", 0.99)
    #     γ_i = config.get("GAMMA_i", 0.99)
    #     k = config["RND_FEATURES"]

    #     # 1. Accumulate Empirical Statistics (EMA)
    #     Sigma_batch = jnp.einsum("nmi, nmj -> ij", Φ, Φ) / batch_size
    #     M_num_batch = jnp.einsum("nmi, nmj -> ij", Φ, next_phi) / batch_size
    #     w_e_num_batch = (Φ * r_e[..., None]).sum(axis=batch_axes) / batch_size
    #     w_i_num_batch = (Φ * rho_i[..., None]).sum(axis=batch_axes) / batch_size

    #     # --- Linearized Uncertainty Targets ---
    #     reg_val = config.get("A_REGULARIZATION_PER_STEP", 1e-4)
    #     S_inv_prev = jnp.linalg.inv(model_state["Sigma"] + jnp.eye(k) * reg_val)
    #     rho_local = jnp.sqrt(jnp.einsum("nmi,ij,nmj->nm", Φ, S_inv_prev, Φ))
    #     w_u_num_batch = (Φ * rho_local[..., None]).sum(axis=batch_axes) / batch_size

    #     # 2. Absorbing Ghost Transitions
    #     absorb_mask = jnp.where(config.get("ABSORBING_TERMINAL_STATE", True), transitions.done, 0)
    #     phi_C_s = next_phi 
        
    #     Sigma_abs = jnp.einsum("nmi, nmj -> ij", phi_C_s * absorb_mask[..., None], phi_C_s)
    #     M_num_abs = jnp.einsum("nmi, nmj -> ij", phi_C_s * absorb_mask[..., None], next_phi)
        
    #     w_e_num_abs = (phi_C_s * (r_e[..., None] * absorb_mask[..., None])).sum(axis=batch_axes)
    #     w_i_num_abs = (phi_C_s * (rho_i[..., None] * absorb_mask[..., None])).sum(axis=batch_axes)
        
    #     rho_local_abs = jnp.sqrt(jnp.einsum("nmi,ij,nmj->nm", phi_C_s, S_inv_prev, phi_C_s))
    #     w_u_num_abs = (phi_C_s * (rho_local_abs[..., None] * absorb_mask[..., None])).sum(axis=batch_axes)

    #     # Normalize and Update EMAs
    #     denom = batch_size + absorb_mask.sum()
    #     Sigma_i = helpers.EMA(alpha_fn_lstd(t), model_state["Sigma"], (Sigma_batch * batch_size + Sigma_abs) / denom)
    #     M_num_i = helpers.EMA(alpha_fn_lstd(t), model_state["M_num"], (M_num_batch * batch_size + M_num_abs) / denom)
        
    #     w_e_num_i = helpers.EMA(alpha_fn_lstd_b(t), model_state["w_e_num"], (w_e_num_batch * batch_size + w_e_num_abs) / denom)
    #     w_i_num_i = helpers.EMA(alpha_fn_lstd_b(t), model_state["w_i_num"], (w_i_num_batch * batch_size + w_i_num_abs) / denom)
    #     w_u_num_i = helpers.EMA(alpha_fn_lstd(t), model_state["w_u_num"], (w_u_num_batch * batch_size + w_u_num_abs) / denom)

    #     # ------------------------------------------------------------
    #     # 3. Solve for explicit Model Components
    #     # ------------------------------------------------------------
    #     Sigma_view = Sigma_i + jnp.eye(k) * reg_val
    #     Sigma_inv = jnp.linalg.inv(Sigma_view)
        
    #     M = Sigma_inv @ M_num_i          # R^{k x k}
    #     w_e_base = Sigma_inv @ w_e_num_i # Extrinsic base
    #     w_i_base = Sigma_inv @ w_i_num_i # Intrinsic base
        
    #     # Scale the uncertainty weights down to match the decayed intrinsic reward scale
    #     w_u = Sigma_inv @ w_u_num_i * ri_scale      

    #     # 4. Concurrent Weight-Space Value Iteration
    #     def vi_step(w_vs, _):
    #         w_v_e, w_v_i = w_vs
            
    #         # The JOINT NORM for scaling the bonus
    #         # Align the intrinsic scale before taking the norm
    #         aligned_w_v_i = w_v_i * ri_scale
    #         w_v_joint_norm = jnp.linalg.norm(w_v_e + aligned_w_v_i)
            
    #         # Cap the norm to prevent exponential explosion
    #         w_v_joint_norm = jnp.minimum(w_v_joint_norm, model_state["V_max"])
            
    #         # Linearized Optimistic Bonus
    #         beta_M = config.get("BETA_M", 0.05)
    #         optimistic_bonus = (γ_i * w_v_joint_norm * beta_M) * w_u
            
    #         # On-Policy Bellman Updates
    #         w_v_e_out = w_e_base + γ_e * (M @ w_v_e)
    #         w_v_i_out = w_i_base + γ_i * (M @ w_v_i) + optimistic_bonus
            
    #         return (w_v_e_out, w_v_i_out), None

    #     num_vi_rounds = config.get("VI_ROUNDS", 100)
    #     # (w_v_e_final, w_v_i_final), _ = jax.lax.scan(
    #     #     vi_step, 
    #     #     (model_state["w_v_e"], model_state["w_v_i"]), 
    #     #     None, 
    #     #     length=num_vi_rounds
    #     # )

    #     (w_v_e_final, w_v_i_final), _ = jax.lax.scan(
    #         vi_step, 
    #         (w_e_base, w_i_base), 
    #         None, 
    #         length=num_vi_rounds
    #     )


    #     return {
    #         "Sigma": Sigma_i,
    #         "M_num": M_num_i,
    #         "w_e_num": w_e_num_i,
    #         "w_i_num": w_i_num_i,
    #         "w_u_num": w_u_num_i,
    #         "w_v_e": w_v_e_final,
    #         "w_v_i": w_v_i_final, 
    #         "t": t + 1,
    #         "Beta": model_state["Beta"],
    #         "V_max": model_state["V_max"],                 
    #     }

    def LinearModelVI(model_state: Dict, transitions, features, next_features, ri_scale):
        """
        Learns M (k -> k), w_e (k), w_i (k), and w_u (k).
        Performs on-policy weight-space VI using the joint norm ||w_e + w_i||.
        """
        batch_axes = tuple(range(transitions.done.ndim))
        batch_size = transitions.done.size
        t = model_state["t"]
        
        r_e = transitions.reward
        
        # rho_i is passed in from update_cov_and_get_rho. 
        # It represents the global epistemic novelty of the NEXT state.
        rho_i = transitions.intrinsic_reward
        
        Φ = features             # phi(s)   -> Shape: (Batch, Envs, k)
        next_phi = next_features # phi(s')  -> Shape: (Batch, Envs, k)
        
        γ_e = config.get("GAMMA", 0.99)
        γ_i = config.get("GAMMA_i", 0.99)
        k = config["RND_FEATURES"]

        # 1. Accumulate Empirical Statistics (EMA)
        Sigma_batch = jnp.einsum("nmi, nmj -> ij", Φ, Φ) / batch_size
        M_num_batch = jnp.einsum("nmi, nmj -> ij", Φ, next_phi) / batch_size
        w_e_num_batch = (Φ * r_e[..., None]).sum(axis=batch_axes) / batch_size
        
        # Both intrinsic rewards and uncertainty targets are driven by rho(s')
        w_i_num_batch = (Φ * rho_i[..., None]).sum(axis=batch_axes) / batch_size
        w_u_num_batch = w_i_num_batch 

        # 2. Absorbing Ghost Transitions
        absorb_mask = jnp.where(config.get("ABSORBING_TERMINAL_STATE", True), transitions.done, 0)
        phi_C_s = next_phi 
        
        Sigma_abs = jnp.einsum("nmi, nmj -> ij", phi_C_s * absorb_mask[..., None], phi_C_s)
        M_num_abs = jnp.einsum("nmi, nmj -> ij", phi_C_s * absorb_mask[..., None], next_phi)
        
        w_e_num_abs = (phi_C_s * (r_e[..., None] * absorb_mask[..., None])).sum(axis=batch_axes)
        
        # Ghost transitions loop to themselves, so the "next" state novelty is still rho_i
        w_i_num_abs = (phi_C_s * (rho_i[..., None] * absorb_mask[..., None])).sum(axis=batch_axes)
        w_u_num_abs = w_i_num_abs

        # Normalize and Update EMAs
        denom = batch_size + absorb_mask.sum()
        Sigma_i = helpers.EMA(alpha_fn_lstd(t), model_state["Sigma"], (Sigma_batch * batch_size + Sigma_abs) / denom)
        M_num_i = helpers.EMA(alpha_fn_lstd(t), model_state["M_num"], (M_num_batch * batch_size + M_num_abs) / denom)
        
        w_e_num_i = helpers.EMA(alpha_fn_lstd_b(t), model_state["w_e_num"], (w_e_num_batch * batch_size + w_e_num_abs) / denom)
        w_i_num_i = helpers.EMA(alpha_fn_lstd_b(t), model_state["w_i_num"], (w_i_num_batch * batch_size + w_i_num_abs) / denom)
        w_u_num_i = helpers.EMA(alpha_fn_lstd(t), model_state["w_u_num"], (w_u_num_batch * batch_size + w_u_num_abs) / denom)

        # ------------------------------------------------------------
        # 3. Solve for explicit Model Components
        # ------------------------------------------------------------
        reg_val = config.get("A_REGULARIZATION_PER_STEP", 1e-4)
        Sigma_view = Sigma_i + jnp.eye(k) * reg_val
        Sigma_inv = jnp.linalg.inv(Sigma_view)
        
        M = Sigma_inv @ M_num_i          # R^{k x k}
        w_e_base = Sigma_inv @ w_e_num_i # Extrinsic base
        w_i_base = Sigma_inv @ w_i_num_i # Intrinsic base
        
        # Uncertainty weights (Calculated in purely UNSCALED units)
        w_u = Sigma_inv @ w_u_num_i      

        # 4. Concurrent Weight-Space Value Iteration
        def vi_step(w_vs, _):
            w_v_e, w_v_i = w_vs
            
            # The JOINT NORM for scaling the bonus
            # Align the intrinsic scale before taking the norm
            aligned_w_v_i = w_v_i * ri_scale
            w_v_joint_norm = jnp.linalg.norm(w_v_e + aligned_w_v_i)
            
            # Cap the norm to prevent exponential explosion
            w_v_joint_norm = jnp.minimum(w_v_joint_norm, model_state["V_max"])
            
            # Linearized Optimistic Bonus 
            # (Scale by ri_scale HERE, preventing double-scaling)
            beta_M = config.get("BETA_M", 0.05)
            optimistic_bonus = (γ_i * w_v_joint_norm * beta_M * ri_scale) * w_u
            
            # On-Policy Bellman Updates
            w_v_e_out = w_e_base + γ_e * (M @ w_v_e)
            w_v_i_out = w_i_base + γ_i * (M @ w_v_i) + optimistic_bonus
            
            return (w_v_e_out, w_v_i_out), None

        num_vi_rounds = config.get("VI_ROUNDS", 100)
        
        # COLD START: Iterate from the empirical base to wash out stale optimism
        (w_v_e_final, w_v_i_final), _ = jax.lax.scan(
            vi_step, 
            (w_e_base, w_i_base), 
            None, 
            length=num_vi_rounds
        )

        return {
            "Sigma": Sigma_i,
            "M_num": M_num_i,
            "w_e_num": w_e_num_i,
            "w_i_num": w_i_num_i,
            "w_u_num": w_u_num_i,
            "w_v_e": w_v_e_final,
            "w_v_i": w_v_i_final, 
            "t": t + 1,
            "Beta": model_state["Beta"],
            "V_max": model_state["V_max"],                 
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

        dim_k = k
        initial_model_state = {
            "Sigma": jnp.eye(dim_k) * config["A_REGULARIZATION"],
            "M_num": jnp.zeros((dim_k, k)),
            "w_e_num": jnp.zeros(dim_k),
            "w_i_num": jnp.zeros(dim_k),
            "w_u_num": jnp.zeros(dim_k),
            "w_v_e": jnp.zeros(dim_k),
            "w_v_i": jnp.zeros(dim_k),
            "t": 1,
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
        }
        initial_sigma_state = {
            "S": jnp.zeros((k, k)),
            "N": 1,
            "t": 1,
        }

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            (train_state, model_state, sigma_state, rnd_state, env_state, last_obs, rng, idx) = runner_state

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
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + config['GRAM_REG'] * jnp.eye(k) ,jnp.eye(k),)

            ρ_from_phi = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            phi_next_s = get_phi(traj_batch.next_obs)
            rho = ρ_from_phi(phi_next_s)  

            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state["N"]))
            ri_scale = model_state["Beta"] / sqrt_n

            # --- ON-POLICY LSTD ---
            phi_s = get_phi(traj_batch.obs) 
            
            traces = helpers.calculate_traces(
                traj_batch, phi_s, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing
            )
            
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            model_state = LinearModelVI(model_state, traj_batch, phi_s, phi_next_s, ri_scale)
            
            # Intrinsic values (scaled):
            def get_vi(obs):
                phi = get_phi(obs) 
                return phi @ model_state["w_v_i"] * ri_scale

            v_i = get_vi(traj_batch.obs)
            next_v_i = get_vi(traj_batch.next_obs)

            # Scale vi and ri in traj_batch for GAE.
            traj_batch = traj_batch._replace(i_value=v_i, next_i_val=next_v_i, intrinsic_reward=rho * ri_scale)

            # --- GAE ---
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
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "mean_rew": traj_batch.reward.mean(),
                    "beta": model_state["Beta"],
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
                    model_state["Beta"], network, train_state, traj_batch, get_vi,
                )

            runner_state = (train_state, model_state, sigma_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_model_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)