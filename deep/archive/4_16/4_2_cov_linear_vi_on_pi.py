# Model Based Evaluation of ON-POLICY behavior
# State Features (LSTD-V system)
from logging import config

from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_2_cov_linear_vi_on_pi" 
LEAK_FACTOR = (1 - 1e-3)

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
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]  
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(bonus_sq)

    def LinearModelVI(model_state: Dict, transitions, features, next_features):
        """
        Learns M (k -> k) and w_r (k), then performs weight-space VI.
        Evaluates the On-Policy state transitions directly.
        """
        batch_axes = tuple(range(transitions.done.ndim))
        t = model_state["t"]
        rho = transitions.intrinsic_reward
        
        Φ = features             # phi(s)   -> Shape: (Batch, Envs, k)
        next_phi = next_features # phi(s')  -> Shape: (Batch, Envs, k)
        
        γ = config["GAMMA_i"]
        k = config["RND_FEATURES"]

        # 1. Accumulate Empirical Statistics (EMA)
        # Sigma (k x k): Second moment of state features
        Sigma_batch = jnp.einsum("nmi, nmj -> ij", Φ, Φ)
        # M_num (k x k): Cross-correlation between phi(s) and phi(s') under behavior policy
        M_num_batch = jnp.einsum("nmi, nmj -> ij", Φ, next_phi)
        # w_r_num (k): Correlation between phi(s) and reward
        w_r_num_batch = (Φ * rho[..., None]).sum(axis=batch_axes)

        # 2. Absorbing Ghost Transitions
        # Terminal states transition to themselves
        absorb_mask = jnp.where(config.get("ABSORBING_GOAL_STATE", True), transitions.done, 0)
        phi_C_s = next_phi 
        
        Sigma_abs = jnp.einsum("nmi, nmj -> ij", phi_C_s * absorb_mask[..., None], phi_C_s)
        M_num_abs = jnp.einsum("nmi, nmj -> ij", phi_C_s * absorb_mask[..., None], next_phi)
        w_r_num_abs = (phi_C_s * (rho[..., None] * absorb_mask[..., None])).sum(axis=batch_axes)

        # Normalize and Update EMAs
        Sigma_i = LEAK_FACTOR * model_state['Sigma'] + Sigma_batch + Sigma_abs
        M_num_i = LEAK_FACTOR * model_state['M_num'] + M_num_batch + M_num_abs
        w_r_num_i = LEAK_FACTOR * model_state['w_r_num'] + w_r_num_batch + w_r_num_abs
        # ------------------------------------------------------------
        # 3. Track Diagonal Prior Counts
        # ------------------------------------------------------------
        batch_s_precision = (Φ**2).sum(axis=batch_axes)
        absorbing_s_precision = (phi_C_s**2 * absorb_mask[..., None]).sum(axis=batch_axes)
        
        # Safe local immutable update
        new_s_diag_counts = model_state["s_diag_counts"] + batch_s_precision + absorbing_s_precision 

        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_s_diag_counts)
        lambda_k = jnp.where(lambda_k >= 0.1, lambda_k, 0.0)
        Lambda_mat = jnp.diag(lambda_k) 

        # ------------------------------------------------------------
        # 4. Apply Prior to solve for explicit Model Components
        # ------------------------------------------------------------
        reg = jnp.eye(k) * config.get("A_REGULARIZATION_PER_STEP", 1e-4)
        
        # Add prior to the precision matrix
        Sigma_view = Sigma_i + Lambda_mat + reg
        Sigma_inv = jnp.linalg.inv(Sigma_view)
        # Add prior to the reward targets
        prior_b = lambda_k * model_state["V_max"]
        w_r_view = w_r_num_i + prior_b
        
        M = jnp.linalg.solve(Sigma_view, M_num_i) # Transition Model M: R^{k x k}
        w_r = jnp.linalg.solve(Sigma_view, w_r_view) # Reward Weights w_r: R^{k}
        
        # 5. Weight-Space Value Iteration
        def vi_step(w_v_in, _):
            # Apply Bellman Operator directly on state weights
            w_v_out = w_r + γ * (M @ w_v_in)
            w_v_out = jnp.clip(w_v_out, -1/(1-config['GAMMA_i']), 1/(1-config['GAMMA_i']))
            return w_v_out, None

        num_vi_rounds = config.get("VI_ROUNDS", 200)
        # w_v_final, _ = jax.lax.scan(vi_step, model_state["w_v"], None, length=num_vi_rounds)
        # COLD START:
        w_v_final, _ = jax.lax.scan(vi_step, w_r, None, length=num_vi_rounds)

        return {
            "Sigma": Sigma_i,
            "M_num": M_num_i,
            "w_r_num": w_r_num_i,
            "w_v": w_v_final, 
            "t": t + 1,
            "Beta": model_state["Beta"],
            "V_max": model_state["V_max"],                 
            "s_diag_counts": new_s_diag_counts,          
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
        
        V_max = (config['BONUS_SCALE']) / (1 - config["GAMMA_i"]) # maximum intrinsic values
        if config["NORMALIZE_FEATURES"]:
            V_max /= jnp.sqrt(k)

        # Everything uses k dimensions now, instead of kA
        dim_k = k
        initial_model_state = {
            "Sigma": jnp.eye(dim_k) * config["A_REGULARIZATION"],
            "M_num": jnp.zeros((dim_k, k)),
            "w_r_num": jnp.zeros(dim_k),
            "w_v": jnp.zeros(dim_k), 
            "t": 1,
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            "s_diag_counts": jnp.zeros(dim_k),
        }
        initial_sigma_state = {
            "S": jnp.eye(k),
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
            Sigma_inv = jnp.linalg.solve(sigma_state["S"] + 1e-8 * jnp.eye(k) , jnp.eye(k),)

            ρ_from_phi = lambda phi: get_scale_free_bonus(Sigma_inv, phi)
            phi_next_s = get_phi(traj_batch.next_obs)
            rho = ρ_from_phi(phi_next_s)  

            ri_scale = model_state["Beta"]

            # --- ON-POLICY LSTD ---
            # Using raw state features directly (no sa expansion)
            phi_s = get_phi(traj_batch.obs) 
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            model_state = LinearModelVI(model_state, traj_batch, phi_s, phi_next_s)
            
            # Intrinsic values (scaled):
            def get_vi(obs):
                phi = get_phi(obs) 
                # Direct dot product with the state-value weights calculated via VI
                return phi @ model_state["w_v"] * ri_scale

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
            sigma_state = helpers.update_cov(
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