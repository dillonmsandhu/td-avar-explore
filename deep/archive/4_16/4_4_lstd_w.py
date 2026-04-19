# Full accumulation (Global A, Buffer B)
# Simultaneous Dual-Solve (w_i, w_e) for Adaptive ||w||_infty Bonus
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_4_cov_lstd_buffer_w"

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

class LSTDBufferState(NamedTuple):
    traces: jnp.ndarray
    features: jnp.ndarray
    next_features: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    ext_rewards: jnp.ndarray   # <-- ADDED to compute b_e online
    ptr: jnp.ndarray
    full: jnp.ndarray

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        """bonus = sqrt(x^T Σ^{-1} x)"""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

    def update_buffer(buffer_state: LSTDBufferState, traces, features, next_features, terminals, absorb_masks, ext_rew):
        """Inserts a new batch of transitions into the JAX FIFO ring buffer."""
        traces = traces.reshape(-1, k)
        features = features.reshape(-1, k)
        next_features = next_features.reshape(-1, k)
        terminals = terminals.reshape(-1, 1)
        absorb_masks = absorb_masks.reshape(-1, 1)
        ext_rew = ext_rew.reshape(-1, 1) # <-- Shape extrinsic rewards for buffer
        
        B = features.shape[0]
        indices = (buffer_state.ptr + jnp.arange(B)) % BUFFER_CAPACITY
        
        new_traces = buffer_state.traces.at[indices].set(traces)
        new_features = buffer_state.features.at[indices].set(features)
        new_next_features = buffer_state.next_features.at[indices].set(next_features)
        new_terminals = buffer_state.terminals.at[indices].set(terminals)
        new_absorb_masks = buffer_state.absorb_masks.at[indices].set(absorb_masks)
        new_ext_rewards = buffer_state.ext_rewards.at[indices].set(ext_rew)
        
        new_ptr = (buffer_state.ptr + B) % BUFFER_CAPACITY
        new_full = jnp.logical_or(buffer_state.full, buffer_state.ptr + B >= BUFFER_CAPACITY)
        
        return LSTDBufferState(
            traces=new_traces, features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks, ext_rewards=new_ext_rewards,
            ptr=new_ptr, full=new_full
        )

    def solve_lstd_buffer(buffer_state: LSTDBufferState, Sigma_inv, lstd_state_dict, config):
        """Computes b_i and b_e from the buffer, divides by N_buffer, and solves against global A / N_total."""
        CHUNK_SIZE = 10_000 
        NUM_CHUNKS = BUFFER_CAPACITY // CHUNK_SIZE
        
        chunked_phi = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_next_phi = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_traces = buffer_state.traces.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_ext_rew = buffer_state.ext_rewards.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = jnp.where(buffer_state.full, True, jnp.arange(BUFFER_CAPACITY) < buffer_state.ptr)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)

        def process_chunk(carry, chunk_data):
            b_i_acc, b_e_acc, n_acc = carry
            c_phi, c_next_phi, c_traces, c_term, c_absorb, c_ext_rew, c_mask = chunk_data
            
            # Retroactive Intrinsic Reward
            next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi)
            
            c_traces_masked = c_traces * c_mask
            
            # 1. Standard Targets
            b_batch_i = jnp.einsum("ni, n -> i", c_traces_masked, next_rho)
            b_batch_e = jnp.einsum("ni, n -> i", c_traces_masked, c_ext_rew.squeeze(-1))
            
            # 2. Absorbing Targets
            abs_traces = c_next_phi * c_absorb
            b_abs_i = jnp.einsum("ni, n -> i", abs_traces * c_mask, next_rho)
            b_abs_e = jnp.zeros_like(b_batch_e) # Absorbing state has 0 extrinsic target
            
            chunk_valid_n = jnp.sum(c_mask) + jnp.sum(c_absorb * c_mask)
            
            return (b_i_acc + b_batch_i + b_abs_i, b_e_acc + b_batch_e + b_abs_e, n_acc + chunk_valid_n), None

        init_b_i = jnp.zeros(k)
        init_b_e = jnp.zeros(k)
        init_n = jnp.array(0.0)
        
        (final_b_i, final_b_e, buf_N), _ = jax.lax.scan(
            process_chunk, 
            (init_b_i, init_b_e, init_n), 
            (chunked_phi, chunked_next_phi, chunked_traces, chunked_terminals, chunked_absorb, chunked_ext_rew, chunked_mask)
        )

        # --- Scale A and b to Expected Values ---
        global_N = jnp.maximum(1.0, lstd_state_dict["N_total"])
        A_mean = lstd_state_dict["A"] / global_N
        
        safe_buf_N = jnp.maximum(1.0, buf_N)
        b_i_mean = final_b_i / safe_buf_N
        b_e_mean = final_b_e / safe_buf_N
        
        # --- Bayesian Optimistic Prior (Diagonal) ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        new_phi_diag_counts = lstd_state_dict["phi_diag_counts"]
        
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts)
        lambda_k = jnp.where(lambda_k >= 0.1, lambda_k, 0.0)
        Lambda_mat = jnp.diag(lambda_k)
        
        prior_b_i = jnp.diag(Lambda_mat) * lstd_state_dict["V_max"]
        # Note: We do NOT apply optimism to the extrinsic value vector
        
        # --- Simultaneous Solve ---
        reg = jnp.eye(k) * config["A_REGULARIZATION_PER_STEP"]
        A_view = A_mean + Lambda_mat + reg
        
        b_i_view = b_i_mean + prior_b_i
        b_e_view = b_e_mean # No prior added
        
        # Stack targets [k, 2] and solve simultaneously
        B_mat = jnp.stack([b_i_view, b_e_view], axis=1) 
        W_mat = jnp.linalg.solve(A_view, B_mat)
        
        w_i = W_mat[:, 0]
        w_e = W_mat[:, 1]
        
        return {
            "w": w_i,
            "w_e": w_e,
            "Beta": lstd_state_dict["Beta"],
            "V_max": lstd_state_dict["V_max"],
            "phi_diag_counts": new_phi_diag_counts,
            "A": lstd_state_dict["A"],            
            "N_total": lstd_state_dict["N_total"] 
        }

    V_max = 1.0 / (1 - config["GAMMA_i"]) # maximum intrinsic values (before scaling by beta)
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k)

    def train(rng):
        initial_lstd_state = {
            "w": jnp.zeros(k),
            "w_e": jnp.zeros(k),
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            "phi_diag_counts": jnp.zeros(k),
            "A": jnp.zeros((k, k)),           
            "N_total": jnp.array(0.0, dtype=jnp.float32) 
        }
        
        initial_buffer_state = LSTDBufferState(
            traces=jnp.zeros((BUFFER_CAPACITY, k)),
            features=jnp.zeros((BUFFER_CAPACITY, k)),
            next_features=jnp.zeros((BUFFER_CAPACITY, k)),
            terminals=jnp.zeros((BUFFER_CAPACITY, 1)),
            absorb_masks=jnp.zeros((BUFFER_CAPACITY, 1)),
            ext_rewards=jnp.zeros((BUFFER_CAPACITY, 1)),
            ptr=jnp.array(0, dtype=jnp.int32),
            full=jnp.array(False, dtype=jnp.bool_)
        )
        
        initial_sigma_state = {"S": jnp.eye(k)} 

        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k
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
            train_state, lstd_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, rng, idx = runner_state

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
                    done, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, rnd_state, env_state, obsv, rng), transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # --- 1. FEATURE EXTRACTION ---
            phi = batch_get_features(traj_batch.obs)
            next_phi = batch_get_features(traj_batch.next_obs)
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            
            # --- 2. PRE-UPDATE SURPRISE (For GAE) ---
            Sigma_inv_old = jnp.linalg.solve(sigma_state["S"] + 1e-8 * jnp.eye(k), jnp.eye(k))
            batch_next_rho_pre = get_scale_free_bonus(Sigma_inv_old, next_phi)

            # --- 3. COVARIANCE UPDATE (Pure Sum) ---
            phi_flat = phi.reshape(-1, k)
            new_S = sigma_state["S"] + jnp.einsum("ni,nj->ij", phi_flat, phi_flat)
            sigma_state = {"S": new_S}
            Sigma_inv_new = jnp.linalg.solve(sigma_state["S"] + 1e-8 * jnp.eye(k), jnp.eye(k))

            # --- 4. BUFFER UPDATE (Including Extrinsic Reward) ---
            buffer_state = update_buffer(
                buffer_state, traces, phi, next_phi, terminals, absorb_masks, traj_batch.reward
            )

            # --- 5. UPDATE GLOBAL A MATRIX & PRIORS ---
            gamma_i = config["GAMMA_i"]
            
            delta_Phi = phi - gamma_i * (1 - terminals[..., None]) * next_phi
            A_batch = jnp.einsum("nmi, nmj -> ij", traces, delta_Phi)
            
            abs_features_for_A = next_phi * absorb_masks[..., None]
            abs_traces_for_A = abs_features_for_A 
            A_abs = (1 - gamma_i) * jnp.einsum("nmi, nmj -> ij", abs_traces_for_A, abs_features_for_A)
            
            batch_valid_N = phi.shape[0] * phi.shape[1] + jnp.sum(absorb_masks)
            
            batch_phi_precision = (phi**2).sum(axis=(0, 1))
            absorbing_phi_precision = (abs_features_for_A**2).sum(axis=(0, 1))
            new_phi_diag_counts = lstd_state["phi_diag_counts"] + batch_phi_precision + absorbing_phi_precision

            lstd_state = {
                **lstd_state,
                "A": lstd_state["A"] + A_batch + A_abs,
                "N_total": lstd_state["N_total"] + batch_valid_N,
                "phi_diag_counts": new_phi_diag_counts
            }

            # --- 6. SIMULTANEOUS LSTD SOLVE OVER BUFFER ---
            lstd_state = solve_lstd_buffer(buffer_state, Sigma_inv_new, lstd_state, config)

            # --- 7. DYNAMIC BETA & BATCH EVALUATION ---
            # Adaptive Bonus Scale using L_infty norm of Extrinsic Weights
            w_e_norm = jnp.max(jnp.abs(lstd_state["w_e"]))
            scaling_factor = jnp.minimum(lstd_state['V_max'], w_e_norm)
            dynamic_beta = config["BONUS_SCALE"] + config["BONUS_SCALE"] * scaling_factor # (beta_r + beta_M)  rho ||w|| 
            
            lstd_state["Beta"] = dynamic_beta 
            rho_scale = dynamic_beta
            
            v_i = phi @ lstd_state["w"] * rho_scale
            next_v_i = next_phi @ lstd_state["w"] * rho_scale

            # Critical: Reward Uses PRE-UPDATE Surprise
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=batch_next_rho_pre * rho_scale, next_i_val=next_v_i)

            # --- ADVANTAGE CALCULATION ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"],
                is_episodic=is_episodic, is_absorbing=is_absorbing,
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            
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
                    "w_e_norm": w_e_norm, 
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
                    rho = get_scale_free_bonus(Sigma_inv_new, phi) * rho_scale
                    return rho

                def get_vi(obs):
                    return batch_get_features(obs) @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(
                    config, metric, int_rew_from_state, evaluator,
                    lstd_state["Beta"], network, train_state, traj_batch, get_vi,
                )

            runner_state = (train_state, lstd_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)