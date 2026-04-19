from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_8_lstd_prb"

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
    size: jnp.ndarray  # Replaces ptr and full to support extended buffer compaction

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    # --- Feature Augmentation Toggle ---
    augment_lstd = config.get("AUGMENT_LSTD_FEATURES", False)
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    # --- Buffer Padding for Extended Collection and Chunking ---
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    CHUNK_SIZE = 100_000 + batch_size
    NUM_CHUNKS = (EXTENDED_CAPACITY + CHUNK_SIZE - 1) // CHUNK_SIZE
    PADDED_CAPACITY = NUM_CHUNKS * CHUNK_SIZE
    
    k_base = config.get("RND_FEATURES", 128)
    k_lstd = k_base + 2 if augment_lstd else k_base  # Adjust LSTD weight dimension
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        """bonus = sqrt(x^T Σ^{-1} x)"""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    
    def update_phi_precision(lstd_state, features, next_features, done):
        phi_diag_precision = lstd_state['phi_diag_counts']
        absorb_mask = jnp.where(is_absorbing, done, 0)
        # Calculate precision for this batch
        batch_phi_prec = jnp.sum(features**2, axis=(0, 1)) 
        # Calculate precision for absorbing transitions
        absorbing_features = next_features * absorb_mask[..., None]
        abs_phi_prec = jnp.sum(absorbing_features**2, axis=(0, 1)) 
        new_counts = batch_phi_prec + abs_phi_prec
        return {**lstd_state, "phi_diag_counts": phi_diag_precision + new_counts}
    
    def update_buffer(buffer_state: LSTDBufferState, traces, features, next_features, terminals, absorb_masks):
            """Appends a new batch of transitions to the extended JAX buffer."""
            # Cast ALL inputs to float32 to match jnp.zeros() initialization
            traces = traces.reshape(-1, k_base).astype(jnp.float32)
            features = features.reshape(-1, k_base).astype(jnp.float32)
            next_features = next_features.reshape(-1, k_base).astype(jnp.float32)
            terminals = terminals.reshape(-1, 1).astype(jnp.float32)
            absorb_masks = absorb_masks.reshape(-1, 1).astype(jnp.float32)
            
            B = features.shape[0]
            start_idx = buffer_state.size
            
            # Dynamically append to the exact end of the currently valid buffer
            new_traces = jax.lax.dynamic_update_slice(buffer_state.traces, traces, (start_idx, 0))
            new_features = jax.lax.dynamic_update_slice(buffer_state.features, features, (start_idx, 0))
            new_next_features = jax.lax.dynamic_update_slice(buffer_state.next_features, next_features, (start_idx, 0))
            new_terminals = jax.lax.dynamic_update_slice(buffer_state.terminals, terminals, (start_idx, 0))
            new_absorb_masks = jax.lax.dynamic_update_slice(buffer_state.absorb_masks, absorb_masks, (start_idx, 0))
            
            return LSTDBufferState(
                traces=new_traces, features=new_features, next_features=new_next_features,
                terminals=new_terminals, absorb_masks=new_absorb_masks,
                size=start_idx + B
            )

    def solve_lstd_buffer(buffer_state: LSTDBufferState, Sigma_inv, lstd_state, config):
        """Solves LSTD over the entire extended buffer using a memory-safe chunked scan."""
        # Reshape buffer into chunks
        chunked_phi = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_next_phi = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_traces = buffer_state.traces.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = (jnp.arange(PADDED_CAPACITY) < buffer_state.size)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]

        def process_chunk(carry, chunk_data):
            A_acc, b_acc = carry
            phi, next_phi, traces, term, absorb, mask = chunk_data
            
            curr_rho = get_scale_free_bonus(Sigma_inv, phi)
            next_rho = get_scale_free_bonus(Sigma_inv, next_phi)

            delta_Phi = phi - gamma_i * (1 - term) * next_phi
            A_batch = jnp.einsum("ni, nj -> ij", traces, delta_Phi)
            b_batch = jnp.einsum("ni, n -> i", traces, next_rho * mask.squeeze())
            
            abs_features = next_phi * absorb
            abs_traces = abs_features 
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_traces, abs_features)
            b_abs = jnp.einsum("ni, n -> i", abs_traces, next_rho * absorb.squeeze() * mask.squeeze())
            
            return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs), None

        init_A = jnp.zeros((k_lstd, k_lstd))
        init_b = jnp.zeros(k_lstd)
        
        (final_A, final_b), _ = jax.lax.scan(
            process_chunk, 
            (init_A, init_b), 
            (chunked_phi, chunked_next_phi, chunked_traces, chunked_terminals, chunked_absorb, chunked_mask)
        )
        
        # --- Bayesian Optimistic Prior (Diagonal) ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        new_phi_diag_counts = lstd_state["phi_diag_counts"]
        
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts)
        Lambda_mat = jnp.diag(lambda_k)
        prior_b = lambda_k * lstd_state["V_max"]
        
        # 5. Regularize and Solve ONLY ONCE at the end
        reg = jnp.eye(k_lstd) * config["A_REGULARIZATION_PER_STEP"]
        A_view = final_A + Lambda_mat + reg
        final_b = final_b + prior_b
        
        w_i = jnp.linalg.solve(A_view, final_b)
        
        return {
            "w": w_i,
            "Beta": lstd_state["Beta"],
            "V_max": lstd_state["V_max"],
            "phi_diag_counts": new_phi_diag_counts,
            "A_view": A_view  # Exported specifically for the leverage score computation
        }
        
    def evict_buffer(buffer_state: LSTDBufferState, A_inv, Sigma_inv, config):
        """Computes IV Trace Leverage scores and evicts the lowest-scoring points."""
        size = buffer_state.size
        phi = buffer_state.features
        next_phi = buffer_state.next_features
        traces = buffer_state.traces
        terminals = buffer_state.terminals
        
        valid_mask = jnp.arange(PADDED_CAPACITY) < size
        
        # Reconstruct exactly matching augmented features
        curr_rho = get_scale_free_bonus(Sigma_inv, phi)
        next_rho = get_scale_free_bonus(Sigma_inv, next_phi)
            
        Z = traces
        X = phi - config["GAMMA_i"] * (1 - terminals) * next_phi
        
        # Step 1: Dense Projections into Inverse Matrix spaces
        U = Z @ A_inv.T
        V = X @ A_inv
        W = V @ A_inv.T
        
        # Step 2: Extract Sherman-Morrison scalars
        c = 1.0 - jnp.sum(X * U, axis=-1)
        c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
        
        u_norm = jnp.sum(U * U, axis=-1)
        v_norm = jnp.sum(V * V, axis=-1)
        cross = jnp.sum(U * W, axis=-1)
        
        # Step 3: Compute the exact variance penalty per transition
        scores = (2.0 * cross / c) + (u_norm * v_norm) / (c * c)
        
        # Force invalid buffer items to -infinity so they are guaranteed to drop
        scores = jnp.where(valid_mask, scores, -jnp.inf)
        
        # Step 4: Evict & Compact to exact BUFFER_CAPACITY
        _, keep_indices = jax.lax.top_k(scores, BUFFER_CAPACITY)
        
        new_traces = jnp.zeros_like(buffer_state.traces).at[:BUFFER_CAPACITY].set(buffer_state.traces[keep_indices])
        new_features = jnp.zeros_like(buffer_state.features).at[:BUFFER_CAPACITY].set(buffer_state.features[keep_indices])
        new_next_features = jnp.zeros_like(buffer_state.next_features).at[:BUFFER_CAPACITY].set(buffer_state.next_features[keep_indices])
        new_terminals = jnp.zeros_like(buffer_state.terminals).at[:BUFFER_CAPACITY].set(buffer_state.terminals[keep_indices])
        new_absorb_masks = jnp.zeros_like(buffer_state.absorb_masks).at[:BUFFER_CAPACITY].set(buffer_state.absorb_masks[keep_indices])
        
        # Size caps securely at standard CAPACITY
        new_size = jnp.minimum(size, BUFFER_CAPACITY)
        
        return LSTDBufferState(
            traces=new_traces, features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=new_size
        )

    V_max = 1.0 / (1 - config["GAMMA_i"]) # maximum intrinsic values
    
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k_base)

    def train(rng):
        initial_lstd_state = {
                    "w": jnp.zeros(k_lstd), 
                    "Beta": config["BONUS_SCALE"],
                    "V_max": V_max,
                    "phi_diag_counts": jnp.zeros(k_lstd),
                    "A_view": jnp.eye(k_lstd)  # <-- ADD THIS PLACEHOLDER
                }
            
        # Buffer initialized at PADDED_CAPACITY to support safe extended processing overhead
        initial_buffer_state = LSTDBufferState(
            traces=jnp.zeros((PADDED_CAPACITY, k_base)),
            features=jnp.zeros((PADDED_CAPACITY, k_base)),
            next_features=jnp.zeros((PADDED_CAPACITY, k_base)),
            terminals=jnp.zeros((PADDED_CAPACITY, 1)),
            absorb_masks=jnp.zeros((PADDED_CAPACITY, 1)),
            size=jnp.array(0, dtype=jnp.int32)
        )
        
        initial_sigma_state = {"S": jnp.eye(k_base)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_base
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_base
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

            # Feature Extraction for Current Batch
            phi = batch_get_features(traj_batch.obs)
            next_phi = batch_get_features(traj_batch.next_obs)
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            
            # --- 1. UPDATE EXTENDED BUFFER ---
            buffer_state = update_buffer(buffer_state, traces, phi, next_phi, terminals, absorb_masks)
            
            # --- GLOBAL COVARIANCE UPDATE (Pure Accumulation) --
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_base))
            # Sigma_inv = jnp.linalg.solve(sigma_state["S"], jnp.eye(k_base))
            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)

            # --- 2. SOLVE LSTD OVER EXTENDED BUFFER ---
            lstd_state = update_phi_precision(lstd_state, phi, next_phi, traj_batch.done)
            lstd_state = solve_lstd_buffer(buffer_state, Sigma_inv, lstd_state, config)

            # --- 3 & 4. SCORE AND EVICT BUFFER ---
            A_inv = jnp.linalg.inv(lstd_state["A_view"])
            buffer_state = evict_buffer(buffer_state, A_inv, Sigma_inv, config)

            # --- BATCH INTRINSIC VALUES FOR GAE ---
            batch_curr_rho = get_scale_free_bonus(Sigma_inv, phi)
            batch_next_rho = get_scale_free_bonus(Sigma_inv, next_phi)
            
            # Augment current batch features (Optional based on config)
            if augment_lstd:
                eval_phi = jnp.concatenate([phi, batch_curr_rho[..., None], jnp.ones_like(batch_curr_rho[..., None])], axis=-1)
                eval_next_phi = jnp.concatenate([next_phi, batch_next_rho[..., None], jnp.ones_like(batch_next_rho[..., None])], axis=-1)
            else:
                eval_phi = phi
                eval_next_phi = next_phi

            rho_scale = lstd_state["Beta"]
            v_i = eval_phi @ lstd_state["w"] * rho_scale
            next_v_i = eval_next_phi @ lstd_state["w"] * rho_scale
            
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=batch_next_rho * rho_scale, next_i_val=next_v_i)

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
            # update covariance to incorporate new data
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
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
                    "lambda_k": lstd_state['phi_diag_counts'],
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

                def int_rew_from_state(s,):  # for computing the intrinsic reward given an arbitrary state
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
