
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import flax.linen as nn
SAVE_DIR = "4_9_lstd_prb_rho"

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
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    # --- Buffer Padding for Extended Collection and Chunking ---
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    CHUNK_SIZE = config['NUM_STEPS'] * config['NUM_ENVS']
    NUM_CHUNKS = (EXTENDED_CAPACITY + CHUNK_SIZE - 1) // CHUNK_SIZE
    PADDED_CAPACITY = NUM_CHUNKS * CHUNK_SIZE
    
    k_base = config.get("RND_FEATURES", 128)
    k_lstd = k_base + 2
    
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
            traces = traces.reshape(-1, k_lstd).astype(jnp.float32)
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
            
    def solve_lstd_buffer(buffer_state: LSTDBufferState, Sigma_inv, lstd_state, get_phi_lstd, config):
        """Solves LSTD using dynamic projection to high-dimensional space."""
        chunked_phi_base = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_next_phi_base = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_traces_lstd = buffer_state.traces.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = (jnp.arange(PADDED_CAPACITY) < buffer_state.size)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]
        k_lstd = chunked_traces_lstd.shape[-1]

        def process_chunk(carry, chunk_data):
            A_acc, b_acc = carry
            phi_b, next_phi_b, traces_lstd, term, absorb, mask = chunk_data
            
            # Reward uses base features
            next_rho = get_scale_free_bonus(Sigma_inv, next_phi_b)

            # LSTD system uses LSTD sketched features
            phi_lstd = get_phi_lstd(phi_b)
            next_phi_lstd = get_phi_lstd(next_phi_b)

            delta_Phi = phi_lstd - gamma_i * (1 - term) * next_phi_lstd
            A_batch = jnp.einsum("ni, nj -> ij", traces_lstd, delta_Phi)
            b_batch = jnp.einsum("ni, n -> i", traces_lstd, next_rho * mask.squeeze())
            
            abs_features = next_phi_lstd * absorb
            abs_traces = abs_features 
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_traces, abs_features)
            b_abs = jnp.einsum("ni, n -> i", abs_traces, next_rho * absorb.squeeze() * mask.squeeze())
            
            return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs), None

        init_A = jnp.zeros((k_lstd, k_lstd))
        init_b = jnp.zeros(k_lstd)
        
        (final_A, final_b), _ = jax.lax.scan(
            process_chunk, 
            (init_A, init_b), 
            (chunked_phi_base, chunked_next_phi_base, chunked_traces_lstd, chunked_terminals, chunked_absorb, chunked_mask)
        )
        
        # --- Bayesian Optimistic Prior (Diagonal) ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        new_phi_diag_counts = lstd_state["phi_diag_counts"]
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts)
        Lambda_mat = jnp.diag(lambda_k)
        prior_b = lambda_k * lstd_state["V_max"]
        
        reg = jnp.eye(k_lstd) * config["A_REGULARIZATION_PER_STEP"]
        A_view = final_A + Lambda_mat + reg
        b_view = final_b + prior_b
        w_i = jnp.linalg.pinv(A_view, rtol=1e-8) @ b_view
        
        # tracking:
        singular_values = jnp.linalg.svd(A_view, compute_uv=False)
        cond_number = singular_values[0] / (singular_values[-1] + 1e-8)
        
        return {
            "w": w_i,
            "Beta": lstd_state["Beta"],
            "V_max": lstd_state["V_max"],
            "phi_diag_counts": new_phi_diag_counts,
            "cond_number": cond_number
        }
        
    def evict_buffer(buffer_state: LSTDBufferState, get_phi_lstd, config, rng):
        """Computes IV Trace Leverage scores and evicts the lowest-scoring points iteratively."""
        size = buffer_state.size
        phi_base = buffer_state.features
        next_phi_base = buffer_state.next_features
        traces = buffer_state.traces
        terminals = buffer_state.terminals

        phi = get_phi_lstd(phi_base)
        next_phi = get_phi_lstd(next_phi_base)

        # Dynamically grab the dimension for the Ridge penalty
        k_val = traces.shape[-1]
        
        # --- STATIC COMPILATION FIX ---
        # Calculate exactly how many drops are needed mathematically as pure Python ints.
        # We know the buffer always overshoots by exactly `batch_size` when it gets full.
        static_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        percent_fifo = config.get("PERCENT_FIFO", 0.2)
        NUM_CUTS = config.get("EVICTION_CUTS", 4) 
        
        static_fifo_drops = int(static_batch_size * percent_fifo)
        static_prb_drops = static_batch_size - static_fifo_drops
        static_drops_per_cut = static_prb_drops // NUM_CUTS
        
        # Dynamic check: Are we actually full yet?
        buffer_is_full = size > BUFFER_CAPACITY
        
        # --- PHASE 1: FIFO Masking ---
        indices = jnp.arange(PADDED_CAPACITY)
        
        # Valid data only goes up to the current size
        valid_mask = indices < size
        
        # If the buffer is full, mark the oldest 'static_fifo_drops' as False to kill them immediately
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))
            
        Z_all = traces
        X_all = phi - config["GAMMA_i"] * (1 - terminals) * next_phi
        # Pre-reshape the matrices into chunks outside the loop
        Z_chunks = Z_all.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        X_chunks = X_all.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        
        def cut_step(carry, step_idx):
            mask_curr = carry
            mask_chunks = mask_curr.reshape(NUM_CHUNKS, CHUNK_SIZE)
            
            # 1. Chunked A_curr computation (Avoids materializing massive valid_Z)
            def A_chunk(carry_A, chunk_data):
                z_c, x_c, m_c = chunk_data
                # Multiply mask into one operand, saving ~1.6GB of VRAM
                A_c = jnp.einsum("ni, nj -> ij", z_c, x_c * m_c[:, None])
                return carry_A + A_c, None
                
            A_curr, _ = jax.lax.scan(A_chunk, jnp.zeros((k_val, k_val)), (Z_chunks, X_chunks, mask_chunks))
            A_curr += jnp.eye(k_val) * config.get("A_REGULARIZATION_PER_STEP", 1e-3)
            A_inv_curr = jnp.linalg.pinv(A_curr, rtol=1e-8)
            
            # 2. Chunked Score computation
            def score_chunk(carry_unused, chunk_data):
                z_c, x_c = chunk_data
                u_c = z_c @ A_inv_curr.T
                v_c = x_c @ A_inv_curr
                w_c = v_c @ A_inv_curr.T
                
                c = 1.0 - jnp.sum(x_c * u_c, axis=-1)
                c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
                
                u_norm = jnp.sum(u_c * u_c, axis=-1)
                v_norm = jnp.sum(v_c * v_c, axis=-1)
                cross = jnp.sum(u_c * w_c, axis=-1)
                
                s_c = (2.0 * cross / c) + (u_norm * v_norm) / (c * c)
                return None, s_c
                
            _, chunked_scores = jax.lax.scan(score_chunk, None, (Z_chunks, X_chunks))
            scores = chunked_scores.flatten()
            
            # Stochastic Sampling (Gumbel-Max Trick)
            drop_logits = -scores / config.get("STOCHASTIC_TEMP", 1.0)
            drop_logits = jnp.where(mask_curr, drop_logits, -jnp.inf)
            
            rng_key = jax.random.fold_in(rng, step_idx) 
            gumbel_noise = jax.random.gumbel(rng_key, drop_logits.shape)
            noisy_logits = drop_logits + gumbel_noise
            
            _, drop_indices = jax.lax.top_k(noisy_logits, static_drops_per_cut)
            
            mask_next_candidate = mask_curr.at[drop_indices].set(False)
            mask_next = jnp.where(buffer_is_full, mask_next_candidate, mask_curr)
            
            return mask_next, None

        # Run the C-cut loop (statically defined)
        final_mask, _ = jax.lax.scan(cut_step, initial_mask, jnp.arange(NUM_CUTS))
        
        # --- PHASE 3: Compaction & Remainder Trimming ---
        # selection_scores ensures surviving True masks = ~1.0, False masks = ~0.0.
        # The tie-breaker (+ indices) prefers keeping NEWER items when resolving remainders.
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, BUFFER_CAPACITY)
        
        new_traces = jnp.zeros_like(buffer_state.traces).at[:BUFFER_CAPACITY].set(buffer_state.traces[keep_indices])
        new_features = jnp.zeros_like(buffer_state.features).at[:BUFFER_CAPACITY].set(buffer_state.features[keep_indices])
        new_next_features = jnp.zeros_like(buffer_state.next_features).at[:BUFFER_CAPACITY].set(buffer_state.next_features[keep_indices])
        new_terminals = jnp.zeros_like(buffer_state.terminals).at[:BUFFER_CAPACITY].set(buffer_state.terminals[keep_indices])
        new_absorb_masks = jnp.zeros_like(buffer_state.absorb_masks).at[:BUFFER_CAPACITY].set(buffer_state.absorb_masks[keep_indices])
        
        new_size = jnp.minimum(size, BUFFER_CAPACITY)
        
        return LSTDBufferState(
            traces=new_traces, features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=new_size
        )

    V_max = 1.0 / (1 - config["GAMMA_i"]) # maximum intrinsic values
    
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k_lstd)

    def train(rng):
        initial_lstd_state = {
                    "w": jnp.zeros(k_lstd), 
                    "Beta": config["BONUS_SCALE"],
                    "V_max": V_max,
                    "phi_diag_counts": jnp.zeros(k_lstd),
                    "cond_number": 0.0
                }
            
        # Buffer initialized at PADDED_CAPACITY to support safe extended processing overhead
        initial_buffer_state = LSTDBufferState(
            traces=jnp.zeros((PADDED_CAPACITY, k_lstd)),
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
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], bias = False, k=k_base
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], bias = False, k=k_base
        )
        
        def get_phi_map_lstd(phi_base, Sigma_inv):
            """
            Maps base features to LSTD features by appending Bias and current Novelty (rho).
            """
            # 1. Base processing (ReLU + Norm)
            # phi_relu = nn.relu(phi_base)
            # norm = jnp.linalg.norm(phi_relu, axis=-1, keepdims=True)
            # phi_norm = phi_relu / jnp.maximum(norm, 1e-8)
            
            # 2. Compute Intrinsic Novelty Feature
            # get_scale_free_bonus uses the raw phi_base as per your original logic
            rho = get_scale_free_bonus(Sigma_inv, phi_base)
            
            # 3. Hierarchical Concatenation: [Bias, Base, Rho]
            bias = jnp.ones(phi_base.shape[:-1] + (1,))
            
            # Note: rho is a scalar per state, so we add a feature dimension [..., None]
            return jnp.concatenate([bias, phi_base, rho[..., None]], axis=-1)

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
            phi_base = batch_get_features(traj_batch.obs)
            next_phi_base = batch_get_features(traj_batch.next_obs)
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            # traces_base = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            
            # --- GLOBAL COVARIANCE UPDATE (Pure Accumulation) --
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_base))
            # Sigma_inv = jnp.linalg.solve(sigma_state["S"], jnp.eye(k_lstd))
            int_rew_from_features = lambda feats: get_scale_free_bonus(Sigma_inv, feats)
            get_phi_lstd = lambda x: get_phi_map_lstd(x, Sigma_inv)

            # --- 2. SOLVE LSTD OVER EXTENDED BUFFER ---
            phi, next_phi = jax.tree.map(lambda x: get_phi_lstd(x), (phi_base, next_phi_base))
            traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            
            buffer_state = update_buffer(buffer_state, traces, phi_base, next_phi_base, terminals, absorb_masks)
            
            lstd_state = update_phi_precision(lstd_state, phi, next_phi, traj_batch.done)
            lstd_state = solve_lstd_buffer(buffer_state, Sigma_inv, lstd_state, get_phi_lstd, config)

            # --- 3 & 4. SCORE AND EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = evict_buffer(buffer_state,get_phi_lstd, config, prb_rng)

            # --- BATCH INTRINSIC VALUES FOR GAE ---
            batch_curr_rho = get_scale_free_bonus(Sigma_inv, phi_base)
            batch_next_rho = get_scale_free_bonus(Sigma_inv, next_phi_base)

            rho_scale = lstd_state["Beta"]
            v_i = phi @ lstd_state["w"] * rho_scale
            next_v_i = next_phi @ lstd_state["w"] * rho_scale

            v_i, next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, lstd_state['V_max']) , (v_i, next_v_i) )
            
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
                    'condition_number': lstd_state['cond_number']
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
                    phi_base = batch_get_features(obs)
                    phi_lstd = get_phi_lstd(phi_base)
                    return  phi_lstd @ lstd_state["w"] * rho_scale

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


