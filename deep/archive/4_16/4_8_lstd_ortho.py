# Covariance-Based Intrinsic Reward, propagated by LSTD over Intrinsic Value Features
# Toggleable Eviction: IV Leverage Scores (Prioritized) OR standard FIFO
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_8_lstd_prb_ortho"

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
    traces: jnp.ndarray          # Stored in k_lstd (Sketched)
    features: jnp.ndarray        # Stored in k_base (Raw RND)
    next_features: jnp.ndarray   # Stored in k_base (Raw RND)
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    size: jnp.ndarray

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
    CHUNK_SIZE = 10_000
    NUM_CHUNKS = (EXTENDED_CAPACITY + CHUNK_SIZE - 1) // CHUNK_SIZE
    PADDED_CAPACITY = NUM_CHUNKS * CHUNK_SIZE
    
    # --- Dimensionality Setup ---
    k_base = config.get("RND_FEATURES", 128)
    k_sketch = config.get("SKETCH_FEATURES", 512) # Safe to bump to 1024 on modern GPUs
    k_lstd = k_sketch + 2 if augment_lstd else k_sketch 
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        """bonus = sqrt(x^T Σ^{-1} x) computed natively in the k_base space."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    
    def update_phi_precision(lstd_state, eval_features, eval_next_features, done):
        """Accumulates diagonal prior strictly in the sketched (k_lstd) space."""
        phi_diag_precision = lstd_state['phi_diag_counts']
        absorb_mask = jnp.where(is_absorbing, done, 0)
        
        batch_phi_prec = jnp.sum(eval_features**2, axis=(0, 1)) 
        absorbing_features = eval_next_features * absorb_mask[..., None]
        abs_phi_prec = jnp.sum(absorbing_features**2, axis=(0, 1)) 
        
        new_counts = batch_phi_prec + abs_phi_prec
        return {**lstd_state, "phi_diag_counts": phi_diag_precision + new_counts}
    
    def update_buffer(buffer_state: LSTDBufferState, traces, features, next_features, terminals, absorb_masks):
        """Appends a new batch of transitions. Notice traces map to k_lstd, features to k_base."""
        traces = traces.reshape(-1, k_lstd).astype(jnp.float32)
        features = features.reshape(-1, k_base).astype(jnp.float32)
        next_features = next_features.reshape(-1, k_base).astype(jnp.float32)
        terminals = terminals.reshape(-1, 1).astype(jnp.float32)
        absorb_masks = absorb_masks.reshape(-1, 1).astype(jnp.float32)
        
        B = features.shape[0]
        start_idx = buffer_state.size
        
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

    def train(rng):
        initial_lstd_state = {
            "w": jnp.zeros(k_lstd), 
            "Beta": config["BONUS_SCALE"],
            "V_max": (1.0 / (1 - config["GAMMA_i"])) / (jnp.sqrt(k_lstd) if config["NORMALIZE_FEATURES"] else 1.0),
            "phi_diag_counts": jnp.zeros(k_lstd),
            "A_view": jnp.eye(k_lstd) 
        }
            
        initial_buffer_state = LSTDBufferState(
            traces=jnp.zeros((PADDED_CAPACITY, k_lstd)),
            features=jnp.zeros((PADDED_CAPACITY, k_base)),
            next_features=jnp.zeros((PADDED_CAPACITY, k_base)),
            terminals=jnp.zeros((PADDED_CAPACITY, 1)),
            absorb_masks=jnp.zeros((PADDED_CAPACITY, 1)),
            size=jnp.array(0, dtype=jnp.int32)
        )
        
        initial_sigma_state = {"S": jnp.eye(k_base)} 

        # --- 1. RND INITIALIZATION ---
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_base
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_base
        )
        batch_get_features = jax.vmap(lambda obs: rnd_net.apply(target_params, obs))

        # --- 2. ORTHOGONAL RANDOM MACLAURIN SKETCH INITIALIZATION ---
        def get_stacked_orthogonal_matrix(rng_key, in_dim, out_dim):
            """Generates a perfectly orthogonal random projection matrix (Stacked ORF)."""
            n_blocks = (out_dim + in_dim - 1) // in_dim
            blocks = []
            for _ in range(n_blocks):
                rng_key, sub_rng = jax.random.split(rng_key)
                G = jax.random.normal(sub_rng, (in_dim, in_dim))
                Q, R = jnp.linalg.qr(G)
                d = jnp.diag(R)
                ph = d / jnp.abs(d)
                Q = Q * ph
                blocks.append(Q * jnp.sqrt(in_dim))
            return jnp.concatenate(blocks, axis=1)[:, :out_dim]

        w1_rng, w2_rng, rng = jax.random.split(rng, 3)
        W1_orf = get_stacked_orthogonal_matrix(w1_rng, k_base, k_sketch)
        W2_orf = get_stacked_orthogonal_matrix(w2_rng, k_base, k_sketch)
        
        def apply_sketch(phi_128):
            """Unbiased degree-2 polynomial expansion using Orthogonal Maclaurin."""
            proj1 = jnp.dot(phi_128, W1_orf)
            proj2 = jnp.dot(phi_128, W2_orf)
            return (proj1 * proj2) / jnp.sqrt(k_sketch)

        # --- 3. ACTOR CRITIC INITIALIZATION (n_heads=2) ---
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, _ = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )

        def solve_lstd_buffer(buffer_state: LSTDBufferState, Sigma_inv, lstd_state, config):
            chunked_phi = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, k_base)
            chunked_next_phi = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, k_base)
            chunked_traces = buffer_state.traces.reshape(NUM_CHUNKS, CHUNK_SIZE, k_lstd)
            chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
            chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
            
            valid_mask = (jnp.arange(PADDED_CAPACITY) < buffer_state.size)[..., None]
            chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
            gamma_i = config["GAMMA_i"]

            def process_chunk(carry, chunk_data):
                A_acc, b_acc = carry
                c_phi, c_next_phi, c_traces, c_term, c_absorb, c_mask = chunk_data
                
                # Compute rho natively in 128D
                curr_rho = get_scale_free_bonus(Sigma_inv, c_phi)
                next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi)
                
                # On-the-fly Sketch to 512D/1024D
                c_sketched = apply_sketch(c_phi)
                c_next_sketched = apply_sketch(c_next_phi)
                
                if augment_lstd:
                    aug_phi = jnp.concatenate([c_sketched, curr_rho[..., None], jnp.ones_like(curr_rho[..., None])], axis=-1)
                    aug_next_phi = jnp.concatenate([c_next_sketched, next_rho[..., None], jnp.ones_like(next_rho[..., None])], axis=-1)
                else:
                    aug_phi = c_sketched
                    aug_next_phi = c_next_sketched
                    
                aug_phi = aug_phi * c_mask
                aug_next_phi = aug_next_phi * c_mask
                aug_traces = c_traces * c_mask
                
                delta_Phi = aug_phi - gamma_i * (1 - c_term) * aug_next_phi
                A_batch = jnp.einsum("ni, nj -> ij", aug_traces, delta_Phi)
                b_batch = jnp.einsum("ni, n -> i", aug_traces, next_rho * c_mask.squeeze())
                
                abs_features = aug_next_phi * c_absorb
                abs_traces = abs_features 
                A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_traces, abs_features)
                b_abs = jnp.einsum("ni, n -> i", abs_traces, next_rho * c_absorb.squeeze() * c_mask.squeeze())
                
                return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs), None

            init_A = jnp.zeros((k_lstd, k_lstd))
            init_b = jnp.zeros(k_lstd)
            
            (final_A, final_b), _ = jax.lax.scan(
                process_chunk, (init_A, init_b), 
                (chunked_phi, chunked_next_phi, chunked_traces, chunked_terminals, chunked_absorb, chunked_mask)
            )
            
            PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
            new_phi_diag_counts = lstd_state["phi_diag_counts"]
            lambda_k = jnp.where(PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts) >= 0.1, 
                                 PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts), 0.0)
            Lambda_mat = jnp.diag(lambda_k)
            prior_b = jnp.diag(Lambda_mat) * lstd_state["V_max"]
            
            reg = jnp.eye(k_lstd) * config["A_REGULARIZATION_PER_STEP"]
            A_view = final_A + Lambda_mat + reg
            final_b = final_b + prior_b
            
            w_i = jnp.linalg.solve(A_view, final_b)
            
            return {**lstd_state, "w": w_i, "phi_diag_counts": new_phi_diag_counts, "A_view": A_view}

        def evict_buffer(buffer_state: LSTDBufferState, A_inv, Sigma_inv, config):
            size = buffer_state.size
            
            if config.get("PRIORITIZED_EVICTION", False):
                valid_mask = jnp.arange(PADDED_CAPACITY) < size
                
                # Reconstruct Sketches on the fly to conserve RAM
                curr_rho = get_scale_free_bonus(Sigma_inv, buffer_state.features)
                next_rho = get_scale_free_bonus(Sigma_inv, buffer_state.next_features)
                sketched_phi = apply_sketch(buffer_state.features)
                sketched_next_phi = apply_sketch(buffer_state.next_features)
                
                if augment_lstd:
                    aug_phi = jnp.concatenate([sketched_phi, curr_rho[..., None], jnp.ones_like(curr_rho[..., None])], axis=-1)
                    aug_next_phi = jnp.concatenate([sketched_next_phi, next_rho[..., None], jnp.ones_like(next_rho[..., None])], axis=-1)
                else:
                    aug_phi = sketched_phi
                    aug_next_phi = sketched_next_phi
                    
                Z = buffer_state.traces
                X = aug_phi - config["GAMMA_i"] * (1 - buffer_state.terminals) * aug_next_phi
                
                # Pure Sketch-Dimensional Sherman-Morrison 
                U = Z @ A_inv.T
                V = X @ A_inv
                W = V @ A_inv.T
                
                c = 1.0 - jnp.sum(X * U, axis=-1)
                c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
                
                u_norm = jnp.sum(U * U, axis=-1)
                v_norm = jnp.sum(V * V, axis=-1)
                cross = jnp.sum(U * W, axis=-1)
                
                scores = (2.0 * cross / c) + (u_norm * v_norm) / (c * c)
                scores = jnp.where(valid_mask, scores, -jnp.inf)
                
                _, keep_indices = jax.lax.top_k(scores, BUFFER_CAPACITY)
            else:
                # Standard FIFO: retain the newest BUFFER_CAPACITY elements
                start_idx = jnp.maximum(0, size - BUFFER_CAPACITY)
                keep_indices = jnp.arange(BUFFER_CAPACITY) + start_idx
            
            return LSTDBufferState(
                traces=jnp.zeros_like(buffer_state.traces).at[:BUFFER_CAPACITY].set(buffer_state.traces[keep_indices]),
                features=jnp.zeros_like(buffer_state.features).at[:BUFFER_CAPACITY].set(buffer_state.features[keep_indices]),
                next_features=jnp.zeros_like(buffer_state.next_features).at[:BUFFER_CAPACITY].set(buffer_state.next_features[keep_indices]),
                terminals=jnp.zeros_like(buffer_state.terminals).at[:BUFFER_CAPACITY].set(buffer_state.terminals[keep_indices]),
                absorb_masks=jnp.zeros_like(buffer_state.absorb_masks).at[:BUFFER_CAPACITY].set(buffer_state.absorb_masks[keep_indices]),
                size=jnp.minimum(size, BUFFER_CAPACITY)
            )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, rng, idx = runner_state

            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state
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

                transition = Transition(
                    done, action, value, next_val, 
                    jnp.zeros_like(reward), jnp.zeros_like(reward), 
                    reward, jnp.zeros_like(reward), log_prob, last_obs, true_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # 1. Base Feature Extraction
            phi_128 = batch_get_features(traj_batch.obs)
            next_phi_128 = batch_get_features(traj_batch.next_obs)
            
            # 2. Covariance Update
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"])
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_base))

            # 3. Batch Evaluation (Sketched + Augmented space)
            batch_curr_rho = get_scale_free_bonus(Sigma_inv, phi_128)
            batch_next_rho = get_scale_free_bonus(Sigma_inv, next_phi_128)
            
            sketched_phi = apply_sketch(phi_128)
            sketched_next_phi = apply_sketch(next_phi_128)
            
            if augment_lstd:
                eval_phi = jnp.concatenate([sketched_phi, batch_curr_rho[..., None], jnp.ones_like(batch_curr_rho[..., None])], axis=-1)
                eval_next_phi = jnp.concatenate([sketched_next_phi, batch_next_rho[..., None], jnp.ones_like(batch_next_rho[..., None])], axis=-1)
            else:
                eval_phi = sketched_phi
                eval_next_phi = sketched_next_phi

            # 4. Traces & Buffer Operations
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            traces = helpers.calculate_traces(traj_batch, eval_phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            
            buffer_state = update_buffer(buffer_state, traces, phi_128, next_phi_128, terminals, absorb_masks)
            lstd_state = update_phi_precision(lstd_state, eval_phi, eval_next_phi, traj_batch.done)
            lstd_state = solve_lstd_buffer(buffer_state, Sigma_inv, lstd_state, config)
            
            A_inv = jnp.linalg.inv(lstd_state["A_view"])
            buffer_state = evict_buffer(buffer_state, A_inv, Sigma_inv, config)

            # 5. Pure LSTD Value Assignment (No Network Distillation)
            rho_scale = lstd_state["Beta"]
            v_i = jnp.clip(eval_phi @ lstd_state["w"], 0, lstd_state['V_max']) * rho_scale
            next_v_i = jnp.clip(eval_next_phi @ lstd_state["w"], 0, lstd_state['V_max']) * rho_scale
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=batch_next_rho * rho_scale, next_i_val=next_v_i)

            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"],
                is_episodic=is_episodic, is_absorbing=is_absorbing,
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(train_state.params, network, traj_batch, advantages, targets, config)
                    return train_state.apply_gradients(grads=grads), total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                minibatches = helpers.shuffle_and_batch(_rng, (traj_batch, advantages, targets), config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            metric.update({
                "ppo_loss": loss_info[0],
                "bonus_mean": gaes[1].mean(),
                "lambda_ret_mean": targets[0].mean(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "mean_rew": traj_batch.reward.mean(),
            })

            if evaluator is not None:
                def int_rew_from_state(s,): 
                    return get_scale_free_bonus(Sigma_inv, batch_get_features(s)) * rho_scale

                def get_vi(obs):
                    base = batch_get_features(obs)
                    sk = apply_sketch(base)
                    rh = get_scale_free_bonus(Sigma_inv, base)
                    if augment_lstd:
                        ev = jnp.concatenate([sk, rh[..., None], jnp.ones_like(rh[..., None])], axis=-1)
                    else:
                        ev = sk
                    return ev @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, lstd_state["Beta"], network, train_state, traj_batch, get_vi)
            else:
                metric.update(
                    {
                        "vi_pred": traj_batch.i_value.mean(),
                        "v_e_pred": traj_batch.value.mean(),
                    }
                )

            return (train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, rng, idx + 1), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)