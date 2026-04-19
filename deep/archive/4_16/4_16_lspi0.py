# does not use traces
from core.imports import *
import core.helpers as helpers
import core.networks as networks
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "4_16_lspi0"

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

class LSPIBufferState(NamedTuple):
    features: jnp.ndarray      # shape: (..., dim_kA)
    next_features: jnp.ndarray # shape: (..., k_lstd)
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    size: jnp.ndarray          # Extended buffer compaction sizing

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    # --- Buffer Padding for Extended Collection and Chunking ---
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    CHUNK_SIZE = 100_000 + batch_size
    NUM_CHUNKS = (EXTENDED_CAPACITY + CHUNK_SIZE - 1) // CHUNK_SIZE
    PADDED_CAPACITY = NUM_CHUNKS * CHUNK_SIZE
    
    k_lstd = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    dim_kA = k_lstd * n_actions
    
    evaluator = helpers.initialize_evaluator(config)
    
    if config.get('SCHEDULE_BETA', False):
        # goes up until peak and then linearly decays to 0.
        beta_sch = helpers.make_triangle_schedule(total_updates = config['NUM_UPDATES'], max_beta=config['BONUS_SCALE'], peak_at=0.01) 
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    def get_scale_free_bonus(S_inv, features):
        """bonus = sqrt(x^T Σ^{-1} x)"""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

    def expand_to_sa_features(phi_s, n_actions, taken_actions):
        one_hots = jax.nn.one_hot(taken_actions, n_actions)  
        phi_sa_unflattened = phi_s[..., None, :] * one_hots[..., :, None]
        return phi_sa_unflattened.reshape(*phi_s.shape[:-1], dim_kA)

    def expected_next_sa_features(next_phi, Pi):
        expected_next_sa = next_phi[..., None, :] * Pi[..., :, None]
        return expected_next_sa.reshape(*next_phi.shape[:-1], dim_kA)
    
    def update_buffer(buffer_state: LSPIBufferState, features_sa, next_features_s, terminals, absorb_masks):
        """Appends a new batch of transitions to the extended JAX buffer."""
        features_sa = features_sa.reshape(-1, dim_kA).astype(jnp.float32)
        next_features_s = next_features_s.reshape(-1, k_lstd).astype(jnp.float32)
        terminals = terminals.reshape(-1, 1).astype(jnp.float32)
        absorb_masks = absorb_masks.reshape(-1, 1).astype(jnp.float32)
        
        B = features_sa.shape[0]
        start_idx = buffer_state.size
        
        new_features = jax.lax.dynamic_update_slice(buffer_state.features, features_sa, (start_idx, 0))
        new_next_features = jax.lax.dynamic_update_slice(buffer_state.next_features, next_features_s, (start_idx, 0))
        new_terminals = jax.lax.dynamic_update_slice(buffer_state.terminals, terminals, (start_idx, 0))
        new_absorb_masks = jax.lax.dynamic_update_slice(buffer_state.absorb_masks, absorb_masks, (start_idx, 0))
        
        return LSPIBufferState(
            features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=start_idx + B
        )

    def solve_lspi_buffer(buffer_state: LSPIBufferState, Sigma_inv, lstd_state, config):
        """Solves LSPI over the entire extended buffer using a memory-safe chunked scan."""
        chunked_phi_sa = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_next_phi_s = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = (jnp.arange(PADDED_CAPACITY) < buffer_state.size)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]

        def lspi_step(w_current, _):
            def process_chunk(carry, chunk_data):
                A_acc, b_acc = carry
                c_phi_sa, c_next_phi_s, c_term, c_absorb, c_mask = chunk_data
                
                next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi_s)
                
                # 1. Greedy Policy Evaluation
                w_reshaped = w_current.reshape(n_actions, k_lstd)
                Q_next = jnp.einsum("...k, ak -> ...a", c_next_phi_s, w_reshaped)
                greedy_actions = jnp.argmax(Q_next, axis=-1)
                Pi_greedy = jax.nn.one_hot(greedy_actions, n_actions)
                
                PΠφ = expected_next_sa_features(c_next_phi_s, Pi_greedy)
                
                # 2. Construction of A
                c_traces_masked = c_phi_sa * c_mask
                
                S_chunk = jnp.einsum("ni, nj -> ij", c_traces_masked, c_phi_sa)
                γPΠφ = gamma_i * (1 - c_term) * PΠφ
                γPΠΦ_chunk = jnp.einsum("ni, nj -> ij", c_traces_masked, γPΠφ)
                A_std = S_chunk - γPΠΦ_chunk
                
                abs_features = PΠφ * c_absorb
                abs_traces = c_phi_sa * c_absorb
                A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_traces * c_mask, abs_features)
                
                A_batch = A_std + A_abs
                
                # 3. Construction of b
                b_std = jnp.einsum("ni, n -> i", c_traces_masked, next_rho)
                b_abs = jnp.einsum("ni, n -> i", abs_traces * c_mask, next_rho)
                b_batch = b_std + b_abs
                
                return (A_acc + A_batch, b_acc + b_batch), None

            init_A = jnp.zeros((dim_kA, dim_kA))
            init_b = jnp.zeros(dim_kA)
            
            (final_A, final_b), _ = jax.lax.scan(
                process_chunk, 
                (init_A, init_b), 
                (chunked_phi_sa, chunked_next_phi_s, chunked_terminals, chunked_absorb, chunked_mask)
            )
            
            reg = jnp.eye(dim_kA) * config["LSTD_L2_REG"] * buffer_state.size
            A_view = final_A + reg
            w_new = jnp.linalg.solve(A_view, final_b)
            
            return w_new, None

        w_init = lstd_state["w"]
        w_final, _ = jax.lax.scan(lspi_step, w_init, None, length=config.get("LSPI_NUM_ITERS", 3))
        
        return {
            "w": w_final,
            "V_max": lstd_state["V_max"],
        }
        
    def evict_buffer(buffer_state: LSPIBufferState, lstd_state, config, rng):
        """Computes IV Trace Leverage scores using the NEW optimal policy."""
        size = buffer_state.size
        phi_sa = buffer_state.features
        next_phi_s = buffer_state.next_features
        terminals = buffer_state.terminals

        k_val = phi_sa.shape[-1]
        
        static_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        percent_fifo = config.get("PERCENT_FIFO", 0.2)
        NUM_CUTS = config.get("EVICTION_CUTS", 4) 
        
        static_fifo_drops = int(static_batch_size * percent_fifo)
        static_prb_drops = static_batch_size - static_fifo_drops
        static_drops_per_cut = static_prb_drops // NUM_CUTS
        
        buffer_is_full = size > BUFFER_CAPACITY
        
        # --- PHASE 1: FIFO Masking ---
        indices = jnp.arange(PADDED_CAPACITY)
        valid_mask = indices < size
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))
            
        # --- PHASE 2: Stochastic Leverage Cuts ---
        # Evaluate greedy policy using the newly updated weights
        w_reshaped = lstd_state["w"].reshape(n_actions, k_lstd)
        Q_next = jnp.einsum("...k, ak -> ...a", next_phi_s, w_reshaped)
        greedy_actions = jnp.argmax(Q_next, axis=-1)
        Pi_greedy = jax.nn.one_hot(greedy_actions, n_actions)
        
        PΠφ = expected_next_sa_features(next_phi_s, Pi_greedy)
        
        Z_all = phi_sa
        X_all = phi_sa - config["GAMMA_i"] * (1 - terminals) * PΠφ
        
        def cut_step(carry, step_idx):
            mask_curr = carry
            
            valid_Z = Z_all * mask_curr[:, None]
            valid_X = X_all * mask_curr[:, None]
            
            A_curr = jnp.einsum("ni, nj -> ij", valid_Z, valid_X) 
            A_curr += jnp.eye(k_val) * config.get("LSTD_L2_REG", 1e-3) * size
            A_inv_curr = jnp.linalg.pinv(A_curr, rtol=1e-8) 
            
            U = Z_all @ A_inv_curr.T
            V = X_all @ A_inv_curr
            W = V @ A_inv_curr.T
            
            c = 1.0 - jnp.sum(X_all * U, axis=-1)
            c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
            
            u_norm = jnp.sum(U * U, axis=-1)
            v_norm = jnp.sum(V * V, axis=-1)
            cross = jnp.sum(U * W, axis=-1)
            
            scores = (2.0 * cross / c) + (u_norm * v_norm) / (c * c)
            drop_logits = -scores / config.get("STOCHASTIC_TEMP", 1.0)
            drop_logits = jnp.where(mask_curr, drop_logits, -jnp.inf)
            
            rng_key = jax.random.fold_in(rng, step_idx) 
            gumbel_noise = jax.random.gumbel(rng_key, drop_logits.shape)
            noisy_logits = drop_logits + gumbel_noise
            
            _, drop_indices = jax.lax.top_k(noisy_logits, static_drops_per_cut)
            mask_next_candidate = mask_curr.at[drop_indices].set(False)
            mask_next = jnp.where(buffer_is_full, mask_next_candidate, mask_curr)
            
            return mask_next, None

        final_mask, _ = jax.lax.scan(cut_step, initial_mask, jnp.arange(NUM_CUTS))
        
        # --- PHASE 3: Compaction ---
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, BUFFER_CAPACITY)
        
        new_features = jnp.zeros_like(buffer_state.features).at[:BUFFER_CAPACITY].set(buffer_state.features[keep_indices])
        new_next_features = jnp.zeros_like(buffer_state.next_features).at[:BUFFER_CAPACITY].set(buffer_state.next_features[keep_indices])
        new_terminals = jnp.zeros_like(buffer_state.terminals).at[:BUFFER_CAPACITY].set(buffer_state.terminals[keep_indices])
        new_absorb_masks = jnp.zeros_like(buffer_state.absorb_masks).at[:BUFFER_CAPACITY].set(buffer_state.absorb_masks[keep_indices])
        
        new_size = jnp.minimum(size, BUFFER_CAPACITY)
        
        return LSPIBufferState(
            features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=new_size
        )

    V_max = 1.0 / (1 - config["GAMMA_i"]) 
    
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k_lstd)

    def train(rng):
        initial_lstd_state = {
                    "w": jnp.zeros(dim_kA), 
                    "V_max": V_max,
                }
            
        initial_buffer_state = LSPIBufferState(
            features=jnp.zeros((PADDED_CAPACITY, dim_kA)),
            next_features=jnp.zeros((PADDED_CAPACITY, k_lstd)),
            terminals=jnp.zeros((PADDED_CAPACITY, 1)),
            absorb_masks=jnp.zeros((PADDED_CAPACITY, 1)),
            size=jnp.array(0, dtype=jnp.int32)
        )
        
        initial_sigma_state = {"S": jnp.eye(k_lstd, dtype=jnp.float64)} 

        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_lstd
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_lstd
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
                true_next_obs = info["real_next_obs"].reshape(last_obs.shape)
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
            
            # --- FEATURE EXTRACTION ---
            phi_s = batch_get_features(traj_batch.obs)
            next_phi_s = batch_get_features(traj_batch.next_obs)
            phi_sa = expand_to_sa_features(phi_s, n_actions, traj_batch.action)
            
            terminals = jnp.where(terminate_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
    

            # --- 1. UPDATE EXTENDED BUFFER ---
            buffer_state = update_buffer(buffer_state, phi_sa, next_phi_s, terminals, absorb_masks)
            
            # --- GLOBAL COVARIANCE UPDATE ---
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) 
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_lstd))
            
            # --- 2. SOLVE LSPI OVER EXTENDED BUFFER ---
            absorb_mask_batch = jnp.where(is_absorbing, traj_batch.done, 0)
            lstd_state = solve_lspi_buffer(buffer_state, Sigma_inv, lstd_state, config)

            # --- 3 & 4. SCORE AND EVICT BUFFER (USING NEW OPTIMAL POLICY) ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = evict_buffer(buffer_state, lstd_state, config, prb_rng)

            # --- BATCH INTRINSIC VALUES FOR GAE ---
            rho = get_scale_free_bonus(Sigma_inv, next_phi_s) # unscaled
            
            w_reshaped = lstd_state["w"].reshape(n_actions, k_lstd)
            Q_curr = jnp.einsum("...k, ak -> ...a", phi_s, w_reshaped)
            v_i = jnp.max(Q_curr, axis=-1)

            Q_next = jnp.einsum("...k, ak -> ...a", next_phi_s, w_reshaped)
            next_v_i = jnp.max(Q_next, axis=-1)

            v_i, next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, 1.0/(1-config['GAMMA_i'])) , (v_i, next_v_i) )
            
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho, next_i_val=next_v_i)

            # --- ADVANTAGE CALCULATION ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"],
                is_episodic=is_episodic, is_absorbing=is_absorbing,
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            rho_scale = beta_sch(idx) # triangle schedule
            advantages = gaes[0] + rho_scale * gaes[1]
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
                    "feat_norm": jnp.linalg.norm(next_phi_s, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                    "mean_rew": traj_batch.reward.mean(),
                    "rho_scale": rho_scale,
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
                    rho = get_scale_free_bonus(Sigma_inv, phi) * rho_scale
                    return rho

                def get_vi(obs):
                    phi = batch_get_features(obs)
                    w_r = lstd_state["w"].reshape(n_actions, k_lstd)
                    Q_v = jnp.einsum("...k, ak -> ...a", phi, w_r)
                    return jnp.max(Q_v, axis=-1) * rho_scale

                metric = helpers.add_values_to_metric(
                    config, metric, int_rew_from_state, evaluator,
                    rho_scale, network, train_state, traj_batch, get_vi,
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