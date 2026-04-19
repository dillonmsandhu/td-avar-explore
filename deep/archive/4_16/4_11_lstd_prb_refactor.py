from core.imports import *
import core.helpers as helpers
import core.networks as networks
from flax import struct
from typing import Any

SAVE_DIR = "4_11_lstd_distill"

# --- STATE DATACLASSES ---

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

@struct.dataclass
class LSTDBufferState:
    traces: jnp.ndarray
    features: jnp.ndarray
    next_features: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    size: jnp.ndarray

@struct.dataclass
class LSTDState:
    w: jnp.ndarray
    Beta: jnp.ndarray
    V_max: jnp.ndarray
    phi_diag_counts: jnp.ndarray

@struct.dataclass
class RunnerState:
    train_state: Any
    lstd_state: LSTDState
    sigma_state: Any  # Dict for helpers compatibility
    buffer_state: LSTDBufferState
    rnd_state: Any
    env_state: Any
    last_obs: jnp.ndarray
    rng: Any
    idx: int

# --- MAIN TRAIN FUNCTION ---

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    CHUNK_SIZE = 100_000 + batch_size
    NUM_CHUNKS = (EXTENDED_CAPACITY + CHUNK_SIZE - 1) // CHUNK_SIZE
    PADDED_CAPACITY = NUM_CHUNKS * CHUNK_SIZE
    
    k_lstd = config.get("RND_FEATURES", 128)
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    # ---------------------------------------------------------
    # PURE FUNCTIONS: MATH & BUFFER
    # ---------------------------------------------------------
    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    
    def update_phi_precision(lstd_state: LSTDState, features, next_features, done):
        absorb_mask = jnp.where(is_absorbing, done, 0)
        batch_phi_prec = jnp.sum(features**2, axis=(0, 1)) 
        absorbing_features = next_features * absorb_mask[..., None]
        abs_phi_prec = jnp.sum(absorbing_features**2, axis=(0, 1)) 
        new_counts = batch_phi_prec + abs_phi_prec
        return lstd_state.replace(phi_diag_counts=lstd_state.phi_diag_counts + new_counts)
    
    def update_buffer(buffer_state: LSTDBufferState, traces, features, next_features, terminals, absorb_masks):
        traces = traces.reshape(-1, k_lstd).astype(jnp.float32)
        features = features.reshape(-1, k_lstd).astype(jnp.float32)
        next_features = next_features.reshape(-1, k_lstd).astype(jnp.float32)
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

    def solve_lstd_buffer(buffer_state: LSTDBufferState, S_inv, lstd_state: LSTDState, config):
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
            next_rho = get_scale_free_bonus(S_inv, next_phi)
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
            process_chunk, (init_A, init_b), 
            (chunked_phi, chunked_next_phi, chunked_traces, chunked_terminals, chunked_absorb, chunked_mask)
        )
        
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + lstd_state.phi_diag_counts)
        Lambda_mat = jnp.diag(lambda_k)
        prior_b = lambda_k * lstd_state.V_max
        
        reg = jnp.eye(k_lstd) * config["A_REGULARIZATION_PER_STEP"]
        A_view = final_A + Lambda_mat + reg
        b_view = final_b + prior_b
        w_i = jnp.linalg.pinv(A_view, rtol=1e-8) @ b_view
        
        return lstd_state.replace(w=w_i)
        
    def evict_buffer(buffer_state: LSTDBufferState, S_inv, config, rng):
        size = buffer_state.size
        phi = buffer_state.features
        next_phi = buffer_state.next_features
        traces = buffer_state.traces
        terminals = buffer_state.terminals

        k_val = traces.shape[-1]
        
        static_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        percent_fifo = config.get("PERCENT_FIFO", 0.2)
        NUM_CUTS = config.get("EVICTION_CUTS", 4) 
        
        static_fifo_drops = int(static_batch_size * percent_fifo)
        static_prb_drops = static_batch_size - static_fifo_drops
        static_drops_per_cut = static_prb_drops // NUM_CUTS
        
        buffer_is_full = size > BUFFER_CAPACITY
        
        indices = jnp.arange(PADDED_CAPACITY)
        valid_mask = indices < size
        
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))
            
        Z_all = traces
        X_all = phi - config["GAMMA_i"] * (1 - terminals) * next_phi
        
        def cut_step(carry, step_idx):
            mask_curr = carry
            valid_Z = Z_all * mask_curr[:, None]
            valid_X = X_all * mask_curr[:, None]
            
            A_curr = jnp.einsum("ni, nj -> ij", valid_Z, valid_X) 
            A_curr += jnp.eye(k_val) * config.get("A_REGULARIZATION_PER_STEP", 1e-3)
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
        
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, BUFFER_CAPACITY)
        
        new_traces = jnp.zeros_like(buffer_state.traces).at[:BUFFER_CAPACITY].set(buffer_state.traces[keep_indices])
        new_features = jnp.zeros_like(buffer_state.features).at[:BUFFER_CAPACITY].set(buffer_state.features[keep_indices])
        new_next_features = jnp.zeros_like(buffer_state.next_features).at[:BUFFER_CAPACITY].set(buffer_state.next_features[keep_indices])
        new_terminals = jnp.zeros_like(buffer_state.terminals).at[:BUFFER_CAPACITY].set(buffer_state.terminals[keep_indices])
        new_absorb_masks = jnp.zeros_like(buffer_state.absorb_masks).at[:BUFFER_CAPACITY].set(buffer_state.absorb_masks[keep_indices])
        
        return LSTDBufferState(
            traces=new_traces, features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=jnp.minimum(size, BUFFER_CAPACITY)
        )

    # ---------------------------------------------------------
    # PURE FUNCTIONS: RL LOOP COMPONENTS
    # ---------------------------------------------------------
    def rollout_trajectories(runner_state: RunnerState, network):
        def _env_step(env_scan_state, unused):
            ts, rs, es, last_obs, rng = env_scan_state

            rng, _rng = jax.random.split(rng)
            pi, value, i_val = network.apply(ts.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, es, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, es, action, env_params
            )
            true_next_obs = info["real_next_obs"]
            
            _, next_val, next_i_val = network.apply(ts.params, true_next_obs)
            intrinsic_reward = jnp.zeros_like(reward)

            transition = Transition(
                done, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info
            )
            return (ts, rs, es, obsv, rng), transition

        env_step_state = (runner_state.train_state, runner_state.rnd_state, runner_state.env_state, runner_state.last_obs, runner_state.rng)
        (ts, rs, es, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])
        
        return runner_state.replace(train_state=ts, rnd_state=rs, env_state=es, last_obs=last_obs, rng=rng), traj_batch

    def process_intrinsic_data(traj_batch, buffer_state: LSTDBufferState, lstd_state: LSTDState, sigma_state, batch_get_features, idx, rng):
        tot_N_samples = idx * config['NUM_STEPS'] * config['NUM_ENVS']

        phi = batch_get_features(traj_batch.obs)
        next_phi = batch_get_features(traj_batch.next_obs)
        terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
        absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
        traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
        
        # 1. Update Buffer
        new_buffer_state = update_buffer(buffer_state, traces, phi, next_phi, terminals, absorb_masks)
        
        # 2. Covariance Calculation
        eps = jnp.maximum(1e-8 / tot_N_samples, 1e-12)
        cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]+ eps * jnp.eye(k_lstd))
        S_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_lstd))
        rho_scale = config['BONUS_SCALE'] / jnp.sqrt(tot_N_samples)

        # 3. Solve LSTD
        new_lstd_state = update_phi_precision(lstd_state, phi, next_phi, traj_batch.done)
        new_lstd_state = solve_lstd_buffer(new_buffer_state, S_inv, new_lstd_state, config)

        # 4. Evict Buffer
        rng, prb_rng = jax.random.split(rng)
        new_buffer_state = evict_buffer(new_buffer_state, S_inv, config, prb_rng)

        # 5. Dual GAE Imputation
        batch_curr_rho = get_scale_free_bonus(S_inv, phi)
        batch_next_rho = get_scale_free_bonus(S_inv, next_phi)
        intrinsic_reward = batch_next_rho * rho_scale

        lstd_v_i = phi @ new_lstd_state.w * rho_scale
        lstd_next_v_i = next_phi @ new_lstd_state.w * rho_scale
        lstd_v_i, lstd_next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, new_lstd_state.V_max), (lstd_v_i, lstd_next_v_i))
        
        exact_terminal_i_val = (batch_curr_rho * rho_scale) / (1.0 - config["GAMMA_i"])
        
        fixed_lstd_next_i_val = jnp.where(
            jnp.logical_and(traj_batch.done, is_absorbing), exact_terminal_i_val, lstd_next_v_i
        )
        
        fixed_net_next_i_val = jnp.where(
            jnp.logical_and(traj_batch.done, is_absorbing), exact_terminal_i_val, traj_batch.next_i_val
        )

        lstd_traj = traj_batch._replace(
            i_value=lstd_v_i, next_i_val=fixed_lstd_next_i_val, intrinsic_reward=intrinsic_reward
        )
        net_traj = traj_batch._replace(
            next_i_val=fixed_net_next_i_val, intrinsic_reward=intrinsic_reward
        )

        return net_traj, lstd_traj, new_buffer_state, new_lstd_state, rng, rho_scale, S_inv, next_phi

    def compute_advantages(net_traj, lstd_traj, config):
        _, lstd_targets = helpers.calculate_gae(
            lstd_traj, config["GAMMA"], config["GAE_LAMBDA"],
            is_episodic=is_episodic, is_absorbing=is_absorbing,
            γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
        )
        distilled_i_target = lstd_targets[1]

        gaes, targets = helpers.calculate_gae(
            net_traj, config["GAMMA"], config["GAE_LAMBDA"],
            is_episodic=is_episodic, is_absorbing=is_absorbing,
            γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
        )
        
        advantages = gaes[0] + gaes[1]
        combined_targets = (targets[0], distilled_i_target)
        return advantages, combined_targets, gaes, targets

    def update_networks(train_state, network, traj_batch, advantages, combined_targets, rng):
        def _update_epoch(update_state, unused):
            def _update_minbatch(ts, batch_info):
                tb, adv, tgts = batch_info
                grad_fn = jax.value_and_grad(helpers._loss_fn_intrinsic_v, has_aux=True)
                (total_loss, aux_losses), grads = grad_fn(
                    ts.params, network, tb, adv, tgts, config
                )
                ts = ts.apply_gradients(grads=grads)
                return ts, (total_loss, aux_losses)

            ts, tb, adv, tgts, current_rng = update_state
            current_rng, _rng = jax.random.split(current_rng)
            minibatches = helpers.shuffle_and_batch(_rng, (tb, adv, tgts), config["NUM_MINIBATCHES"])
            ts, losses = jax.lax.scan(_update_minbatch, ts, minibatches)
            return (ts, tb, adv, tgts, current_rng), losses

        initial_update_state = (train_state, traj_batch, advantages, combined_targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
        return update_state[0], loss_info, update_state[4]

    def build_metrics(traj_batch, loss_info, next_phi, gaes, targets, lstd_state, rho_scale, S_inv, network, train_state):
        total_loss_info, aux_loss_info = loss_info
        metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
        
        metric.update({
            "ppo_loss": total_loss_info.mean(),
            "i_value_loss": aux_loss_info[0].mean(),
            "e_value_loss": aux_loss_info[1].mean(),
            "pi_loss": aux_loss_info[2].mean(),
            "entropy": aux_loss_info[3].mean(),
            "feat_norm": jnp.linalg.norm(next_phi, axis=-1).mean(),
            "bonus_mean": gaes[1].mean(),
            "bonus_std": gaes[1].std(),
            "bonus_max": gaes[1].max(),
            "lambda_ret_mean": targets[0].mean(),
            "lambda_ret_std": targets[0].std(),
            "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
            "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
            "mean_rew": traj_batch.reward.mean(),
            "lambda_k": lstd_state.phi_diag_counts,
            "beta": lstd_state.Beta,
            "rho_scale": rho_scale,
        })

        if evaluator is None: 
            metric.update({
                "vi_pred": traj_batch.i_value.mean(),
                "v_e_pred": traj_batch.value.mean(),
            })
        else:
            def int_rew_from_state(s): 
                return get_scale_free_bonus(S_inv, batch_get_features(s)) * rho_scale

            def get_vi(obs):
                _, _, i_val = network.apply(train_state.params, obs)
                return i_val

            metric = helpers.add_values_to_metric(
                config, metric, int_rew_from_state, evaluator, lstd_state.Beta,
                network, train_state, traj_batch, get_vi, rho_scale=rho_scale
            )
        return metric

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------
    V_max = (1.0 / (1 - config["GAMMA_i"])) / (jnp.sqrt(k_lstd) if config.get("NORMALIZE_FEATURES") else 1.0)

    def train(rng):
        initial_lstd_state = LSTDState(
            w=jnp.zeros(k_lstd), 
            Beta=config["BONUS_SCALE"],
            V_max=V_max,
            phi_diag_counts=jnp.zeros(k_lstd),
        )
            
        initial_buffer_state = LSTDBufferState(
            traces=jnp.zeros((PADDED_CAPACITY, k_lstd)),
            features=jnp.zeros((PADDED_CAPACITY, k_lstd)),
            next_features=jnp.zeros((PADDED_CAPACITY, k_lstd)),
            terminals=jnp.zeros((PADDED_CAPACITY, 1)),
            absorb_masks=jnp.zeros((PADDED_CAPACITY, 1)),
            size=jnp.array(0, dtype=jnp.int32)
        )
        
        initial_sigma_state = {"S": jnp.eye(k_lstd, dtype=jnp.float64), "N": 1}

        rnd_rng, target_rng, rng = jax.random.split(rng, 3)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_lstd
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_lstd
        )
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=3)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        # ---------------------------------------------------------
        # THE CLEAN PPO LOOP
        # ---------------------------------------------------------
        def _update_step(runner_state: RunnerState, unused):
            
            # 1. ROLLOUT
            runner_state, traj_batch = rollout_trajectories(runner_state, network)

            # 2. INTRINSIC MATH (LSTD, Buffer, Imputation)
            (net_traj, lstd_traj, new_buffer, new_lstd, new_rng, 
             rho_scale, S_inv, next_phi) = process_intrinsic_data(
                traj_batch, runner_state.buffer_state, runner_state.lstd_state, 
                runner_state.sigma_state, batch_get_features, runner_state.idx, runner_state.rng
            )
            runner_state = runner_state.replace(buffer_state=new_buffer, lstd_state=new_lstd, rng=new_rng)

            # 3. ADVANTAGES
            advantages, combined_targets, gaes, targets = compute_advantages(net_traj, lstd_traj, config)

            # 4. PPO UPDATE
            new_train_state, loss_info, new_rng = update_networks(
                runner_state.train_state, network, net_traj, advantages, combined_targets, runner_state.rng
            )
            runner_state = runner_state.replace(train_state=new_train_state, rng=new_rng)

            # 5. COVARIANCE UPDATE
            new_sigma_state = helpers.update_cov(net_traj, runner_state.sigma_state, batch_get_features)
            runner_state = runner_state.replace(sigma_state=new_sigma_state)

            # 6. METRICS
            metric = build_metrics(
                net_traj, loss_info, next_phi, gaes, targets, 
                runner_state.lstd_state, rho_scale, S_inv, network, runner_state.train_state
            )

            # Increment Step
            runner_state = runner_state.replace(idx=runner_state.idx + 1)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        initial_runner_state = RunnerState(
            train_state=train_state, lstd_state=initial_lstd_state, 
            sigma_state=initial_sigma_state, buffer_state=initial_buffer_state, 
            rnd_state=rnd_state, env_state=env_state, last_obs=obsv, 
            rng=_rng, idx=1
        )
        
        runner_state, metrics = jax.lax.scan(_update_step, initial_runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)