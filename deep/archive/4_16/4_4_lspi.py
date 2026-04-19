# LSPI with a replay buffer
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_4_cov_lspi_buffer"

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
    traces: jnp.ndarray        # shape: (..., k * n_actions)
    features: jnp.ndarray      # shape: (..., k * n_actions)
    next_features: jnp.ndarray # shape: (..., k) -> State features only
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    ptr: jnp.ndarray
    full: jnp.ndarray

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    k = config.get("RND_FEATURES", 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    dim_kA = k * n_actions
    
    evaluator = helpers.initialize_evaluator(config)

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

    def update_buffer(buffer_state: LSPIBufferState, traces, features_sa, next_features_s, terminals, absorb_masks):
        """Inserts a new batch of transitions into the JAX FIFO ring buffer."""
        traces = traces.reshape(-1, dim_kA)
        features_sa = features_sa.reshape(-1, dim_kA)
        next_features_s = next_features_s.reshape(-1, k)
        terminals = terminals.reshape(-1, 1)
        absorb_masks = absorb_masks.reshape(-1, 1)
        
        B = features_sa.shape[0]
        indices = (buffer_state.ptr + jnp.arange(B)) % BUFFER_CAPACITY
        
        new_traces = buffer_state.traces.at[indices].set(traces)
        new_features = buffer_state.features.at[indices].set(features_sa)
        new_next_features = buffer_state.next_features.at[indices].set(next_features_s)
        new_terminals = buffer_state.terminals.at[indices].set(terminals)
        new_absorb_masks = buffer_state.absorb_masks.at[indices].set(absorb_masks)
        
        new_ptr = (buffer_state.ptr + B) % BUFFER_CAPACITY
        new_full = jnp.logical_or(buffer_state.full, buffer_state.ptr + B >= BUFFER_CAPACITY)
        
        return LSPIBufferState(
            traces=new_traces, features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            ptr=new_ptr, full=new_full
        )

    def solve_lspi_buffer(buffer_state: LSPIBufferState, Sigma_inv, lstd_state_dict, config):
        """Solves LSPI iteratively over the buffer using chunked memory-safe scans."""
        CHUNK_SIZE = 10_000 
        NUM_CHUNKS = BUFFER_CAPACITY // CHUNK_SIZE
        
        # Reshape buffer into chunks
        chunked_phi_sa = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_next_phi_s = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_traces = buffer_state.traces.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = jnp.where(buffer_state.full, True, jnp.arange(BUFFER_CAPACITY) < buffer_state.ptr)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]
        
        # --- Bayesian Optimistic Prior ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        new_sa_diag_counts = lstd_state_dict["sa_diag_counts"]
        lambda_kA = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_sa_diag_counts)
        lambda_kA = jnp.where(lambda_kA >= 0.1, lambda_kA, 0.0)
        Lambda_mat = jnp.diag(lambda_kA)
        prior_b = jnp.diag(Lambda_mat) * lstd_state_dict["V_max"]
        reg = jnp.eye(dim_kA) * config["A_REGULARIZATION_PER_STEP"]

        def lspi_step(w_current, _):
            def process_chunk(carry, chunk_data):
                A_acc, b_acc = carry
                c_phi_sa, c_next_phi_s, c_traces, c_term, c_absorb, c_mask = chunk_data
                
                # Retroactively evaluate rho on the next state features
                next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi_s)
                
                # 1. Greedy Policy Evaluation
                w_reshaped = w_current.reshape(n_actions, k)
                Q_next = jnp.einsum("...k, ak -> ...a", c_next_phi_s, w_reshaped)
                greedy_actions = jnp.argmax(Q_next, axis=-1)
                Pi_greedy = jax.nn.one_hot(greedy_actions, n_actions)
                
                PΠφ = expected_next_sa_features(c_next_phi_s, Pi_greedy)
                
                # 2. Construction of A
                c_traces_masked = c_traces * c_mask
                
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
                (chunked_phi_sa, chunked_next_phi_s, chunked_traces, chunked_terminals, chunked_absorb, chunked_mask)
            )
            
            # Add prior and solve ONLY ONCE per LSPI loop iteration
            A_view = final_A + Lambda_mat + reg
            b_view = final_b + prior_b
            w_new = jnp.linalg.solve(A_view, b_view)
            return w_new, None

        w_init = lstd_state_dict["w"]
        w_final, _ = jax.lax.scan(lspi_step, w_init, None, length=config.get("LSPI_NUM_ITERS", 3))
        
        return {
            "w": w_final,
            "Beta": lstd_state_dict["Beta"],
            "V_max": lstd_state_dict["V_max"],
            "sa_diag_counts": new_sa_diag_counts
        }
    
    V_max = 1.0 / (1 - config["GAMMA_i"])
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k)

    def train(rng):
        initial_lstd_state = {
            "w": jnp.zeros(dim_kA),
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            "sa_diag_counts": jnp.zeros(dim_kA) 
        }
        
        initial_buffer_state = LSPIBufferState(
            traces=jnp.zeros((BUFFER_CAPACITY, dim_kA)),
            features=jnp.zeros((BUFFER_CAPACITY, dim_kA)),
            next_features=jnp.zeros((BUFFER_CAPACITY, k)),
            terminals=jnp.zeros((BUFFER_CAPACITY, 1)),
            absorb_masks=jnp.zeros((BUFFER_CAPACITY, 1)),
            ptr=jnp.array(0, dtype=jnp.int32),
            full=jnp.array(False, dtype=jnp.bool_)
        )
        
        initial_sigma_state = {"S": jnp.eye(k)} # Global feature accumulation

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

            # --- 1. FEATURE EXTRACTION ---
            phi_s = batch_get_features(traj_batch.obs)
            next_phi_s = batch_get_features(traj_batch.next_obs)
            phi_sa = expand_to_sa_features(phi_s, n_actions, traj_batch.action)
            
            terminals = jnp.where(terminate_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            traces = helpers.calculate_traces(traj_batch, phi_sa, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_episodic, is_absorbing)
            
            # --- 2. PRE-UPDATE SURPRISE (For GAE) ---
            Sigma_inv_old = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))
            batch_next_rho_pre_update = get_scale_free_bonus(Sigma_inv_old, next_phi_s)

            # --- 3. KNOWLEDGE ASSIMILATION ---
            buffer_state = update_buffer(buffer_state, traces, phi_sa, next_phi_s, terminals, absorb_masks)
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
            Sigma_inv_new = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))

            # --- 4. UPDATE GLOBAL COUNTS FOR PRIOR ---
            batch_sa_precision = (phi_sa**2).sum(axis=(0, 1))
            Pi_uniform = jnp.ones((*traj_batch.done.shape, n_actions)) * (1.0 / n_actions)
            uniform_abs_features = expected_next_sa_features(next_phi_s, Pi_uniform) * absorb_masks[..., None]
            absorbing_sa_precision = (uniform_abs_features**2).sum(axis=(0, 1))
            
            new_sa_diag_counts = lstd_state["sa_diag_counts"] + batch_sa_precision + absorbing_sa_precision
            lstd_state = {**lstd_state, "sa_diag_counts": new_sa_diag_counts}

            # --- 5. SOLVE LSPI (POST-UPDATE) ---
            lstd_state = solve_lspi_buffer(buffer_state, Sigma_inv_new, lstd_state, config)

            # --- 6. BATCH INTRINSIC VALUES FOR GAE ---
            rho_scale = lstd_state["Beta"]
            
            # Evaluate greedy values using post-update weights
            w_reshaped = lstd_state["w"].reshape(n_actions, k)
            Q_curr = jnp.einsum("...k, ak -> ...a", phi_s, w_reshaped)
            v_i = jnp.max(Q_curr, axis=-1) * rho_scale

            Q_next = jnp.einsum("...k, ak -> ...a", next_phi_s, w_reshaped)
            next_v_i = jnp.max(Q_next, axis=-1) * rho_scale
            
            # Critical: Reward uses PRE-UPDATE surprise to drive exploration
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=batch_next_rho_pre_update * rho_scale, next_i_val=next_v_i)

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
                    "feat_norm": jnp.linalg.norm(next_phi_s, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                    "mean_rew": traj_batch.reward.mean(),
                    "lambda_kA": lstd_state['sa_diag_counts'],
                    "beta": lstd_state["Beta"],
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
                    rho = get_scale_free_bonus(Sigma_inv_new, phi) * rho_scale
                    return rho

                def get_vi(obs):
                    phi = batch_get_features(obs)
                    w_r = lstd_state["w"].reshape(n_actions, k)
                    Q_v = jnp.einsum("...k, ak -> ...a", phi, w_r)
                    return jnp.max(Q_v, axis=-1) * rho_scale

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