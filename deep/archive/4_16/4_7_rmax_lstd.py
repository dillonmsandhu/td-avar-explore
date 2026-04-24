# RMax style - no intrinsic reward, but just a v-max scaling down by 
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_7_rmax_lstd"

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
    ptr: jnp.ndarray
    full: jnp.ndarray

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    # --- Feature Augmentation Toggle ---
    augment_lstd = config.get("AUGMENT_LSTD_FEATURES", False)
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    k_base = config.get("RND_FEATURES", 128)
    k_lstd = k_base + 2 if augment_lstd else k_base  # Adjust LSTD weight dimension
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    def update_phi_precision(lstd_state, features, next_features, done):
        # NOTE: features here should already be augmented (dimension k_lstd)
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
        """Inserts a new batch of transitions into the JAX FIFO ring buffer."""
        # Flatten batch axes
        traces = traces.reshape(-1, k_base)
        features = features.reshape(-1, k_base)
        next_features = next_features.reshape(-1, k_base)
        terminals = terminals.reshape(-1, 1)
        absorb_masks = absorb_masks.reshape(-1, 1)
        
        B = features.shape[0]
        indices = (buffer_state.ptr + jnp.arange(B)) % BUFFER_CAPACITY
        
        new_traces = buffer_state.traces.at[indices].set(traces)
        new_features = buffer_state.features.at[indices].set(features)
        new_next_features = buffer_state.next_features.at[indices].set(next_features)
        new_terminals = buffer_state.terminals.at[indices].set(terminals)
        new_absorb_masks = buffer_state.absorb_masks.at[indices].set(absorb_masks)
        
        new_ptr = (buffer_state.ptr + B) % BUFFER_CAPACITY
        new_full = jnp.logical_or(buffer_state.full, buffer_state.ptr + B >= BUFFER_CAPACITY)
        
        return LSTDBufferState(
            traces=new_traces, features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            ptr=new_ptr, full=new_full
        )

    # Chunked version for larger buffer:
    def solve_lstd_buffer(buffer_state: LSTDBufferState, lstd_state, config):
        """Solves LSTD over the entire buffer using pure prior-driven optimization."""
        CHUNK_SIZE = 100_000  # Process 10k transitions at a time to prevent OOM
        NUM_CHUNKS = BUFFER_CAPACITY // CHUNK_SIZE
        
        # Reshape buffer into chunks
        chunked_phi = buffer_state.features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_next_phi = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_traces = buffer_state.traces.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = jnp.where(buffer_state.full, True, jnp.arange(BUFFER_CAPACITY) < buffer_state.ptr)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]

        def process_chunk(carry, chunk_data):
            A_acc, count_acc  = carry
            c_phi, c_next_phi, c_traces, c_term, c_absorb, c_mask = chunk_data

            phi = c_phi * c_mask
            next_phi = c_next_phi * c_mask
            traces = c_traces * c_mask
            
            delta_Phi = phi - gamma_i * (1 - c_term) * next_phi
            A_batch = jnp.einsum("ni, nj -> ij", traces, delta_Phi)
            
            abs_features = next_phi * c_absorb
            abs_traces = abs_features 
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_traces, abs_features)
            
            return (A_acc + A_batch + A_abs, count_acc + jnp.sum(c_mask)), None

        init_A = jnp.zeros((k_lstd, k_lstd))
        
        (final_A, N_buffer), _ = jax.lax.scan(
            process_chunk, 
            (init_A, 0.0), 
            (chunked_phi, chunked_next_phi, chunked_traces, chunked_terminals, chunked_absorb, chunked_mask)
        )

        # --- Bayesian Optimistic Prior (Diagonal) ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 10.0) 
        
        # Square Root decay for persistent, smooth late-stage optimism
        safe_counts = jnp.maximum(lstd_state["phi_diag_counts"], 0.0)
        trust_ratio = PRIOR_SAMPLES / (PRIOR_SAMPLES + jnp.sqrt(safe_counts))
        
        scaled_optimism = jnp.maximum(trust_ratio , trust_ratio) # length k_lstd
        l2_reg = config.get("A_REGULARIZATION_PER_STEP", 1e-6)
        
        # 4. Construct Lambda and Prior_b
        Lambda_mat = jnp.diag(scaled_optimism + l2_reg)
        prior_b = scaled_optimism * lstd_state["V_max"]        
        
        # 5. Solve (Empirical Data vs Prior Optimism)
        A_view = final_A + Lambda_mat
        
        # Because empirical intrinsic reward is 0, the target is purely the prior.
        w_i = jnp.linalg.solve(A_view, prior_b)
        
        return {
            "w": w_i,
            "Beta": lstd_state["Beta"],
            "V_max": lstd_state["V_max"],
            "phi_diag_counts": lstd_state["phi_diag_counts"]
        }
        
    # Scale-free V_max Target (Unscaled by Beta)
    V_max = 1.0 / (1 - config["GAMMA_i"]) 
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k_base)

    def train(rng):
        initial_lstd_state = {
            "w": jnp.zeros(k_lstd),  
            "Beta": config["BONUS_SCALE"],
            "V_max": V_max,
            "phi_diag_counts": jnp.zeros(k_lstd) 
        }
        
        initial_buffer_state = LSTDBufferState(
            traces=jnp.zeros((BUFFER_CAPACITY, k_base)),
            features=jnp.zeros((BUFFER_CAPACITY, k_base)),
            next_features=jnp.zeros((BUFFER_CAPACITY, k_base)),
            terminals=jnp.zeros((BUFFER_CAPACITY, 1)),
            absorb_masks=jnp.zeros((BUFFER_CAPACITY, 1)),
            ptr=jnp.array(0, dtype=jnp.int32),
            full=jnp.array(False, dtype=jnp.bool_)
        )

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
            train_state, lstd_state, buffer_state, rnd_state, env_state, last_obs, rng, idx = runner_state

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

            # --- UPDATE BUFFER ---
            buffer_state = update_buffer(buffer_state, traces, phi, next_phi, terminals, absorb_masks)

            eval_phi = phi
            eval_next_phi = next_phi

            # --- UPDATE PRECISION PRIOR ---
            # Pass eval_phi so it tracks exactly k_lstd dimensions
            lstd_state = update_phi_precision(lstd_state, eval_phi, eval_next_phi, traj_batch.done)
            
            # --- SOLVE LSTD OVER BUFFER ---
            lstd_state = solve_lstd_buffer(buffer_state, lstd_state, config)
            
            # --- BATCH INTRINSIC VALUES FOR GAE ---
            rho_scale = lstd_state["Beta"]
            v_i = eval_phi @ lstd_state["w"] * rho_scale
            next_v_i = eval_next_phi @ lstd_state["w"] * rho_scale
            
            # NOTE: Intrinsic step reward is permanently zeroed out.
            # The prior shapes the Advantage entirely via v_i and next_v_i!
            zero_intrinsic_reward = jnp.zeros_like(traj_batch.reward)
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=zero_intrinsic_reward, next_i_val=next_v_i)

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
            # Shared Metrics
            metric.update(
                {
                    "ppo_loss": loss_info[0],
                    "rnd_loss": loss_info[1],
                    "feat_norm": jnp.linalg.norm(next_phi, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(), # This captures the intrinsic Advantage!
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(), # Will be exactly 0
                    "mean_rew": traj_batch.reward.mean(),
                    "lambda_k": lstd_state['phi_diag_counts'],
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
                    return 0.0

                def int_rew_from_state(s,):  
                    # Pure Optimism: There is no explicit step reward.
                    # We return zeros matching the batch dimension of s.
                    # (Assuming s has shape [Batch, ...])
                    return jnp.zeros(s.shape[0])

                def get_vi(obs):
                    # Pure Optimism: The value is simply the base features 
                    # projected directly onto the optimistic weights.
                    phi_eval = batch_get_features(obs)
                    return phi_eval @ lstd_state["w"] * rho_scale

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

            runner_state = (train_state, lstd_state, buffer_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_buffer_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)