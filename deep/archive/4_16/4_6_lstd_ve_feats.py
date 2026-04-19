from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = "4_6_lstd_ve_feats"

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

# Buffer stores observations so we can pass them through the shifting Value Net
class LSTDBufferState(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
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
    
    k_rnd = config.get("RND_FEATURES", 128)
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

    def update_buffer(buffer_state: LSTDBufferState, obs, next_obs, terminals, absorb_masks):
        """Inserts a new batch of transitions into the JAX FIFO ring buffer."""
        obs = obs.reshape(-1, *obs_shape)
        next_obs = next_obs.reshape(-1, *obs_shape)
        terminals = terminals.reshape(-1, 1)
        absorb_masks = absorb_masks.reshape(-1, 1)
        
        B = obs.shape[0]
        indices = (buffer_state.ptr + jnp.arange(B)) % BUFFER_CAPACITY
        
        new_obs = buffer_state.obs.at[indices].set(obs)
        new_next_obs = buffer_state.next_obs.at[indices].set(next_obs)
        new_terminals = buffer_state.terminals.at[indices].set(terminals)
        new_absorb_masks = buffer_state.absorb_masks.at[indices].set(absorb_masks)
        
        new_ptr = (buffer_state.ptr + B) % BUFFER_CAPACITY
        new_full = jnp.logical_or(buffer_state.full, buffer_state.ptr + B >= BUFFER_CAPACITY)
        
        return LSTDBufferState(
            obs=new_obs, next_obs=new_next_obs, terminals=new_terminals, absorb_masks=new_absorb_masks,
            ptr=new_ptr, full=new_full
        )

    # --- HYBRID SOLVER: RND for Reward, Learned Features for Value ---
    def solve_lstd_buffer(buffer_state: LSTDBufferState, rnd_params, rnd_net, network_params, network, Sigma_inv, k_val, config):
        CHUNK_SIZE = 10_000  
        NUM_CHUNKS = BUFFER_CAPACITY // CHUNK_SIZE
        
        chunked_obs = buffer_state.obs.reshape(NUM_CHUNKS, CHUNK_SIZE, *obs_shape)
        chunked_next_obs = buffer_state.next_obs.reshape(NUM_CHUNKS, CHUNK_SIZE, *obs_shape)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = jnp.where(buffer_state.full, True, jnp.arange(BUFFER_CAPACITY) < buffer_state.ptr)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]

        def process_lstd_chunk(carry, chunk_data):
            A_acc, b_acc, diag_acc, n_acc = carry
            c_obs, c_next_obs, c_term, c_absorb, c_mask = chunk_data
            
            # 1. RND Features (For Target Reward)
            c_next_phi_rnd = rnd_net.apply(rnd_params, c_next_obs)
            next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi_rnd)
            
            # 2. Learned Value Features (For LSTD Matrices)
            c_phi_val = network.apply(network_params, c_obs, method=network.get_value_features)
            c_next_phi_val = network.apply(network_params, c_next_obs, method=network.get_value_features)
            
            # Apply Masks
            c_phi_val = c_phi_val * c_mask
            c_next_phi_val = c_next_phi_val * c_mask
            
            # Standard Targets
            delta_Phi = c_phi_val - gamma_i * (1 - c_term) * c_next_phi_val
            A_batch = jnp.einsum("ni, nj -> ij", c_phi_val, delta_Phi)
            b_batch = jnp.einsum("ni, n -> i", c_phi_val, next_rho * c_mask.squeeze(-1))
            
            # Absorbing Targets
            abs_features = c_next_phi_val * c_absorb
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_features, abs_features)
            b_abs = jnp.einsum("ni, n -> i", abs_features, next_rho * c_absorb.squeeze(-1) * c_mask.squeeze(-1))
            
            chunk_valid_n = jnp.sum(c_mask) + jnp.sum(c_absorb * c_mask)
            chunk_diag = (c_phi_val**2).sum(axis=0) + (abs_features**2).sum(axis=0)
            
            return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs, diag_acc + chunk_diag, n_acc + chunk_valid_n), None

        init_A = jnp.zeros((k_val, k_val))
        init_b = jnp.zeros(k_val)
        init_diag = jnp.zeros(k_val)
        init_n = jnp.array(0.0)
        
        (final_A, final_b, final_diag, buf_N), _ = jax.lax.scan(
            process_lstd_chunk, 
            (init_A, init_b, init_diag, init_n), 
            (chunked_obs, chunked_next_obs, chunked_terminals, chunked_absorb, chunked_mask)
        )
        
        # --- Prior & Solve ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + final_diag)
        lambda_k = jnp.where(lambda_k >= 0.1, lambda_k, 0.0)
        Lambda_mat = jnp.diag(lambda_k)
        
        # Scale by mass to prevent regularization dominance
        safe_buf_N = jnp.maximum(1.0, buf_N)
        A_mean = final_A / safe_buf_N
        b_mean = final_b / safe_buf_N
        
        V_max_dynamic = 1.0 / (1 - config["GAMMA_i"])
        if config["NORMALIZE_FEATURES"]:
            V_max_dynamic /= jnp.sqrt(k_val)
            
        prior_b = jnp.diag(Lambda_mat) * V_max_dynamic
        
        reg = jnp.eye(k_val) * config["A_REGULARIZATION_PER_STEP"]
        A_view = A_mean + Lambda_mat + reg
        b_view = b_mean + prior_b
        
        w_i = jnp.linalg.solve(A_view, b_view)
        
        return w_i, final_diag

    def train(rng):
        # 1. Initialize RND Network (Fixed)
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_rnd
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"], k_rnd
        )
        batch_get_rnd_features = jax.vmap(lambda obs: rnd_net.apply(target_params, obs))

        # 2. Initialize Actor-Critic (Learned)
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        
        # Dynamically extract k_val from the network torso
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_val_features = network.apply(network_params, dummy_obs, method=network.get_value_features)
        k_val = dummy_val_features.shape[-1]
        
        train_state, _ = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        
        # 3. Initialize Global RND Sigma Accumulator
        initial_sigma_state = {"S": jnp.eye(k_rnd)}

        # 4. Initialize Observation Buffer
        initial_buffer_state = LSTDBufferState(
            obs=jnp.zeros((BUFFER_CAPACITY, *obs_shape)),
            next_obs=jnp.zeros((BUFFER_CAPACITY, *obs_shape)),
            terminals=jnp.zeros((BUFFER_CAPACITY, 1)),
            absorb_masks=jnp.zeros((BUFFER_CAPACITY, 1)),
            ptr=jnp.array(0, dtype=jnp.int32),
            full=jnp.array(False, dtype=jnp.bool_)
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            train_state, sigma_state, buffer_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
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

                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                next_i_val = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            
            # --- 1. GLOBAL RND COVARIANCE UPDATE ---
            phi_rnd = batch_get_rnd_features(traj_batch.obs)
            next_phi_rnd = batch_get_rnd_features(traj_batch.next_obs)
            
            phi_rnd_flat = phi_rnd.reshape(-1, k_rnd)
            new_S = sigma_state["S"] + jnp.einsum("ni,nj->ij", phi_rnd_flat, phi_rnd_flat)
            sigma_state = {"S": new_S}
            Sigma_inv_rnd = jnp.linalg.solve(new_S + 1e-8 * jnp.eye(k_rnd), jnp.eye(k_rnd))

            # --- 2. UPDATE BUFFER WITH OBS ---
            buffer_state = update_buffer(buffer_state, traj_batch.obs, traj_batch.next_obs, terminals, absorb_masks)
            
            # --- 3. HYBRID LSTD SOLVE ---
            w_i, diag_counts = solve_lstd_buffer(
                buffer_state, target_params, rnd_net, train_state.params, network, Sigma_inv_rnd, k_val, config
            )

            # --- 4. EVALUATE CURRENT BATCH ---
            # Reward uses RND Features
            batch_next_rho = get_scale_free_bonus(Sigma_inv_rnd, next_phi_rnd)
            
            # Value uses Learned Features
            phi_val = network.apply(train_state.params, traj_batch.obs, method=network.get_value_features)
            next_phi_val = network.apply(train_state.params, traj_batch.next_obs, method=network.get_value_features)
            
            rho_scale = config["BONUS_SCALE"]
            v_i = phi_val @ w_i * rho_scale
            next_v_i = next_phi_val @ w_i * rho_scale
            
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
            
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update(
                {
                    "ppo_loss": loss_info[0],
                    "feat_norm_val": jnp.linalg.norm(next_phi_val, axis=-1).mean(),
                    "feat_norm_rnd": jnp.linalg.norm(next_phi_rnd, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
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
                    phi_eval_rnd = batch_get_rnd_features(s)
                    rho = get_scale_free_bonus(Sigma_inv_rnd, phi_eval_rnd) * rho_scale
                    return rho

                def get_vi(obs):
                    phi_eval_val = network.apply(train_state.params, obs, method=network.get_value_features)
                    return phi_eval_val @ w_i * rho_scale

                metric = helpers.add_values_to_metric(
                    config, metric, int_rew_from_state, evaluator,
                    rho_scale, network, train_state, traj_batch, get_vi,
                )

            runner_state = (train_state, sigma_state, buffer_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_sigma_state, initial_buffer_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
