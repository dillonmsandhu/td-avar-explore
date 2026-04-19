# Covariance-Based Intrinsic Reward, propagated by LSTD over Intrinsic Value Features
# Network distills LSTD targets to learn optimal representations
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = '4_6_lstd_vi_feats'

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
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    ptr: jnp.ndarray
    full: jnp.ndarray

def calculate_gae(
    traj_batch, 
    γ, λ, 
    is_episodic: bool, 
    is_absorbing: bool, 
    γi=None, λi=None
):
    """Unified extrinsic and intrinsic GAE, handles episodic, continuing, and absorbing formulation."""
    if γi is None: γi = γ
    if λi is None: λi = λ

    i_mask_boot = is_episodic and not is_absorbing
    i_mask_gae = is_episodic or is_absorbing

    def _get_advantages(gae_accs, transition):
        gae, i_gae = gae_accs
        
        done = transition.done 
        i_boot_mult = 1.0 - (done * i_mask_boot)
        i_gae_mult  = 1.0 - (done * i_mask_gae)

        delta = transition.reward + γ * transition.next_value * (1-done) - transition.value
        gae = delta + (γ * λ * (1-done) * gae)
        
        i_delta = transition.intrinsic_reward + γi * transition.next_i_val * i_boot_mult - transition.i_value 
        i_gae = i_delta + (γi * λi * i_gae_mult * i_gae)
        
        return (gae, i_gae), (gae, i_gae)

    initial_accs = (jnp.zeros_like(traj_batch.value[0]), jnp.zeros_like(traj_batch.i_value[0]))

    _, (advantages, i_advantages) = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    return (advantages, i_advantages), (advantages + traj_batch.value, i_advantages + traj_batch.i_value)


def make_train(config):
    is_episodic = config.get('EPISODIC', True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] 
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    k = config.get('RND_FEATURES', 128)
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))

    def update_buffer(buffer_state: LSTDBufferState, obs, next_obs, terminals, absorb_masks):
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
            
            # 2. Intrinsic Value Features (For LSTD Matrices)
            c_phi_vi = network.apply(network_params, c_obs, method=network.get_i_value_features)
            c_next_phi_vi = network.apply(network_params, c_next_obs, method=network.get_i_value_features)
            
            c_phi_vi = c_phi_vi * c_mask
            c_next_phi_vi = c_next_phi_vi * c_mask
            
            delta_Phi = c_phi_vi - gamma_i * (1 - c_term) * c_next_phi_vi
            A_batch = jnp.einsum("ni, nj -> ij", c_phi_vi, delta_Phi)
            b_batch = jnp.einsum("ni, n -> i", c_phi_vi, next_rho * c_mask.squeeze(-1))
            
            abs_features = c_next_phi_vi * c_absorb
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_features, abs_features)
            b_abs = jnp.einsum("ni, n -> i", abs_features, next_rho * c_absorb.squeeze(-1) * c_mask.squeeze(-1))
            
            chunk_valid_n = jnp.sum(c_mask) + jnp.sum(c_absorb * c_mask)
            chunk_diag = (c_phi_vi**2).sum(axis=0) + (abs_features**2).sum(axis=0)
            
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
        
        # Prior & Solve (Unscaled Target)
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + final_diag)
        lambda_k = jnp.where(lambda_k >= 0.1, lambda_k, 0.0)
        Lambda_mat = jnp.diag(lambda_k)
        
        safe_buf_N = jnp.maximum(1.0, buf_N)
        A_mean = final_A / safe_buf_N
        b_mean = final_b / safe_buf_N
        
        V_max_unscaled = 1.0 / (1 - config["GAMMA_i"])
        if config["NORMALIZE_FEATURES"]:
            V_max_unscaled /= jnp.sqrt(k_val)
            
        prior_b = jnp.diag(Lambda_mat) * V_max_unscaled
        
        reg = jnp.eye(k_val) * config["A_REGULARIZATION_PER_STEP"]
        A_view = A_mean + Lambda_mat + reg
        b_view = b_mean + prior_b
        
        w_i = jnp.linalg.solve(A_view, b_view)
        
        return w_i, final_diag

    def train(rng):
        # Initialize RND
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        # Initialize 3-head value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=3)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

        # Get the dimension of the Intrinsic Value Features
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_vi_features = network.apply(network_params, dummy_obs, method=network.get_i_value_features)
        k_val = dummy_vi_features.shape[-1]

        initial_sigma_state = {'S': jnp.eye(k),}
        initial_buffer_state = LSTDBufferState(
            obs=jnp.zeros((BUFFER_CAPACITY, *obs_shape)),
            next_obs=jnp.zeros((BUFFER_CAPACITY, *obs_shape)),
            terminals=jnp.zeros((BUFFER_CAPACITY, 1)),
            absorb_masks=jnp.zeros((BUFFER_CAPACITY, 1)),
            ptr=jnp.array(0, dtype=jnp.int32),
            full=jnp.array(False, dtype=jnp.bool_)
        )

        def batch_get_features(obs): 
            if obs.ndim == len(obs_shape) + 2: 
                def scan_fn(carry, obs_step):
                    return None, rnd_net.apply(target_params, obs_step)
                _, out = jax.lax.scan(scan_fn, None, obs)
                return out
            return rnd_net.apply(target_params, obs) 
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, beta, rng, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value, i_val = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                true_next_obs = info["real_next_obs"]
                _, next_val, next_i_val = network.apply(train_state.params, true_next_obs)

                intrinsic_reward = jnp.zeros_like(reward)
                transition = Transition(
                    done, action, value, next_val, i_val, next_i_val, 
                    reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, rnd_state, env_state, obsv, rng), transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # -------------------------------------------------------------
            # 1. Update Sigma with RND
            phi_rnd_flat = batch_get_features(traj_batch.obs).reshape(-1, k)
            new_S = sigma_state["S"] + jnp.einsum("ni,nj->ij", phi_rnd_flat, phi_rnd_flat)
            sigma_state = {"S": new_S}
            Sigma_inv_rnd = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))

            # 2. Update Buffer
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            buffer_state = update_buffer(buffer_state, traj_batch.obs, traj_batch.next_obs, terminals, absorb_masks)

            # 3. Solve LSTD using Intrinsic Value Network Features
            w_i, diag_counts = solve_lstd_buffer(
                buffer_state, target_params, rnd_net, train_state.params, network, Sigma_inv_rnd, k_val, config
            )

            # 4. Evaluate Batch with LSTD for GAE
            batch_next_rho = get_scale_free_bonus(Sigma_inv_rnd, batch_get_features(traj_batch.next_obs))
            
            eval_phi_vi = network.apply(train_state.params, traj_batch.obs, method=network.get_i_value_features)
            eval_next_phi_vi = network.apply(train_state.params, traj_batch.next_obs, method=network.get_i_value_features)
            
            v_i_lstd = eval_phi_vi @ w_i
            next_v_i_lstd = eval_next_phi_vi @ w_i
            
            exact_terminal_i_val = batch_next_rho / (1.0 - config["GAMMA_i"])
            fixed_next_i_val = jnp.where(
                jnp.logical_and(traj_batch.done, is_absorbing), 
                exact_terminal_i_val, 
                next_v_i_lstd
            )

            # Replace batch values with Unscaled LSTD values
            traj_batch = traj_batch._replace(
                intrinsic_reward=batch_next_rho,
                i_value=v_i_lstd,
                next_i_val=fixed_next_i_val
            )            

            # -------------------------------------------------------------
            # ADVANTAGE CALCULATION (Using LSTD Values)
            gaes, targets = calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
                is_episodic=is_episodic, is_absorbing=is_absorbing, 
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            
            rho_scale = config['BONUS_SCALE']
            advantages = gae_e + (rho_scale * gae_i)

            # UPDATE NETWORK
            # targets[1] contains the LSTD-bootstrapped TD(\lambda) returns.
            # Passing this into _loss_fn_intrinsic_v forces the V_i head to distill the LSTD prediction!
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn_intrinsic_v, has_aux=True)
                    (total_loss, (i_value_loss, value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, i_value_loss, value_loss, loss_actor, entropy)
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), losses
            
            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state

            # -------------------------------
            # Metrics
            scaled_reward = traj_batch.intrinsic_reward * rho_scale
            scaled_i_val = traj_batch.i_value * rho_scale
            
            metric = {
                k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update({
                "ppo_loss": loss_info[0].mean(), 
                "i_value_loss": loss_info[1].mean(),
                "e_value_loss": loss_info[2].mean(),
                "pi_loss": loss_info[3].mean(),
                "entropy": loss_info[4].mean(),
                "feat_norm_vi": jnp.linalg.norm(eval_next_phi_vi, axis=-1).mean(),
                "bonus_mean": gae_i.mean(),
                "bonus_max": gae_i.max(),
                "lambda_ret_mean": targets[0].mean(),
                "intrinsic_rew_mean": scaled_reward.mean(),
                "mean_rew": traj_batch.reward.mean(),
                "beta": beta,
                "rho_scale": rho_scale
            })

            if evaluator is None: 
                metric.update({
                "vi_pred": scaled_i_val.mean(),
                "v_e_pred": traj_batch.value.mean()
            })
            else:
                def int_rew_from_state(s):
                    phi = batch_get_features(s)
                    rho = get_scale_free_bonus(Sigma_inv_rnd, phi) * rho_scale
                    return rho
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, 
                                                    beta, network, train_state, traj_batch, rho_scale=rho_scale)
                
            runner_state = (train_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, beta, rng, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        init_runner_state = (train_state, initial_sigma_state, initial_buffer_state, rnd_state, env_state, obsv, config['BONUS_SCALE'], _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, init_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)