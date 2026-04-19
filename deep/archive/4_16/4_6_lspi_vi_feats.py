# Covariance-Based Intrinsic Reward, propagated by LSPI over Intrinsic Value Features
# Network distills the optimal LSPI Q-max targets to learn representations
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = '4_6_lspi_vi_feats'

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
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    actions: jnp.ndarray        # Required for State-Action features
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    ptr: jnp.ndarray
    full: jnp.ndarray

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
    n_actions = env.action_space(env_params).n
    k = config.get('RND_FEATURES', 128)
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
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

    def update_buffer(buffer_state: LSPIBufferState, obs, next_obs, actions, terminals, absorb_masks):
        obs = obs.reshape(-1, *obs_shape)
        next_obs = next_obs.reshape(-1, *obs_shape)
        actions = actions.reshape(-1)
        terminals = terminals.reshape(-1, 1)
        absorb_masks = absorb_masks.reshape(-1, 1)
        
        B = obs.shape[0]
        indices = (buffer_state.ptr + jnp.arange(B)) % BUFFER_CAPACITY
        
        new_obs = buffer_state.obs.at[indices].set(obs)
        new_next_obs = buffer_state.next_obs.at[indices].set(next_obs)
        new_actions = buffer_state.actions.at[indices].set(actions)
        new_terminals = buffer_state.terminals.at[indices].set(terminals)
        new_absorb_masks = buffer_state.absorb_masks.at[indices].set(absorb_masks)
        
        new_ptr = (buffer_state.ptr + B) % BUFFER_CAPACITY
        new_full = jnp.logical_or(buffer_state.full, buffer_state.ptr + B >= BUFFER_CAPACITY)
        
        return LSPIBufferState(
            obs=new_obs, next_obs=new_next_obs, actions=new_actions, terminals=new_terminals, absorb_masks=new_absorb_masks,
            ptr=new_ptr, full=new_full
        )

    def solve_lspi_buffer(buffer_state: LSPIBufferState, rnd_params, rnd_net, network_params, network, Sigma_inv, k_val, lstd_state, config):
        CHUNK_SIZE = 10_000  
        NUM_CHUNKS = BUFFER_CAPACITY // CHUNK_SIZE
        dim_kA = n_actions * k_val
        
        chunked_obs = buffer_state.obs.reshape(NUM_CHUNKS, CHUNK_SIZE, *obs_shape)
        chunked_next_obs = buffer_state.next_obs.reshape(NUM_CHUNKS, CHUNK_SIZE, *obs_shape)
        chunked_actions = buffer_state.actions.reshape(NUM_CHUNKS, CHUNK_SIZE)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        valid_mask = jnp.where(buffer_state.full, True, jnp.arange(BUFFER_CAPACITY) < buffer_state.ptr)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        
        gamma_i = config["GAMMA_i"]

        def get_sa_features(phi_s, actions_one_hot):
            # Expands state features into action-specific blocks
            return jnp.einsum("nk, na -> nak", phi_s, actions_one_hot).reshape(phi_s.shape[0], -1)

        def lspi_step(w_current, _):
            def process_chunk(carry, chunk_data):
                A_acc, b_acc, diag_acc, n_acc = carry
                c_obs, c_next_obs, c_acts, c_term, c_absorb, c_mask = chunk_data
                
                # 1. RND Features (For Target Reward)
                c_next_phi_rnd = rnd_net.apply(rnd_params, c_next_obs)
                next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi_rnd)
                
                # 2. State Features
                c_phi_s = network.apply(network_params, c_obs, method=network.get_i_value_features)
                c_next_phi_s = network.apply(network_params, c_next_obs, method=network.get_i_value_features)
                
                # 3. State-Action Features for current transition
                c_acts_one_hot = jax.nn.one_hot(c_acts, n_actions)
                c_phi_sa = get_sa_features(c_phi_s, c_acts_one_hot)
                c_phi_sa_masked = c_phi_sa * c_mask
                
                # 4. Greedy Next Policy Evaluation
                w_reshaped = w_current.reshape(n_actions, k_val)
                Q_next = jnp.einsum("nk, ak -> na", c_next_phi_s, w_reshaped)
                greedy_actions = jnp.argmax(Q_next, axis=-1)
                Pi_greedy = jax.nn.one_hot(greedy_actions, n_actions)
                PΠφ = get_sa_features(c_next_phi_s, Pi_greedy)
                
                # 5. Construction of A
                S_chunk = jnp.einsum("ni, nj -> ij", c_phi_sa_masked, c_phi_sa)
                γPΠφ = gamma_i * (1 - c_term) * PΠφ
                γPΠΦ_chunk = jnp.einsum("ni, nj -> ij", c_phi_sa_masked, γPΠφ)
                A_std = S_chunk - γPΠΦ_chunk
                
                abs_features = PΠφ * c_absorb
                abs_traces = c_phi_sa * c_absorb
                A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_traces * c_mask, abs_features)
                A_batch = A_std + A_abs
                
                # 6. Construction of b
                b_std = jnp.einsum("ni, n -> i", c_phi_sa_masked, next_rho * c_mask.squeeze(-1))
                b_abs = jnp.einsum("ni, n -> i", abs_traces * c_mask, next_rho * c_absorb.squeeze(-1) * c_mask.squeeze(-1))
                b_batch = b_std + b_abs
                
                chunk_valid_n = jnp.sum(c_mask) + jnp.sum(c_absorb * c_mask)
                chunk_diag = (c_phi_sa**2).sum(axis=0) + (abs_features**2).sum(axis=0)
                
                return (A_acc + A_batch, b_acc + b_batch, diag_acc + chunk_diag, n_acc + chunk_valid_n), None

            init_A = jnp.zeros((dim_kA, dim_kA))
            init_b = jnp.zeros(dim_kA)
            init_diag = jnp.zeros(dim_kA)
            init_n = jnp.array(0.0)
            
            (final_A, final_b, final_diag, buf_N), _ = jax.lax.scan(
                process_chunk, 
                (init_A, init_b, init_diag, init_n), 
                (chunked_obs, chunked_next_obs, chunked_actions, chunked_terminals, chunked_absorb, chunked_mask)
            )
            
            # Prior & Solve (Normalized by Buffer Mass)
            PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
            lambda_kA = PRIOR_SAMPLES / (PRIOR_SAMPLES + final_diag)
            lambda_kA = jnp.where(lambda_kA >= 0.1, lambda_kA, 0.0)
            Lambda_mat = jnp.diag(lambda_kA)
            
            safe_buf_N = jnp.maximum(1.0, buf_N)
            A_mean = final_A / safe_buf_N
            b_mean = final_b / safe_buf_N
            
            V_max_unscaled = 1.0 / (1 - config["GAMMA_i"])
            if config["NORMALIZE_FEATURES"]:
                V_max_unscaled /= jnp.sqrt(k_val)
                
            prior_b = jnp.diag(Lambda_mat) * V_max_unscaled
            reg = jnp.eye(dim_kA) * config["A_REGULARIZATION_PER_STEP"]
            
            A_view = A_mean + Lambda_mat + reg
            b_view = b_mean + prior_b
            
            w_new = jnp.linalg.solve(A_view, b_view)
            return w_new, None

        w_init = lstd_state["w"]
        w_final, _ = jax.lax.scan(lspi_step, w_init, None, length=config.get("LSPI_NUM_ITERS", 3))
        
        return {"w": w_final}

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
        
        # Initialize LSPI State
        initial_lstd_state = {"w": jnp.zeros(n_actions * k_val)}

        initial_sigma_state = {'S': jnp.eye(k)}
        initial_buffer_state = LSPIBufferState(
            obs=jnp.zeros((BUFFER_CAPACITY, *obs_shape)),
            next_obs=jnp.zeros((BUFFER_CAPACITY, *obs_shape)),
            actions=jnp.zeros(BUFFER_CAPACITY, dtype=jnp.int32),
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
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            
            train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, beta, rng, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

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
                return (train_state, env_state, obsv, rng), transition
            
            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # 1. Update Sigma with RND
            phi_rnd_flat = batch_get_features(traj_batch.obs).reshape(-1, k)
            new_S = sigma_state["S"] + jnp.einsum("ni,nj->ij", phi_rnd_flat, phi_rnd_flat)
            sigma_state = {"S": new_S}
            Sigma_inv_rnd = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))

            # 2. Update LSPI Buffer
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            buffer_state = update_buffer(buffer_state, traj_batch.obs, traj_batch.next_obs, traj_batch.action, terminals, absorb_masks)

            # 3. Solve LSPI 
            lstd_state = update_phi_precision(lstd_state, phi, next_phi, traj_batch.done)
            lstd_state = solve_lspi_buffer(
                buffer_state, target_params, rnd_net, train_state.params, network, Sigma_inv_rnd, k_val, lstd_state, config
            )

            # 4. Evaluate Batch with LSPI (V(s) = max_a Q(s,a))
            batch_next_rho = get_scale_free_bonus(Sigma_inv_rnd, batch_get_features(traj_batch.next_obs))
            
            eval_phi_vi = network.apply(train_state.params, traj_batch.obs, method=network.get_i_value_features)
            eval_next_phi_vi = network.apply(train_state.params, traj_batch.next_obs, method=network.get_i_value_features)
            
            w_reshaped = lstd_state["w"].reshape(n_actions, k_val)
            Q_curr = jnp.einsum("...k, ak -> ...a", eval_phi_vi, w_reshaped)
            Q_next = jnp.einsum("...k, ak -> ...a", eval_next_phi_vi, w_reshaped)
            
            v_i_lspi = jnp.max(Q_curr, axis=-1)
            next_v_i_lspi = jnp.max(Q_next, axis=-1)
            
            exact_terminal_i_val = batch_next_rho / (1.0 - config["GAMMA_i"])
            fixed_next_i_val = jnp.where(
                jnp.logical_and(traj_batch.done, is_absorbing), 
                exact_terminal_i_val, 
                next_v_i_lspi
            )

            # Replace batch values with Unscaled LSPI Values
            traj_batch = traj_batch._replace(
                intrinsic_reward=batch_next_rho,
                i_value=v_i_lspi,
                next_i_val=fixed_next_i_val
            )            

            # ADVANTAGE CALCULATION 
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
                is_episodic=is_episodic, is_absorbing=is_absorbing, 
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            
            rho_scale = config['BONUS_SCALE']
            advantages = gae_e + (rho_scale * gae_i)

            # UPDATE NETWORK (Distilling V(s) = max_a Q(s,a))
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
                
                def get_vi(obs):
                    phi_eval_val = network.apply(train_state.params, obs, method=network.get_i_value_features)
                    Q_val = jnp.einsum("nk, ak -> na", phi_eval_val, lstd_state["w"].reshape(n_actions, k_val))
                    return jnp.max(Q_val, axis=-1) * rho_scale

                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, 
                                                    beta, network, train_state, traj_batch, get_vi)
                
            runner_state = (train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, beta, rng, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        init_runner_state = (train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, env_state, obsv, config['BONUS_SCALE'], _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, init_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)