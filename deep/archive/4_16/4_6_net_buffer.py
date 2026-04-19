# Covariance-Based Intrinsic Reward, propagated by a Neural Intrinsic Value Net
# Trained via Off-Policy Fitted Value Iteration (FVI) on a Replay Buffer
from core.imports import *
import core.helpers as helpers
import core.networks as networks

SAVE_DIR = '4_6_net_buffer'

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

class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    ptr: jnp.ndarray
    full: jnp.ndarray

def calculate_gae(
    traj_batch, γ, λ, 
    is_episodic: bool, is_absorbing: bool, 
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

def update_buffer(buffer_state: ReplayBufferState, obs, next_obs, terminals, absorb_masks, config):
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    obs_shape = obs.shape[2:]
    B = obs.shape[0] * obs.shape[1]
    
    obs = obs.reshape(B, *obs_shape)
    next_obs = next_obs.reshape(B, *obs_shape)
    terminals = terminals.reshape(B, 1)
    absorb_masks = absorb_masks.reshape(B, 1)
    
    indices = (buffer_state.ptr + jnp.arange(B)) % BUFFER_CAPACITY
    
    new_obs = buffer_state.obs.at[indices].set(obs)
    new_next_obs = buffer_state.next_obs.at[indices].set(next_obs)
    new_terminals = buffer_state.terminals.at[indices].set(terminals)
    new_absorb_masks = buffer_state.absorb_masks.at[indices].set(absorb_masks)
    
    new_ptr = (buffer_state.ptr + B) % BUFFER_CAPACITY
    new_full = jnp.logical_or(buffer_state.full, buffer_state.ptr + B >= BUFFER_CAPACITY)
    
    return ReplayBufferState(
        obs=new_obs, next_obs=new_next_obs, terminals=new_terminals, absorb_masks=new_absorb_masks,
        ptr=new_ptr, full=new_full
    )

def sample_buffer(rng, buffer_state, batch_size, config):
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    max_idx = jnp.where(buffer_state.full, BUFFER_CAPACITY, buffer_state.ptr)
    max_idx = jnp.maximum(max_idx, 1)
    indices = jax.random.randint(rng, (batch_size,), 0, max_idx)
    return (
        buffer_state.obs[indices],
        buffer_state.next_obs[indices],
        buffer_state.terminals[indices].squeeze(-1),
        buffer_state.absorb_masks[indices].squeeze(-1)
    )


def make_train(config):
    is_episodic = config.get('EPISODIC', True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] 
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    k = config.get('RND_FEATURES', 128)
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    
    def train(rng):
        # Initialize RND
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        # Initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=3)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

        # Target network for stable off-policy DQN-style updates
        target_net_params = train_state.params

        initial_sigma_state = {'S': jnp.eye(k),}
        BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
        initial_buffer_state = ReplayBufferState(
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
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            
            train_state, target_net_params, sigma_state, buffer_state, rnd_state, env_state, last_obs, beta, rng, idx = runner_state
            
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
                    done, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info
                )

                return (train_state, rnd_state, env_state, obsv, rng), transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # --------- Update Sigma and compute intrinsic reward ---------
            Sigma_inv = jnp.linalg.solve(sigma_state["S"], jnp.eye(k))
            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)

            rho = int_rew_from_features(batch_get_features(traj_batch.next_obs))
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            
            exact_terminal_i_val = rho / (1.0 - config["GAMMA_i"])
            fixed_next_i_val = jnp.where(
                jnp.logical_and(traj_batch.done, is_absorbing), 
                exact_terminal_i_val, 
                traj_batch.next_i_val
            )
            traj_batch = traj_batch._replace(next_i_val=fixed_next_i_val)            

            # --------- ADVANTAGE CALCULATION (Unified Absorbing) ---------
            gaes, targets = calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
                is_episodic=is_episodic, is_absorbing=is_absorbing, 
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            ext_targets = targets[0] # We only need extrinsic targets for the on-policy PPO
            
            rho_scale = config['BONUS_SCALE']
            advantages = gae_e + (rho_scale * gae_i) 

            # --------- UPDATE BUFFER ---------
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            buffer_state = update_buffer(buffer_state, traj_batch.obs, traj_batch.next_obs, terminals, absorb_masks, config)

            # --------- UPDATE NETWORK (Dual On/Off Policy) ---------
            def _update_epoch(update_state, unused):
                def _update_minbatch(states, batch_info):
                    train_state, target_net_params, rng = states
                    traj_batch, advantages, ext_targets = batch_info
                    
                    # 1. Pull Random Buffer Batch for FVI
                    rng, _rng = jax.random.split(rng)
                    b_obs, b_next_obs, b_term, b_absorb = sample_buffer(
                        _rng, buffer_state, config["MINIBATCH_SIZE"], config
                    )
                    
                    def combined_loss_fn(params):
                        # --- A. On-Policy PPO (Actor + Extrinsic Value) ---
                        pi, v_ext, _ = network.apply(params, traj_batch.obs)
                        value_loss = jnp.mean(jnp.square(v_ext - ext_targets))
                        
                        log_prob = pi.log_prob(traj_batch.action)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = jnp.clip(ratio, 1.0 - config.get("CLIP_EPS", 0.2), 1.0 + config.get("CLIP_EPS", 0.2)) * gae_norm
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        
                        entropy = pi.entropy().mean()
                        ppo_loss = loss_actor + config.get("VF_COEF", 0.5) * value_loss - config.get("ENT_COEF", 0.01) * entropy
                        
                        # --- B. Off-Policy FVI (Intrinsic Value) ---
                        _, _, pred_v_int = network.apply(params, b_obs)
                        
                        # Recompute fresh reward with Target Net
                        b_next_phi_rnd = rnd_net.apply(target_params, b_next_obs)
                        b_rho = get_scale_free_bonus(Sigma_inv, b_next_phi_rnd)
                        _, _, b_next_v_int = network.apply(target_net_params, b_next_obs)
                        
                        exact_terminal_i_val = b_rho / (1.0 - config["GAMMA_i"])
                        fixed_next_i_val = jnp.where(
                            (b_term == 1) & is_absorbing, 
                            exact_terminal_i_val, 
                            b_next_v_int
                        )
                        
                        i_boot_mult = 1.0 - (b_term * (is_episodic and not is_absorbing))
                        target_v_int = b_rho + config["GAMMA_i"] * i_boot_mult * fixed_next_i_val
                        
                        v_int_loss = jnp.mean(jnp.square(pred_v_int - jax.lax.stop_gradient(target_v_int)))
                        
                        total_loss = ppo_loss + config.get("VF_COEF", 0.5) * v_int_loss
                        return total_loss, (v_int_loss, value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(combined_loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    return (train_state, target_net_params, rng), (total_loss, losses[0], losses[1], losses[2], losses[3])
                
                train_state, target_net_params, traj_batch, advantages, ext_targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, ext_targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                (train_state, target_net_params, rng), losses = jax.lax.scan(_update_minbatch, (train_state, target_net_params, rng), minibatches)
                return (train_state, target_net_params, traj_batch, advantages, ext_targets, rng), losses
            
            initial_update_state = (train_state, target_net_params, traj_batch, advantages, ext_targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, target_net_params, _, _, _, rng = update_state

            # --- Target Network EMA Update ---
            target_net_params = jax.tree_map(lambda p, tp: 0.05 * p + 0.95 * tp, train_state.params, target_net_params)

            # UPDATE Covariance
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
            
            # --------- Update metrics ------
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
                "bonus_mean": gae_i.mean(),
                "bonus_std": gae_i.std(),
                "bonus_max": gae_i.max(),
                "lambda_ret_mean": targets[0].mean(),
                "lambda_ret_std": targets[0].std(),
                "intrinsic_rew_mean": scaled_reward.mean(),
                "intrinsic_rew_std": scaled_reward.std(),
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
                    rho = int_rew_from_features(phi) * rho_scale
                    return rho
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, 
                                                    beta, network, train_state, traj_batch, rho_scale=rho_scale)
                
            runner_state = (train_state, target_net_params, sigma_state, buffer_state, rnd_state, env_state, last_obs, beta, rng, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        init_runner_state = (train_state, target_net_params, initial_sigma_state, initial_buffer_state, rnd_state, env_state, obsv, config['BONUS_SCALE'], _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, init_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)