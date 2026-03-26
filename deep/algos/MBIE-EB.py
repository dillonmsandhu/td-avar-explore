# Covariance-Based Intrinsic Reward, propagated by LSTD (Dual LSTD for Ve and Vi).
# Consolidated version: Handles both standard training and ExactValue logging via config.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
SAVE_DIR = 'MBIE-EB'

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray       # V_e (from LSTD)
    i_value: jnp.ndarray     # V_i (from LSTD, optimistic)
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray 
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def EMA(α, x_start, x_sample):
    return (1-α) * x_start + α * x_sample

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    k = config.get('RND_FEATURES', 128)
    calc_true_values = config.get('CALC_TRUE_VALUES', False)

    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR'])
    alpha_fn_lstd_bi = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR_RI'])
    gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
    
    if config.get('DECAY_BETA', False): 
        beta_fn = helpers.make_beta_schedule(config)
    else:
        beta_fn = lambda n: config['BONUS_SCALE']

    if calc_true_values:
        if config['ENV_NAME'] == 'DeepSea-bsuite':
            evaluator = DeepSeaExactValue(
                size=config['DEEPSEA_SIZE'], 
                unscaled_move_cost=0.01, 
                gamma=config['GAMMA'], 
                episodic=config['EPISODIC']
            )
        if config['ENV_NAME'] == 'Chain':
            evaluator = LongChainExactValue(config.get('CHAIN_LENGTH', 100), config['GAMMA'], config['EPISODIC'])
    
    assert config['EPISODIC'], "must be episodic!"

    def get_int_rew(S, features, N):
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features)
        bonus_sq /= jnp.maximum(1.0, N)
        rho = beta_fn(N) * jnp.sqrt(bonus_sq)
        return rho
    
    def mbie_batch_update(lstd_state, transitions):
        """Solves intrinsic and extrinsic Bellman equations."""
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']
        t = lstd_state['t']
        α = alpha_fn_lstd(lstd_state['t'])

        P_update = jnp.einsum('...i,...j->ij', transitions.obs, transitions.next_obs * (1.0 - transitions.done[..., None]))
        R_update = jnp.einsum('...,i->i', transitions.reward, transitions.obs)
        Ri_update = jnp.einsum('...,i->i', transitions.intrinsic_reward, transitions.obs)

        P_batch = P_update.mean(axis=batch_axes)
        R_batch = R_update.mean(axis=batch_axes)
        Ri_batch = Ri_update.mean(axis=batch_axes)

        P = EMA(α, lstd_state['P'], P_batch)
        R = EMA(α, lstd_state['R'], R_batch)
        Ri = EMA(alpha_fn_lstd_bi(t), lstd_state['Ri'], Ri_batch)

        A = jnp.eye(P.shape[0]) - config['GAMMA'] * P
        reg = config['A_REGULARIZATION_PER_STEP'] * jnp.eye(A.shape[0])
        w_i = jnp.linalg.solve(A + reg, Ri)
        w_e = jnp.linalg.solve(A + reg, R)

        return {'P': P,'R': R,'Ri': Ri,'w_i': w_i,'w_e': w_e,'N': N,'t': t + 1}
    
    # Custom PPO Loss (Actor Only)
    def actor_only_loss(params, apply_fn, minibatch, advantages):
        # We don't need targets here, as we aren't training a value head
        traj_batch = minibatch
        
        # Rerun network to get current log_probs (and entropy)
        # Note: Network is Actor-only or we ignore value head
        pi, _ = apply_fn(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # PPO Ratio
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        
        # Clipped Loss
        clip_eps = config["CLIP_EPS"]
        loss_actor1 = -ratio * advantages
        loss_actor2 = -jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()
        
        entropy = pi.entropy().mean()
        
        total_loss = loss_actor - config["ENT_COEF"] * entropy
        return total_loss, (loss_actor, entropy)

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # LSTD State (Single A, Dual b/w)
        initial_lstd_state = {
            'P': jnp.zeros(k,k), 
            'R': jnp.zeros(k), 
            'Ri': jnp.zeros(k),
            'w_i': jnp.zeros(k),
            'w_e': jnp.zeros(k),
            'N': 0, 't': 1,
        }
        initial_sigma_state = {'S': jnp.eye(k) * config['GRAM_REG'], 'N': 0, 't': 1,}
    # WARMUP with STAGGERED_STARTS
        def _warmup_step(runner_state, step_idx):
            env_state, last_obs, rng = runner_state
            
            # Split RNGs
            rng, _rng = jax.random.split(rng)
            rng_action = jax.random.split(_rng, config["NUM_ENVS"])
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])

            # Sample actions
            action = jax.vmap(env.action_space(env_params).sample)(rng_action)
            
            # Step the environments
            obsv, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                rng_step, env_state, action, env_params
            )

            if config.get("STAGGERED_STARTS", False):
                # Only update env_state if current loop step < random threshold for that env
                # This effectively stops each env at a different point in time
                rng, _rng = jax.random.split(rng)
                start_thresholds = jax.random.randint(_rng, (config["NUM_ENVS"],), 0, config["WARMUP"])
                
                # If step_idx < threshold, we keep stepping. If not, we stay put.
                active_mask = step_idx < start_thresholds
                env_state = jax.tree.map(
                    lambda x, y: jnp.where(active_mask.reshape(-1, *([1] * (x.ndim - 1))), x, y),
                    next_env_state, env_state
                )
                # Note: obsv follows the same logic if you need the specific last observation
                obsv = jnp.where(active_mask.reshape(-1, *([1] * (obsv.ndim - 1))), obsv, last_obs)
            else:
                env_state = next_env_state

            return (env_state, obsv, rng), None

        warmup_runner_state = (env_state, obsv, rng)
        # We pass jnp.arange to track the step index
        (env_state, obsv, rng), _ = jax.lax.scan(
            _warmup_step, warmup_runner_state, jnp.arange(config["WARMUP"])
        )
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # --- 1. COLLECT TRAJECTORIES ---
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                # Get Policy Action
                rng, _rng = jax.random.split(rng)
                pi, _ = network.apply(train_state.params, last_obs) 
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                intrinsic_reward = jnp.zeros_like(reward) # Placeholder
                vi = jnp.zeros_like(reward)
                ve = jnp.zeros_like(reward)
                
                transition = Transition(
                    done, action, ve, vi, reward, intrinsic_reward, log_prob, last_obs, info['true_last_obs'], info, 
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            initial_obs = last_obs 
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # --- 2. FEATURES & INTRINSIC REWARD ---
            initial_obs_expanded = jnp.expand_dims(initial_obs, axis=0)
            all_phi = jnp.concatenate([initial_obs_expanded, traj_batch.next_obs], axis=0)
            next_phi = all_phi[1:]
            phi = all_phi[:-1]
            
            # Update Sigma & Calc Bonus
            sigma_state = helpers.sigma_update(sigma_state, traj_batch, all_phi, alpha_fn(sigma_state['t']))
            int_rew_from_features = lambda features: get_int_rew(sigma_state['S'], features, sigma_state['N'])
            rho = int_rew_from_features(next_phi)
            traj_batch = traj_batch._replace(intrinsic_reward=rho)

            # --- 3. LSTD UPDATE (Extrinsic & Intrinsic) ---
            lstd_state = mbie_batch_update(lstd_state, traj_batch)

            # --- 4. ADVANTAGE CALCULATION ---
            
            # Extrinsic Value (V_e)
            v_e = phi @ lstd_state['w_e']
            last_val_e = last_obs @ lstd_state['w_e']
            
            # Intrinsic Value (V_i) - Optimistic
            v_i = phi @ lstd_state['w_i']
            last_val_i = last_obs @ lstd_state['w_i']

            # Update batch with new values
            traj_batch = traj_batch._replace(value=v_e, i_value=v_i)

            # Compute GAEs
            # Note: gae_fn expects (traj_batch, last_val_e, last_val_i, ...)
            # We pass v_e and v_i_opt into the function via the batch and arguments
            gaes, targets = gae_fn(traj_batch, last_val_e, last_val_i, config["GAMMA"], config["GAE_LAMBDA"])
            e_gae, i_gae = gaes            
            # Total Advantage
            advantages = e_gae + i_gae

            # --- 5. POLICY UPDATE (No Value Head Training) ---
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    # Unpack
                    batch_traj, batch_adv = batch_info
                    
                    grad_fn = jax.value_and_grad(actor_only_loss, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        train_state.params, network.apply, batch_traj, batch_adv
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, aux)

                train_state, traj_batch, advantages, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Shuffle
                batch = (traj_batch, advantages)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                train_state, (total_loss, aux) = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, rng), (total_loss, aux)

            initial_update_state = (train_state, traj_batch, advantages, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, rng = update_state
            
            # --- 6. METRICS ---
            total_loss_mean = loss_info[0].mean()
            actor_loss_mean = loss_info[1][0].mean()
            entropy_mean = loss_info[1][1].mean()
            
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }

            # Common Metrics
            metric.update({
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
            })

            # Branch: Expensive True Values vs. Cheap Proxies
            if calc_true_values:
                # 1. Compute Exact Values using the Evaluator
                
                def int_rew_from_state(s):
                    phi = s
                    rho = int_rew_from_features(phi)
                    return rho
                
                v_e, v_i, _ = evaluator.compute_true_values(network, train_state.params, int_rew_from_state)
                
                # 2. Compute Optimistic reward and value
                ri = int_rew_from_state(evaluator.obs_stack)

                vi_pred = lstd_state['w_i']
                v_pred = lstd_state['w_e']

                if config['ENV_NAME'] == 'DeepSea-bsuite':
                    v_i_pred_opt = evaluator.get_value_grid(vi_pred)
                    vi_pred = evaluator.get_value_grid(vi_pred)
                    ri = evaluator.get_value_grid(ri)
                
                if config['ENV_NAME'] == 'Chain':
                    visitation = traj_batch.obs.sum(0).sum(0) # sum over batch axes the visitation count.
                    metric['visitation_count'] = visitation

                metric.update({
                    "ri_grid": ri,
                    "vi_pred": vi_pred,
                    "v_i_pred_opt": v_i_pred_opt,
                    "v_i": v_i,
                    "v_e": v_e,
                    "v_e_pred": v_pred,
                    "e_value_error": jnp.mean(evaluator.reachable_mask * (v_e - v_pred)**2),
                    "i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - vi_pred)**2),
                    "opt_i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - v_i_pred_opt)**2),
                })
            else:
                # Use batch means as fast proxies
                metric.update({
                    "vi_pred": traj_batch.i_value_slow.mean(),
                    "v_i_pred_opt": traj_batch.i_value_fast.mean(),
                    "v_e_pred": traj_batch.value.mean()
                })

            runner_state = (train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)