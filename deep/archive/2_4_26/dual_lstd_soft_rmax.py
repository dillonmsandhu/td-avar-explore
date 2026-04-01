# Covariance-Based Intrinsic Reward, propagated by LSTD (Dual LSTD for Ve and Vi).
# Consolidated version: Handles both standard training and ExactValue logging via config.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
SAVE_DIR = 'cov_lstd_dual_soft_rmax'

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

def EMA(coeff, x_old, x_new):
    return (1 - coeff) * x_old + coeff * x_new


def sigma_update_weighted(
    sigma_state: Dict,
    features: jnp.ndarray,  # Shape: (..., k)
    mask: jnp.ndarray,      # Shape: (...) matching the batch dimensions of features
    α: float,
):
    """
    Masks out those that are included twice
    Takes a mask that corresponds to what included feature vectors are valid for the update.
    """
    S = sigma_state['S']
    S_update = jnp.einsum("...i, ...j -> ...ij", features, features)
    
    # 2. Apply Mask
    # Expand mask to (..., 1, 1) so it broadcasts over the (k, k) matrix dimensions
    # Zeros out the outer products corresponding to invalid/padding states
    S_masked = S_update * mask[..., None, None]
    
    # 3. Compute Weighted Mean
    batch_axes = tuple(range(mask.ndim))
    total_valid = jnp.sum(mask)
    S_batch_mean = jnp.sum(S_masked, axis=batch_axes) / total_valid
    
    # 4. Update & Force Symmetry
    S_new = EMA(α, S, S_batch_mean)
    S_new = 0.5 * (S_new + S_new.T)
    
    # 5. Update N
    N_new = sigma_state['N'] + jnp.sum(mask)
    
    return {
        'S': S_new, 
        'N': N_new, 
        't': sigma_state['t'] + 1
    }

def make_train(config):
    
    def lambda_i_schedule(step, initial_lambda = 1.0, final_lambda = config["GAE_LAMBDA_i"], decay_steps = config["TOTAL_TIMESTEPS"]):
        """Linear Decay of Lambda from 1.0 (pure rewards) to GAE_LAMBDA_I"""
        fraction = jnp.clip(step / decay_steps, 0.0, 1.0)
        return initial_lambda + fraction * (final_lambda - initial_lambda)
    
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

    if config['EPISODIC']: 
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
        trace_fn = helpers._get_all_traces # continuing due to setting phi' = 0 when done = True. 
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov(z, phi, phi_prime, done, config['GAMMA_i'])
    else:
        gae_fn = helpers.calculate_i_and_e_gae_two_critic
        trace_fn = helpers._get_all_traces_continuing
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov_continuing(z, phi, phi_prime, done, config['GAMMA_i'])

    def get_scale_free_bonus(S, features):
        """bonus = x^T Sigma^{-1} X, where Sigma^{-1} is the empriical second moment inverse."""
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features)
        return jnp.sqrt(bonus_sq)
    
    def lstd_batch_update(
        lstd_state: Dict,
        transitions,
        features,          # phi_t
        next_features,     # phi_{t+1}
        traces,
        lambda_s,          # uncertainty in current observation
    ):
        """
        LSTD update with:
        - intrinsic reward based on next-state uncertainty
        - soft LSPI-RMAX prior on intrinsic value using state-dependent lambda
        """

        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']
        t = lstd_state['t']
        α = alpha_fn_lstd(t)

        rho = transitions.intrinsic_reward
        rew = transitions.reward  # extrinsic

        # ------------------------------------------------------------
        # 2. Standard LSTD A update (policy evaluation)
        # ------------------------------------------------------------
        A_update = jax.vmap(jax.vmap(cross_cov))(
            traces, features, next_features, transitions.done
        )
        lambda_s = jnp.clip(lambda_s, 0.0, 1.0) # lambda_s denotes
        A_i_update = A_update * (1-lambda_s)[..., None, None]# downweight each contribution as (1-lambda)
        # A_i_update = A_update

        A_batch = A_update.mean(axis=batch_axes)
        A_i_batch = A_i_update.mean(axis=batch_axes)
        weighted_gram_batch = jnp.einsum('bt, bti, btj->ij', lambda_s, features, features) / transitions.done.size
        A_batch_rmax = A_i_batch + weighted_gram_batch
        A_i_view = EMA(α,lstd_state['A_i'], A_batch_rmax) 
        
        # for storage (note having two LSTD A's is redundant - they should be the same.):
        A_i = EMA(α, lstd_state["A_i"], A_batch)
        A_e = EMA(α, lstd_state["A_e"], A_batch)
        
        ## feature-reward vector
        b_e_sample = traces * rew[..., None]
        b_e_batch = b_e_sample.mean(axis=batch_axes)

        b_i_sample = traces * rho[..., None]
        b_i_sample_view = (1-lambda_s)[..., None] * b_i_sample  + lambda_s[...,None] * lstd_state['V_max'] * features
        # b_i_sample_view = b_i_sample
        b_i_view = b_i_sample_view.mean(axis=batch_axes)
        b_i_view = EMA(alpha_fn_lstd_bi(t), lstd_state["b_i"], b_i_view)
        
        # for storage:
        b_i = EMA(alpha_fn_lstd_bi(t), lstd_state["b_i"], b_i_sample.mean(axis=batch_axes))
        b_e = EMA(α, lstd_state["b_e"], b_e_batch)

        # ------------------------------------------------------------
        # 6. Solve linear systems
        # ------------------------------------------------------------
        reg = jnp.eye(A_batch.shape[0]) * config['A_REGULARIZATION_PER_STEP']
        w_i = jnp.linalg.solve(A_i_view + reg, b_i_view)
        w_e = jnp.linalg.solve(A_e + reg, b_e)

        return {
            "A_i": A_i,
            "A_e": A_e,
            "b_i": b_i,
            "b_e": b_e,
            "w_i": w_i,
            "w_e": w_e,
            "N": N,
            "t": t + 1,
            "V_max": lstd_state['V_max'] #tracks the highest (unscaled) intrinsic value from the prior batch.
        }

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
        target_rng, rng = jax.random.split(rng)
        
        # RND Networks
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)
        
        # LSTD State (Single A, Dual b/w)
        initial_lstd_state = {
            'A_i': jnp.eye(k) * config['A_REGULARIZATION'], 
            'A_e': jnp.eye(k) * config['A_REGULARIZATION'], 
            'b_i': jnp.zeros(k), 
            'b_e': jnp.zeros(k),
            'w_i': jnp.zeros(k),
            'w_e': jnp.zeros(k),
            'N': 0, 't': 1,
            'V_max': 1/(1-config['GAMMA_i']),
        }
        initial_sigma_state = {'S': jnp.eye(k) * config['GRAM_REG'], 'N': 0, 't': 1,}

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
                    done, action, ve, vi, reward, intrinsic_reward, log_prob, last_obs, obsv, info, 
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            initial_obs = last_obs 
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            # --- 1. Update EMA of Gram Matrix ---
            phi = batch_get_features(traj_batch.obs)          # Contains s_0 and Reset states
            next_phi = batch_get_features(traj_batch.next_obs)# Contains s_T (Terminal)
            terminal_phi = next_phi * traj_batch.done[..., None]
            all_phi_sigma = jnp.concatenate([phi, terminal_phi], axis=0)

            # Update Sigma (using the mask to ignore the zeroed-out non-terminals)
            mask_sigma = jnp.concatenate([jnp.ones_like(traj_batch.done), traj_batch.done], axis=0)
            
            sigma_state = sigma_update_weighted(
                sigma_state, 
                all_phi_sigma, 
                mask_sigma,
                alpha_fn(sigma_state['t'])
            )

            # --- 3. INTRINSIC REWARD & OPTIMISM ---
            beta = beta_fn(sigma_state['N'])
            rho_scale = beta / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))
            # rho_scale = lstd_state['Beta'] / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))
            
            # Calculate Bonus Function
            int_rew_from_features = lambda phi: get_scale_free_bonus(sigma_state['S'], phi) 
            # A. Intrinsic Reward (Future Novelty)
            rho = int_rew_from_features(next_phi)
            rho = rho - rho.min()
            traj_batch = traj_batch._replace(intrinsic_reward=rho) # used by LSTD estimate
            
            # --- 4. LAMBDA CALCULATION ---
            # Now this is mathematically sound:
            # - phi matches the LSTD state.
            # - rho_current matches the uncertainty of that specific phi.
            rho_current = int_rew_from_features(phi)  # corresponds to estimated standard deviation of least squares estimate at phi (for example, least squares reward prediction)
            PRIOR_SAMPLES = config.get('LSTD_PRIOR_SAMPLES', 1.0)
            scaled_uncertainty = PRIOR_SAMPLES * (rho_current**2)
            # Per-State precision-based ratio: 1 / (1 + data_precision / prior_precision)
            lambda_s = scaled_uncertainty / (1.0 + scaled_uncertainty)

            # # --- 2. FEATURES & INTRINSIC REWARD ---
            # initial_obs_expanded = jnp.expand_dims(initial_obs, axis=0)
            # all_encountered_obs = jnp.concatenate([initial_obs_expanded, traj_batch.next_obs], axis=0)
            # all_phi = batch_get_features(all_encountered_obs)
            # next_phi = all_phi[1:]
            # phi = all_phi[:-1]

            # # Update Sigma & Calc Bonus
            # sigma_state = helpers.sigma_update(sigma_state, traj_batch, all_phi, alpha_fn(sigma_state['t']))
            # beta = beta_fn(sigma_state['N'])
            # rho_scale = beta / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))

            # int_rew_from_features = lambda features: get_scale_free_bonus(sigma_state['S'], features) 
            # rho = int_rew_from_features(next_phi)
            # rho = rho - rho.min() # bring minimal novelty to zero.
            # # jax.debug.print('unstandardized rho for lstd: {rho}', rho = rho[20:25, 0])
            # # For LSTD, use unscaled rho
            # traj_batch = traj_batch._replace(intrinsic_reward=rho)    
            
            # # --- 3. LSTD UPDATE (Extrinsic & Intrinsic) ---
            # rho_current = int_rew_from_features(phi) / jnp.sqrt(sigma_state['N'])
            # # ------------------------------------------------------------
            # # 2. Compute Precision-Based lambda_s
            # # ------------------------------------------------------------
            # # PRIOR_SAMPLES acts as the 'k' threshold. 
            # # Increasing this makes the agent stay optimistic (high lambda) for longer.
            # PRIOR_SAMPLES = config.get('LSTD_PRIOR_SAMPLES', 1.0)

            # # Precision-based ratio: 1 / (1 + data_precision / prior_precision)
            # # This smoothly decays from 1.0 (novel) to 0.0 (known)
            # # lambda_s = 1.0 / (1.0 + (PRIOR_SAMPLES * (rho_current**2)))
            # scaled_uncertainty = PRIOR_SAMPLES * (rho_current**2)
            # lambda_s = scaled_uncertainty / (1.0 + scaled_uncertainty)
        
            # ------------------------------------------------------------
            # 3. Update LSTD State
            # ------------------------------------------------------------
            traces = trace_fn(traj_batch, phi, config['GAMMA_i'], config['GAE_LAMBDA_i'])
            
            lstd_state = lstd_batch_update(
                lstd_state, 
                traj_batch, 
                phi, 
                next_phi, 
                traces, 
                lambda_s, 
            )

            # Compute Values:
            last_phi = batch_get_features(last_obs)
            
            v_e = phi @ lstd_state['w_e']
            last_val_e = last_phi @ lstd_state['w_e']
            
            v_i = phi @ lstd_state['w_i']
            last_val_i = last_phi @ lstd_state['w_i']
            # jax.debug.print("unscaled vi {vi}", vi = v_i[20:25])

            # Update V_MAX:
            lstd_state['V_max'] = jnp.maximum(jnp.max(v_i), jnp.max(last_val_i)) # unscaled maximum value.

            v_i *= rho_scale
            last_val_i *= rho_scale
            # --- 4. ADVANTAGE CALCULATION ---            
            traj_batch = traj_batch._replace(value=v_e, i_value=v_i, intrinsic_reward=rho * rho_scale)

            # Compute GAEs
            # Note: gae_fn expects (traj_batch, last_val_e, last_val_i, ...)
            # We pass v_e and v_i_opt into the function via the batch and arguments
            gaes, targets = gae_fn(traj_batch, last_val_e, last_val_i, config["GAMMA"], config["GAE_LAMBDA"], lambda_i_schedule(sigma_state['N']), config["GAMMA_i"])
            advantages = gaes[0] + gaes[1]

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
            metric = {k: v.mean() for k, v in traj_batch.info.items()}
            
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
                    phi = batch_get_features(s)
                    rho = int_rew_from_features(phi) * rho_scale
                    return rho
                
                v_e, v_i, _ = evaluator.compute_true_values(network, train_state.params, int_rew_from_state)
                # jax.debug.print("vi {vi}", vi = v_i[20:25])
                
                # 2. Compute Optimistic reward and value
                ri = int_rew_from_state(evaluator.obs_stack)

                vi_pred = batch_get_features(evaluator.obs_stack) @ lstd_state['w_i'] * rho_scale
                
                # jax.debug.print("vi pred {vi}:", vi = vi_pred[20:25])
                v_pred = batch_get_features(evaluator.obs_stack) @ lstd_state['w_e']

                if config['ENV_NAME'] == 'DeepSea-bsuite':
                    v_i_pred_opt = evaluator.get_value_grid(vi_pred)
                    vi_pred = evaluator.get_value_grid(vi_pred)
                    ri = evaluator.get_value_grid(ri)
                
                if config['ENV_NAME'] == 'Chain':
                    visitation = traj_batch.obs.sum(0).sum(0) # sum over batch axes the visitation count.
                    effective_visits = (beta / ri)**2
                    metric['visitation_count'] = visitation
                    metric['effective_visits'] = effective_visits

                metric.update({
                    "ri_grid": ri,
                    "vi_pred": vi_pred,
                    "v_i_pred_opt": vi_pred,
                    "v_i": v_i,
                    "v_e": v_e,
                    "v_e_pred": v_pred,
                    "e_value_error": jnp.mean(evaluator.reachable_mask * (v_e - v_pred)**2),
                    "i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - vi_pred)**2),
                    "opt_i_value_error": jnp.mean(evaluator.reachable_mask * (v_i - vi_pred)**2),
                    "lambda_s": jnp.mean(lambda_s),
                    "lambda_s_min": jnp.min(lambda_s),
                    "lambda_s_max": jnp.max(lambda_s),
                })
            else:
                # Use batch means as fast proxies
                metric.update({
                    "vi_pred": traj_batch.i_value.mean(),
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