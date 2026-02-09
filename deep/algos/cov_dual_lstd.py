# Covariance-Based Intrinsic Reward, propagated by LSTD (Dual LSTD for Ve and Vi).
# Consolidated version: Handles both standard training and ExactValue logging via config.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
SAVE_DIR = 'cov_lstd_dual_soft_rmax_schedule_beta'

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
    
    def lstd_batch_update(lstd_state: Dict, transitions, features, next_features, traces, lambda_s):
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
            "V_max": lstd_state['V_max'], #tracks the highest (unscaled) intrinsic value from the prior batch.
            "Beta": lstd_state['Beta'] #tracks the highest (unscaled) intrinsic value from the prior batch.
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
            "Beta": config['BONUS_SCALE'],
        }
        initial_sigma_state = {'S': jnp.eye(k) * config['GRAM_REG'], 'N': 0, 't': 1,}

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # --- 1. COLLECT TRAJECTORIES ---
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state
                # act
                rng, _rng = jax.random.split(rng)
                pi, _ = network.apply(train_state.params, last_obs) 
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                # step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                # logging
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
            
            sigma_state = helpers.sigma_update_masked(
                sigma_state, 
                all_phi_sigma, 
                mask_sigma,
                alpha_fn(sigma_state['t'])
            )

            # --- 3. INTRINSIC REWARD & OPTIMISM ---
            int_rew_from_features = lambda phi: get_scale_free_bonus(sigma_state['S'], phi) 
            rho = int_rew_from_features(next_phi)
            rho = rho - rho.min()
            traj_batch = traj_batch._replace(intrinsic_reward=rho) # used by LSTD estimate
            
            # --- 4. LAMBDA CALCULATION ---
            rho_current = int_rew_from_features(phi) / jnp.sqrt(sigma_state['N'])  # corresponds to estimated standard deviation of least squares estimate at phi (for example, least squares reward prediction)
            PRIOR_SAMPLES = config.get('LSTD_PRIOR_SAMPLES', 1.0)
            scaled_uncertainty = PRIOR_SAMPLES * (rho_current**2)
            # Per-State precision-based ratio: 1 / (1 + data_precision / prior_precision)
            lambda_s = scaled_uncertainty / (1.0 + scaled_uncertainty)
        
            # ------------------------------------------------------------
            # 3. Update LSTD State
            # ------------------------------------------------------------
            traces = trace_fn(traj_batch, phi, config['GAMMA_i'], config['GAE_LAMBDA_i'])
            lstd_state = lstd_batch_update(lstd_state, traj_batch, phi, next_phi, traces, lambda_s)

            # Compute Values:
            last_phi = batch_get_features(last_obs)
            
            v_e = phi @ lstd_state['w_e']
            last_val_e = last_phi @ lstd_state['w_e']
            
            v_i = phi @ lstd_state['w_i']
            last_val_i = last_phi @ lstd_state['w_i']
            # Update V_MAX:
            lstd_state['V_max'] = jnp.maximum(jnp.max(v_i), jnp.max(last_val_i)) # unscaled maximum value.

            # set beta adaptively
            lstd_state['Beta'] = helpers.update_beta(lstd_state['Beta'], v_i, traj_batch.value, progress = sigma_state['N'] / config['TOTAL_TIMESTEPS'], update=config['ADAPTIVE_BETA'])
            rho_scale = lstd_state['Beta'] / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))            
            
            # --- 4. ADVANTAGE CALCULATION ---             
            last_val_i *= rho_scale
            traj_batch = traj_batch._replace(value=v_e, i_value= v_i * rho_scale, intrinsic_reward=rho * rho_scale)
            # Compute GAEs
            gaes, targets = gae_fn(traj_batch,
                last_val_e,
                last_val_i*rho_scale,
                config["GAMMA"],
                config["GAE_LAMBDA"],
                config["GAE_LAMBDA_i"],
                config["GAMMA_i"]
            )
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
                "lambda_s": jnp.mean(lambda_s),
                "lambda_s_min": jnp.min(lambda_s),
                "lambda_s_max": jnp.max(lambda_s),
                "beta": lstd_state['Beta'],
            })
            if evaluator is None: # No way to compute true values, just record the batch average prediction.
                metric.update({
                "vi_pred": traj_batch.i_value.mean(),
                "v_e_pred": traj_batch.value.mean()
            })
                
            else:
                def int_rew_from_state(s): # for computing the intrinsic reward given an arbitrary state 
                    phi = batch_get_features(s)
                    rho = int_rew_from_features(phi) * rho_scale
                    return rho
                
                get_vi = lambda obs: batch_get_features(obs) @ lstd_state['w_i'] * rho_scale 
                get_ve = lambda obs: batch_get_features(obs) @ lstd_state['w_e']
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, lstd_state['Beta'], network, train_state, traj_batch, get_vi, get_ve)
            
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