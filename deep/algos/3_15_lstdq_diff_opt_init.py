# LSTD-Q evaluating the Uniform Random Policy
# Employs State-Action features with Optimistic Initialization
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
from gymnax.environments import spaces
SAVE_DIR = '3_14_lstd_q_random_prior_init'

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray 
    i_value: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray 
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def append_frontier_transition(traj_batch: Transition, last_obs: jnp.ndarray) -> Transition:
    """
    Appends a dummy terminal transition from the last observation to the batch.
    This acts as the frontier for LSTD-RMAX optimistic initialization.
    """
    dummy_transition = jax.tree_util.tree_map(
        lambda x: jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), 
        traj_batch
    )
    
    num_envs = last_obs.shape[0]
    dummy_transition = dummy_transition._replace(
        obs=last_obs[None, ...],
        done=jnp.ones((1, num_envs), dtype=traj_batch.done.dtype),
        action=jnp.zeros((1, num_envs), dtype=traj_batch.action.dtype) # Provide dummy action
    )
    
    padded_traj_batch = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0), 
        traj_batch, dummy_transition
    )
    
    return padded_traj_batch

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    calc_true_values = config.get('CALC_TRUE_VALUES', False)
    k = config.get('RND_FEATURES', 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    is_continuous = isinstance(env.action_space(env_params), spaces.Box)
    assert not is_continuous, "LSTD-Q over uniform random policy expects discrete actions"
    action_dim = env.action_space(env_params).n

    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR'])
    alpha_fn_lstd_b = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR_RI'])
    evaluator = None    

    if calc_true_values:
        if config['ENV_NAME'] == 'DeepSea-bsuite':
            evaluator = DeepSeaExactValue(size=config['DEEPSEA_SIZE'], unscaled_move_cost=0.01, gamma=config['GAMMA'], episodic=config['EPISODIC'])
        if config['ENV_NAME'] == 'Chain':
            evaluator = LongChainExactValue(config.get('CHAIN_LENGTH', 100), config['GAMMA'], config['EPISODIC'])

    if config['EPISODIC']: 
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
    else:
        assert False, 'only episodic!!!!! '
    
    def get_scale_free_bonus(S, features_sa):
        """bonus = x^T Sigma^{-1} X, based on State-Action features."""
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features_sa.shape[-1]), jnp.eye(features_sa.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features_sa, Sigma_inv, features_sa)
        return jnp.sqrt(bonus_sq)
    
    def single_cross_cov_q(phi_sa, expected_next_phi, done, gamma):
        target_phi = gamma * (1.0 - done) * expected_next_phi
        diff = phi_sa - target_phi
        return jnp.outer(phi_sa, diff)
    
    cross_cov_vmap = jax.vmap(jax.vmap(lambda p_sa, ep_next, d: single_cross_cov_q(p_sa, ep_next, d, config['GAMMA_i'])))


    def lstd_q_batch_update(lstd_state: Dict, transitions, phi_sa, expected_next_phi, lambda_s, rho_unscaled):
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']
        t = lstd_state['t']
        
        # This is the MBIE-EB bonus! It decays as N increases, so it IS safe for EMA.
        rho = transitions.intrinsic_reward

        is_weights = rho_unscaled ** 2 
        is_weights = is_weights / (is_weights.mean() + 1e-8)

        # ---------------------------------------------------------
        # 1. EMPIRICAL UPDATES (For persistent storage)
        # ---------------------------------------------------------
        # Standard A update
        A_update = cross_cov_vmap(phi_sa, expected_next_phi, transitions.done)
        A_update = A_update * is_weights[..., None, None]
        A_batch = A_update.mean(axis=batch_axes)
        
        # Standard b update (includes MBIE-EB intrinsic reward)
        b_sample = phi_sa * rho[..., None]
        b_sample = b_sample * is_weights[..., None]
        b_batch = b_sample.mean(axis=batch_axes)

        # Update Persistent EMA Storage (NO V_max injected here!)
        A_i = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], A_batch)
        b_i = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_batch)

        # ---------------------------------------------------------
        # 2. DYNAMIC OPTIMISTIC INITIALIZATION (At Solve Time)
        # ---------------------------------------------------------
        # To mimic RMAX globally for 0-visit states, we apply a prior.
        # This acts like a dataset of 1 sample uniformly distributing V_max across all features.
        prior_weight = config.get('OPTIMISM_PRIOR_WEIGHT', 1.0) # Analogous to lambda_s but global
        
        # Global Optimism Injection
        # A_solve = A_empirical + Prior_A
        # b_solve = b_empirical + Prior_b
        
        reg_A = jnp.eye(A_i.shape[0]) * config['A_REGULARIZATION_PER_STEP']
        
        # The prior assumes an isotropic feature space where every orthogonal direction 
        # initially points to V_max.
        optimistic_b_prior = jnp.ones_like(b_i) * (lstd_state['V_max'] / jnp.sqrt(k)) * prior_weight
        
        A_solve = A_i + reg_A + (jnp.eye(k) * prior_weight)
        b_solve = b_i + optimistic_b_prior

        # 3. Solve linear system
        w_i = jnp.linalg.solve(A_solve, b_solve)

        return {
            "A": A_i,
            "b": b_i,
            "w": w_i,
            "N": N,
            "t": t + 1,
            "V_max": lstd_state['V_max'], 
            "Beta": lstd_state['Beta'],
        }

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        
        # Initialize with action_dim attached
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, action_dim, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, action_dim, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        # This returns shape (..., action_dim, k)
        batch_get_features = jax.vmap(lambda obs: rnd_net.apply(target_params, obs))

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        initial_lstd_state = {
            'A': jnp.eye(k) * config['A_REGULARIZATION'], 
            'b': jnp.zeros(k), 'w': jnp.zeros(k),
            'N': 0, 't': 1, 'Beta': config['BONUS_SCALE'], 'V_max': 1/(1-config['GAMMA_i']),
        }
        initial_sigma_state = {'S': jnp.eye(k) * config['GRAM_REG'], 'N': 1, 't': 1, }
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                transition = Transition(
                    done,
                    action,
                    value,
                    i_val,
                    reward,
                    intrinsic_reward,
                    log_prob,
                    last_obs,
                    info["real_next_obs"],
                    info,
                )
                return (train_state, rnd_state, env_state, obsv, rng), transition
                
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # --- LSTD-Q Processing ---
            traj_batch_padded = append_frontier_transition(traj_batch, last_obs)
            
            # 1. Extract Full State-Action Feature Matrix (T+1, B, A, k)
            phi_matrix = batch_get_features(traj_batch_padded.obs)
            
            # 2. Extract Specific phi(s, a) that was taken (T+1, B, k)
            vmap_gather = jax.vmap(jax.vmap(lambda p, a: p[a]))
            phi_sa = vmap_gather(phi_matrix, traj_batch_padded.action)
            
            # 3. Expected Next Feature for Uniform Random Policy (T+1, B, k)
            expected_phi = phi_matrix.mean(axis=-2) 
            
            # Calculate intrinsic reward using the State-Action feature covariance
            rho = get_scale_free_bonus(sigma_state['S'], phi_sa[:-1])
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            
            # 4. Optimistic Initialization
            rho_current = get_scale_free_bonus(sigma_state['S'], phi_sa) / jnp.sqrt(sigma_state['N'])
            PRIOR_SAMPLES = config.get('LSTD_PRIOR_SAMPLES', 1.0)
            scaled_uncertainty = PRIOR_SAMPLES * (rho_current**2)
            lambda_s = scaled_uncertainty / (1.0 + scaled_uncertainty)
            
            # Update LSTD-Q
            # LSTD-Q expects the expected next phi based on the transition s -> s'
            next_phi_matrix = batch_get_features(traj_batch_padded.next_obs)
            expected_next_phi = next_phi_matrix.mean(axis=-2) 
            
            lstd_state = lstd_q_batch_update(lstd_state, traj_batch_padded, phi_sa, expected_next_phi, lambda_s, rho_current)

            # Compute Values for GAE. The Intrinsic *State* Value under a uniform random policy is expected_phi.
            _, last_val = network.apply(train_state.params, last_obs)
            
            v_i = expected_phi[:-1] @ lstd_state['w']
            last_i_val = expected_phi[-1] @ lstd_state['w']
            lstd_state['V_max'] = jnp.maximum(jnp.max(v_i), jnp.max(last_i_val)) 
            
            # Adaptive beta
            sqrt_n = jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))
            lstd_state['Beta'] = helpers.schedule_extrinsic_to_intrinsic_ratio(sigma_state['N'] / config['TOTAL_TIMESTEPS'], config['BONUS_SCALE'])
            rho_scale = lstd_state['Beta'] / sqrt_n

            v_i *= rho_scale
            last_i_val *= rho_scale
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho * rho_scale)

            # --- ADVANTAGE CALCULATION ---            
            gaes, targets = gae_fn(traj_batch, last_val, last_i_val, config["GAMMA"], config["GAE_LAMBDA"], config['GAE_LAMBDA_i'], config["GAMMA_i"])
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state
            
            # UPDATE Covariance explicitly using phi(s, a)
            alpha_cov = alpha_fn(sigma_state['t'])
            S_batch = jnp.mean(jnp.einsum('bti,btj->ij', phi_sa[:-1], phi_sa[:-1]), axis=(0,1))
            sigma_state['S'] = (1 - alpha_cov) * sigma_state['S'] + alpha_cov * S_batch
            sigma_state['N'] += traj_batch.done.size
            sigma_state['t'] += 1

            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            
            metric.update({
                "ppo_loss": loss_info[0], 
                "rnd_loss": loss_info[1],
                "feat_norm": jnp.linalg.norm(expected_next_phi, axis=-1).mean(),
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
                "rho_scale": rho_scale
            })

            if evaluator is None:
                metric.update({
                    "vi_pred": traj_batch.i_value.mean(),
                    "v_e_pred": traj_batch.value.mean()
                })
            else:
                def int_rew_from_state(s): 
                    # State reward represented as the average over actions
                    phi_mat = batch_get_features(s)
                    avg_bonus = jnp.mean(jax.vmap(lambda p_mat: get_scale_free_bonus(sigma_state['S'], p_mat))(phi_mat), axis=-1)
                    return avg_bonus * rho_scale
                
                def get_vi(obs):
                    # Random policy Intrinsic state value is expected_phi @ w
                    return batch_get_features(obs).mean(axis=-2) @ lstd_state['w'] * rho_scale 
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, lstd_state['Beta'], network, train_state, traj_batch, get_vi)

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