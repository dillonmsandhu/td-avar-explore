# LSPI-RMAX for Intrinsic Value
# Evaluates the Optimal Policy for the intrinsic reward using off-policy PPO data.
# Based on Li et al. (2009) "Online Exploration in Least-Squares Policy Iteration"
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
from gymnax.environments import spaces
SAVE_DIR = '3_15_lspi_optimal'

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
    dummy_transition = jax.tree_util.tree_map(
        lambda x: jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), 
        traj_batch
    )
    num_envs = last_obs.shape[0]
    dummy_transition = dummy_transition._replace(
        obs=last_obs[None, ...],
        done=jnp.ones((1, num_envs), dtype=traj_batch.done.dtype),
        action=jnp.zeros((1, num_envs), dtype=traj_batch.action.dtype)
    )
    return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), traj_batch, dummy_transition)

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    k = config.get('RND_FEATURES', 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    is_continuous = isinstance(env.action_space(env_params), spaces.Box)
    assert not is_continuous, "LSPI for discrete actions only in this implementation"
    n_actions = env.action_space(env_params).n

    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR'])
    alpha_fn_lstd_b = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR_RI'])
    
    evaluator = None    
    if config.get('CALC_TRUE_VALUES', False):
        if config['ENV_NAME'] == 'DeepSea-bsuite':
            evaluator = DeepSeaExactValue(size=config['DEEPSEA_SIZE'], unscaled_move_cost=0.01, gamma=config['GAMMA'], episodic=config['EPISODIC'])
        if config['ENV_NAME'] == 'Chain':
            evaluator = LongChainExactValue(config.get('CHAIN_LENGTH', 100), config['GAMMA'], config['EPISODIC'])

    gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
    
    def get_scale_free_bonus(S, phi_sa):
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(phi_sa.shape[-1]), jnp.eye(phi_sa.shape[-1]))
        return jnp.sqrt(jnp.einsum('...i,ij,...j->...', phi_sa, Sigma_inv, phi_sa))

    def lspi_iteration(lstd_state, phi_sa, phi_mat_next, transitions, lambda_s, rho_unscaled, lambda_next_s):
        """
        One LSTD-Q iteration solving for the greedy policy.
        lambda_next_s: novelity of the next state (used for RMAX logic).
        """
        batch_axes = tuple(range(transitions.done.ndim))
        gamma = config['GAMMA_i']
        V_max = lstd_state['V_max']

        # 1. Policy Improvement: Find greedy action for next state under current w
        # Q(s', a') = phi_mat_next @ w
        q_next = jnp.einsum('btak,k->bta', phi_mat_next, lstd_state['w'])
        best_action_idx = jnp.argmax(q_next, axis=-1)
        
        # phi(s', a*)
        phi_next_greedy = jax.vmap(jax.vmap(lambda p_mat, a: p_mat[a]))(phi_mat_next, best_action_idx)

        # 2. RMAX Logic (Algorithm 1 in Paper)
        # s' is unknown with probability lambda_next_s.
        # If unknown, target is r + gamma * V_max.
        # If known, target is r + gamma * Q(s', a*).
        
        # Calculate target phi: (1-done) * gamma * [(1 - lambda_next_s) * phi_next_greedy]
        # The V_max part is added to the b vector.
        phi_target_base = (1.0 - transitions.done[..., None]) * gamma * (1.0 - lambda_next_s[..., None]) * phi_next_greedy
        
        # A matrix: sum phi_sa * (phi_sa - phi_target_base)^T
        A_update_raw = jax.vmap(jax.vmap(lambda p, tp: jnp.outer(p, p - tp)))(phi_sa, phi_target_base)
        
        # IS weights for covariance matching
        is_weights = (rho_unscaled ** 2) / (jnp.mean(rho_unscaled ** 2) + 1e-8)
        A_update = A_update_raw * is_weights[..., None, None]
        
        # 3. Optimism: Current (s,a) is unknown with prob lambda_s. 
        # Target for unknown (s,a) is V_max.
        lambda_s = jnp.clip(lambda_s, 0.0, 1.0)
        A_i_batch = (A_update * (1 - lambda_s)[..., None, None]).mean(axis=batch_axes)
        weighted_gram_batch = jnp.einsum('bt, bti, btj->ij', lambda_s * is_weights, phi_sa, phi_sa) / transitions.done.size
        A_final = A_i_batch + weighted_gram_batch
        
        # 4. Reward Vector (b)
        # Target = Reward + gamma * (Value if known else V_max)
        # If (s,a) is unknown, Value = V_max
        r_int = transitions.intrinsic_reward
        
        # Next state value injection: gamma * V_max * lambda_next_s (if not done)
        v_next_rmax = gamma * (1.0 - transitions.done) * lambda_next_s * V_max
        b_target = (r_int + v_next_rmax) * is_weights
        
        # (s,a) is unknown: push toward V_max
        b_i_sample = phi_sa * b_target[..., None]
        b_i_opt = ((1 - lambda_s)[..., None] * b_i_sample + (lambda_s * is_weights)[..., None] * V_max * phi_sa).mean(axis=batch_axes)
        
        # Solve
        reg = config['A_REGULARIZATION_PER_STEP'] * jnp.eye(k)
        w_new = jnp.linalg.solve(A_final + reg, b_i_opt)
        
        # We only return the update matrices and the new w for the NEXT iteration.
        # The EMA will be applied after the loop.
        return A_update.mean(axis=batch_axes), b_i_sample.mean(axis=batch_axes), w_new

    def train(rng):
        rnd_rng, target_rng, ac_rng, rng = jax.random.split(rng, 4)
        
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k, 
            state_action_features=True, n_actions=n_actions
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k,
            state_action_features=True, n_actions=n_actions
        )
            
        network, network_params = networks.initialize_actor_critic(ac_rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        batch_get_phi_matrix = jax.vmap(lambda obs: rnd_net.apply(target_params, obs))

        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rng, config["NUM_ENVS"]), env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        initial_lstd_state = {'A': jnp.eye(k) * config['A_REGULARIZATION'], 'b': jnp.zeros(k), 'w': jnp.zeros(k), 'N': 0, 't': 1, 'Beta': config['BONUS_SCALE'], 'V_max': 1/(1-config['GAMMA_i'])}
        initial_sigma_state = {'S': jnp.eye(k) * config['GRAM_REG'], 'N': 1, 't': 1}
        
        def _update_step(runner_state, unused):
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            def _env_step(env_scan_state, unused):
                ts, rs, es, lo, r = env_scan_state
                pi, v = network.apply(ts.params, lo)
                r, r_step = jax.random.split(r)
                act = pi.sample(seed=r_step)
                o, es, rew, d, info = jax.vmap(env.step, in_axes=(0,0,0,None))(jax.random.split(r, config["NUM_ENVS"]), es, act, env_params)
                return (ts, rs, es, o, r), Transition(d, act, v, jnp.zeros_like(rew), rew, jnp.zeros_like(rew), pi.log_prob(act), lo, o, info)

            (train_state, rnd_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, (train_state, rnd_state, env_state, last_obs, rng), None, config["NUM_STEPS"])
            
            traj_batch_padded = append_frontier_transition(traj_batch, last_obs)
            
            # Features
            phi_mat = batch_get_phi_matrix(traj_batch_padded.obs)
            phi_sa = jax.vmap(jax.vmap(lambda p, a: p[a]))(phi_mat, traj_batch_padded.action)
            phi_mat_next = batch_get_phi_matrix(traj_batch_padded.next_obs)
            
            # 1. Uncertainty Estimation
            # For optimism, we need novelty of (s,a) and novelty of s'
            rho_curr_sa = get_scale_free_bonus(sigma_state['S'], phi_sa)
            # Novelty of next state is the average novelty of its actions (or max)
            # Li et al suggests a state is known if all actions are known. We'll use the mean precision logic.
            rho_next_s = jnp.mean(jax.vmap(jax.vmap(lambda pm: get_scale_free_bonus(sigma_state['S'], pm)))(phi_mat_next), axis=-1)
            
            rho_curr_unscaled = rho_curr_sa / jnp.sqrt(sigma_state['N'])
            rho_next_unscaled = rho_next_s / jnp.sqrt(sigma_state['N'])
            
            M = config.get('LSTD_PRIOR_SAMPLES', 1.0)
            lambda_s = (M * rho_curr_unscaled**2) / (1.0 + M * rho_curr_unscaled**2)
            lambda_next_s = (M * rho_next_unscaled**2) / (1.0 + M * rho_next_unscaled**2)

            # 2. LSPI Inner Loop
            # We solve for the optimal intrinsic value function using current batch.
            def _lspi_loop(cur_lstd, _):
                A_step, b_step, w_next = lspi_iteration(cur_lstd, phi_sa, phi_mat_next, traj_batch_padded, lambda_s, rho_curr_unscaled, lambda_next_s)
                return {**cur_lstd, 'w': w_next}, (A_step, b_step)
            
            final_lstd, (A_history, b_history) = jax.lax.scan(_lspi_loop, lstd_state, None, config.get('LSPI_ITERATIONS', 5))
            
            # 3. EMA Updates for the global LSTD state (using the average transition statistics from the loop)
            t = lstd_state['t']
            A_avg = A_history.mean(axis=0)
            b_avg = b_history.mean(axis=0)
            
            lstd_state = {
                'A': helpers.EMA(alpha_fn_lstd(t), lstd_state['A'], A_avg),
                'b': helpers.EMA(alpha_fn_lstd_b(t), lstd_state['b'], b_avg),
                'w': final_lstd['w'],
                'N': lstd_state['N'] + traj_batch.done.size,
                't': t + 1,
                'Beta': helpers.schedule_extrinsic_to_intrinsic_ratio(sigma_state['N'] / config['TOTAL_TIMESTEPS'], config['BONUS_SCALE']),
                'V_max': final_lstd['V_max']
            }

            # 4. Values for GAE
            # Intrinsic State Value V(s) = max_a Q(s,a)
            v_i_all_actions = jnp.einsum('btak,k->bta', phi_mat[:-1], lstd_state['w'])
            v_i = jnp.max(v_i_all_actions, axis=-1)
            
            last_phi_mat = phi_mat[-1]
            last_i_val = jnp.max(last_phi_mat @ lstd_state['w'], axis=-1)
            lstd_state['V_max'] = jnp.maximum(lstd_state['V_max'], jnp.maximum(v_i.max(), last_i_val.max()))
            
            rho_scale = lstd_state['Beta'] / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))
            traj_batch = traj_batch._replace(i_value=v_i * rho_scale, intrinsic_reward=rho_curr_sa[:-1] * rho_scale)

            # 5. PPO Update
            gaes, targets = gae_fn(traj_batch, network.apply(train_state.params, last_obs)[1], last_i_val * rho_scale, config["GAMMA"], config["GAE_LAMBDA"], config['GAE_LAMBDA_i'], config["GAMMA_i"])
            
            def _update_epoch(update_state, unused):
                ts, tb, adv, tar, r = update_state
                r, r_shuffle = jax.random.split(r)
                minibatches = helpers.shuffle_and_batch(r_shuffle, (tb, adv, tar), config["NUM_MINIBATCHES"])
                ts, losses = jax.lax.scan(lambda s, b: (s.apply_gradients(grads=jax.grad(helpers._loss_fn, has_aux=True)(s.params, network, *b, config)[1]), jax.grad(helpers._loss_fn)(s.params, network, *b, config)), ts, minibatches)
                return (ts, tb, adv, tar, r), losses

            (train_state, _, _, _, rng), _ = jax.lax.scan(_update_epoch, (train_state, traj_batch, gaes[0]+gaes[1], targets[0], rng), None, config["NUM_EPOCHS"])
            
            # 6. Update Covariance
            alpha_cov = alpha_fn(sigma_state['t'])
            S_batch = jnp.mean(jnp.einsum('bti,btj->ij', phi_sa[:-1], phi_sa[:-1]), axis=(0,1))
            sigma_state.update({'S': (1-alpha_cov)*sigma_state['S'] + alpha_cov*S_batch, 'N': sigma_state['N']+traj_batch.done.size, 't': sigma_state['t']+1})

            # Metrics

            metrics = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metrics.update ( {"intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(), "beta": lstd_state['Beta'], "lambda_s_mean": lambda_s.mean()} )

            if evaluator:
                def int_rew_fn(s):
                    phi_m = batch_get_phi_matrix(s)
                    return jnp.mean(jax.vmap(lambda pm: get_scale_free_bonus(sigma_state['S'], pm))(phi_m), -1) * rho_scale
                def get_vi_fn(o):
                    return jnp.max(batch_get_phi_matrix(o) @ lstd_state['w'], axis=-1) * rho_scale
                metrics = helpers.add_values_to_metric(config, metrics, int_rew_fn, evaluator, config['BONUS_SCALE'], network, train_state, traj_batch, get_vi_fn)

            return (train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx+1), metrics

        runner_state, metrics = jax.lax.scan(_update_step, (train_state, initial_lstd_state, initial_sigma_state, rnd_state, env_state, obsv, rng, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)