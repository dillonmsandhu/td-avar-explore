# Covariance-Based Intrinsic Reward, propegated by LSTD.
# Consolidated version: Handles both standard training and ExactValue logging via config.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
import flax
SAVE_DIR = 'cov_lstd_std'

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

# --- Running Mean Std Helper (Added locally for completeness) ---
@flax.struct.dataclass
class RunningMeanStd:
    mean: jnp.ndarray = jnp.array(0.0)
    var: jnp.ndarray = jnp.array(1.0)
    count: jnp.ndarray = jnp.array(1e-4)

    def update(self, x):
        batch_mean = jnp.mean(x)
        batch_var = jnp.var(x)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        return self.replace(mean=new_mean, var=new_var, count=tot_count)

def make_train(config):
    # Ensure INT_GAMMA is set (default usually 0.99)
    config.setdefault("INT_GAMMA", 0.99)
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    # --- Flag to enable heavy exact value calculation ---
    calc_true_values = config.get('CALC_TRUE_VALUES', False)
    k = config.get('RND_FEATURES', 128)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR'])
    alpha_fn_lstd_b = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR_RI'])
    evaluator = None    
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
        trace_fn = helpers._get_all_traces 
        cross_cov = lambda z, phi, phi_prime, done: helpers.cross_cov(z, phi, phi_prime, done, config['GAMMA_i'])
    else:
        assert False, 'only episodic!!!!! '
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
        # transitions.intrinsic_reward is now the NORMALIZED reward
        rho = transitions.intrinsic_reward

        # ------------------------------------------------------------
        # 2. Standard LSTD A update (policy evaluation)
        # ------------------------------------------------------------
        A_update = jax.vmap(jax.vmap(cross_cov))(
            traces, features, next_features, transitions.done
        )
        lambda_s = jnp.clip(lambda_s, 0.0, 1.0) 
        A_i_update = A_update * (1-lambda_s)[..., None, None]

        A_batch = A_update.mean(axis=batch_axes)
        A_i_batch = A_i_update.mean(axis=batch_axes)
        weighted_gram_batch = jnp.einsum('bt, bti, btj->ij', lambda_s, features, features) / transitions.done.size
        A_batch_rmax = A_i_batch + weighted_gram_batch
        A_i_view = helpers.EMA(alpha_fn_lstd(t),lstd_state['A'], A_batch_rmax) 
        
        # for storage (exclude novelty-based optimsim):
        A_i = helpers.EMA(alpha_fn_lstd(t), lstd_state["A"], A_batch)
        
        ## feature-reward vector
        b_i_sample = traces * rho[..., None]
        # Optimistic bias uses V_max (which will adapt to the normalized scale)
        b_i_sample_opt = (1-lambda_s)[..., None] * b_i_sample  + lambda_s[...,None] * lstd_state['V_max'] * features
        b_i_opt = b_i_sample_opt.mean(axis=batch_axes)
        # ema of stored (non-optimistic) b and optimistic b.
        b_i_view = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_i_opt)
        
        # for storage:
        b_i = helpers.EMA(alpha_fn_lstd_b(t), lstd_state["b"], b_i_sample.mean(axis=batch_axes))

        # ------------------------------------------------------------
        # 6. Solve linear systems
        # ------------------------------------------------------------
        reg = jnp.eye(A_batch.shape[0]) * config['A_REGULARIZATION_PER_STEP']
        w_i = jnp.linalg.solve(A_i_view + reg, b_i_view)

        return {
            "A": A_i,"b": b_i,"w": w_i,"N": N,"t": t + 1,"V_max": lstd_state['V_max'], "Beta": lstd_state['Beta'],
        }


    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        # INIT Running Statistics for Intrinsic Return
        rnd_ret_rms = RunningMeanStd()
        # INIT Reward Forward Filter state
        rnd_return = jnp.zeros(config["NUM_ENVS"])

        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

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
            
            # Added rnd_ret_rms and rnd_return to state
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return, idx = runner_state
            
            # COLLECT TRAJECTORIES
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
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            initial_obs = last_obs 
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # --- Post-Processing: Compute & Normalize Intrinsic Rewards ---
            int_rew_from_features = lambda phi: get_scale_free_bonus(sigma_state['S'], phi) 
            rho_raw = int_rew_from_features(batch_get_features(traj_batch.next_obs))
            rho_raw = rho_raw - rho_raw.min()
            
            # NEW: Run Reward Forward Filter on the batch
            def filter_scan_fn(carry, r_t):
                # R_t = R_{t-1} * gamma + r_t
                new_ret = carry * config["INT_GAMMA"] + r_t
                return new_ret, new_ret
            
            # Scan across time [NUM_STEPS, NUM_ENVS]
            rnd_return, returns_batch = jax.lax.scan(filter_scan_fn, rnd_return, rho_raw)
            
            # Update RMS with discounted returns
            rnd_ret_rms = rnd_ret_rms.update(returns_batch.reshape(-1))
            
            # Normalize rho
            rho_norm = rho_raw / (jnp.sqrt(rnd_ret_rms.var) + 1e-8)
            
            # Update trajectory batch with NORMALIZED reward
            traj_batch = traj_batch._replace(intrinsic_reward=rho_norm)

            # --- 4. Optimistic Initialization ---
            # NOTE: We use UNNORMALIZED rho for lambda_s to preserve 'uncertainty' interpretation
            # relative to the prior samples.
            phi = batch_get_features(traj_batch.obs)
            rho_current = int_rew_from_features(phi) / jnp.sqrt(sigma_state['N']) 
            PRIOR_SAMPLES = config.get('LSTD_PRIOR_SAMPLES', 1.0)
            scaled_uncertainty = PRIOR_SAMPLES * (rho_current**2)
            lambda_s = scaled_uncertainty / (1.0 + scaled_uncertainty)
            
            # Update LSTD (uses traj_batch.intrinsic_reward which is now normalized)
            traces = trace_fn(traj_batch, phi, config['GAMMA_i'], config['GAE_LAMBDA_i'])
            next_phi = batch_get_features(traj_batch.next_obs)
            lstd_state = lstd_batch_update(lstd_state, traj_batch, phi, next_phi, traces, lambda_s)

            # Compute Values for GAE.
            _, last_val = network.apply(train_state.params, last_obs)
            last_phi = batch_get_features(last_obs)
            
            # v_i now estimates the discounted NORMALIZED reward
            v_i = phi @ lstd_state['w']
            last_i_val = last_phi @ lstd_state['w']
            lstd_state['V_max'] = jnp.maximum(jnp.max(v_i), jnp.max(last_i_val)) 
            
            # Scale Factor
            lstd_state['Beta'] = helpers.schedule_extrinsic_to_intrinsic_ratio(sigma_state['N'] / config['TOTAL_TIMESTEPS'], config['BONUS_SCALE'])
            
            # CHANGE: Removed 1/sqrt(N) scaling from reward path because we are using RMS normalization.
            # rho_norm is already unit variance. We just apply Beta.
            rho_scale = lstd_state['Beta'] 

            # Scale vi and ri for GAE
            v_i *= rho_scale
            last_i_val *= rho_scale
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho_norm * rho_scale)

            # --- 4. ADVANTAGE CALCULATION ---            
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
            
            # UPDATE Covariance
            _, sigma_state, _ = helpers.update_cov_and_get_rho(traj_batch, sigma_state, batch_get_features, int_rew_from_features, alpha_fn)

            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }            
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
                "beta": lstd_state['Beta'],
                "rho_scale": rho_scale,
                "rnd_ret_mean": rnd_ret_rms.mean,
                "rnd_ret_std": jnp.sqrt(rnd_ret_rms.var),
            })

            if evaluator is None: 
                metric.update({
                "vi_pred": traj_batch.i_value.mean(),
                "v_e_pred": traj_batch.value.mean()
            })
            else:
                def int_rew_from_state(s): 
                    phi = batch_get_features(s)
                    # Note: Approximation for metrics. True history-dependent filter impossible here.
                    rho = int_rew_from_features(phi) / (jnp.sqrt(rnd_ret_rms.var) + 1e-8) * rho_scale
                    return rho
                
                def get_vi(obs):
                    return batch_get_features(obs) @ lstd_state['w'] * rho_scale 
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, lstd_state['Beta'], network, train_state, traj_batch, get_vi)

            runner_state = (train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, rnd_state, env_state, obsv, _rng, rnd_ret_rms, rnd_return, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
    
if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)