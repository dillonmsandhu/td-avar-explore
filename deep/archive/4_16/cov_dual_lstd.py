# Covariance-Based Intrinsic Reward, propagated by LSTD (Dual LSTD for Ve and Vi).
# Consolidated version: Handles both standard training and ExactValue logging via config.
#
# Feature pipeline: the LSTD basis phi(s) is selected via LSTD_FEATURE_TYPE:
#   - "rnd"        : random frozen CNN/MLP (default; legacy behavior)
#   - "tabular"    : one-hot from the agent-channel argmax (FourRooms only)
#   - "pretrained" : exact lookup into an offline DINOv2/etc. feature cache
#                    (PRETRAINED_CACHE_PATH .npz). FourRooms-only, keyed on
#                    (agent_idx, goal_idx).
# For this algorithm the intrinsic-reward features share the same phi; the
# separate RI_FEATURE_TYPE used in cov_lstd_vf is not exposed here.
from core.imports import *
import numpy as np
import core.helpers as helpers
import core.networks as networks
from envs.fourrooms_custom import FourRoomsExactValue
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
SAVE_DIR = 'cov_dual_lstd'


def _build_feature_pipeline(config, obs_shape, rng):
    """Return (get_features_fn, k) for cov_dual_lstd.

    get_features_fn(obs) -> phi   (no params dependency; frozen features).
    """
    use_bias = config.get("BIAS", True)
    feature_type = config.get("LSTD_FEATURE_TYPE", "rnd")

    if feature_type == "rnd":
        k = int(config.get("RND_FEATURES", 128))
        rnd_rng, target_rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape,
            config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"],
            config["BIAS"],
            k,
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape,
            config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"],
            config["BIAS"],
            k,
        )
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        return get_features_fn, k, rnd_net, rnd_params, target_params

    if feature_type == "tabular":
        env_name = config.get("ENV_NAME", "")
        if env_name not in {"FourRooms-misc", "FourRoomsCustom-v0"}:
            raise ValueError("Tabular features only supported for FourRooms envs.")
        room_size = int(config.get("FOURROOMS_SIZE", 21))
        k = room_size * room_size

        def get_features_fn(obs):
            if obs.ndim >= 3:
                agent_map = obs[..., 1]
                return agent_map.reshape(*agent_map.shape[:-2], k)
            pos = obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, room_size - 1)
            x = jnp.clip(pos[..., 1], 0, room_size - 1)
            idx = y * room_size + x
            return jax.nn.one_hot(idx, k, dtype=jnp.float32)

        # Still need a dummy rnd_net/params for train-state scaffolding downstream.
        rnd_rng, _ = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape,
            config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"],
            config["BIAS"],
            int(config.get("RND_FEATURES", 128)),
        )
        target_params = rnd_params
        return get_features_fn, k, rnd_net, rnd_params, target_params

    if feature_type == "pretrained":
        cache_path = config.get("PRETRAINED_CACHE_PATH")
        if cache_path is None:
            raise ValueError("PRETRAINED_CACHE_PATH must be set for 'pretrained' features")
        env_name = config.get("ENV_NAME", "")
        if env_name not in {"FourRooms-misc", "FourRoomsCustom-v0"}:
            raise ValueError("Pretrained features only supported for FourRooms envs.")

        data = np.load(cache_path)
        feats_np = np.asarray(data["features"], dtype=np.float32)
        obs_np = np.asarray(data["obs_stack"], dtype=np.float32)
        N, H, W, _ = obs_np.shape
        cells = H * W
        agent_keys = obs_np[..., 1].reshape(N, cells).argmax(axis=-1)
        goal_keys = obs_np[..., 2].reshape(N, cells).argmax(axis=-1)
        keys = agent_keys * cells + goal_keys
        lookup_np = np.zeros(cells * cells, dtype=np.int32)
        lookup_np[keys] = np.arange(N, dtype=np.int32)

        feature_table = jnp.asarray(feats_np)
        lookup = jnp.asarray(lookup_np)

        k_raw = int(feats_np.shape[-1])
        normalize = config.get("NORMALIZE_FEATURES", False)
        k = k_raw + (1 if use_bias else 0)

        def get_features_fn(obs):
            agent = jnp.argmax(obs[..., 1].reshape(*obs.shape[:-3], cells), axis=-1)
            goal = jnp.argmax(obs[..., 2].reshape(*obs.shape[:-3], cells), axis=-1)
            row = lookup[agent * cells + goal]
            phi = feature_table[row]
            if normalize:
                phi = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
            if use_bias:
                bias = jnp.ones((*phi.shape[:-1], 1))
                phi = jnp.concatenate([phi, bias], axis=-1)
            return phi

        rnd_rng, _ = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape,
            config["RND_NETWORK_TYPE"],
            config["NORMALIZE_FEATURES"],
            config["BIAS"],
            int(config.get("RND_FEATURES", 128)),
        )
        target_params = rnd_params
        return get_features_fn, k, rnd_net, rnd_params, target_params

    raise ValueError(f"Unknown LSTD_FEATURE_TYPE: {feature_type!r}")

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

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    calc_true_values = config.get('CALC_TRUE_VALUES', False)

    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape

    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    alpha_fn_lstd = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR'])
    alpha_fn_lstd_bi = helpers.get_alpha_schedule(config['ALPHA_SCHEDULE'], config['MIN_LSTD_LR_RI'])
    evaluator = None
    if calc_true_values:
        if config['ENV_NAME'] == 'DeepSea-bsuite':
            evaluator = DeepSeaExactValue(
                size=config['DEEPSEA_SIZE'], 
                unscaled_move_cost=0.01, 
                gamma=config['GAMMA'], 
                episodic=config['EPISODIC']
            )
        elif config["ENV_NAME"] in {"FourRooms-misc", "FourRoomsCustom-v0"}:
            goal_pos = config.get("FOURROOMS_GOAL_POS", None)
            if goal_pos is not None:
                goal_pos = tuple(goal_pos)
            evaluator = FourRoomsExactValue(
                size=int(config.get("FOURROOMS_SIZE", 13)),
                fail_prob=float(config.get("FOURROOMS_FAIL_PROB", 1.0 / 3.0)),
                gamma=config["GAMMA"],
                episodic=config["EPISODIC"],
                use_visual_obs=(config["NETWORK_TYPE"] == "cnn"),
                goal_pos=goal_pos,
            )
        elif config['ENV_NAME'] == 'Chain':
            evaluator = LongChainExactValue(config.get('CHAIN_LENGTH', 100), config['GAMMA'], config['EPISODIC'])
        else:
            raise ValueError(
                f"CALC_TRUE_VALUES=True is only supported for DeepSea/FourRooms/Chain. Got {config['ENV_NAME']}."
            )

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
        A_i_view = helpers.EMA(α,lstd_state['A_i'], A_batch_rmax) 
        
        # for storage (note having two LSTD A's is redundant - they should be the same.):
        A_i = helpers.EMA(α, lstd_state["A_i"], A_batch)
        A_e = helpers.EMA(α, lstd_state["A_e"], A_batch)
        
        ## feature-reward vector
        b_e_sample = traces * rew[..., None]
        b_e_batch = b_e_sample.mean(axis=batch_axes)

        b_i_sample = traces * rho[..., None]
        b_i_sample_view = (1-lambda_s)[..., None] * b_i_sample  + lambda_s[...,None] * lstd_state['V_max'] * features
        # b_i_sample_view = b_i_sample
        b_i_view = b_i_sample_view.mean(axis=batch_axes)
        b_i_view = helpers.EMA(alpha_fn_lstd_bi(t), lstd_state["b_i"], b_i_view)
        
        # for storage:
        b_i = helpers.EMA(alpha_fn_lstd_bi(t), lstd_state["b_i"], b_i_sample.mean(axis=batch_axes))
        b_e = helpers.EMA(α, lstd_state["b_e"], b_e_batch)

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
    def actor_only_loss(params, apply_fn, traj_batch, advantages):
        pi, _ = apply_fn(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        clip_eps = config["CLIP_EPS"]
        loss_actor1 = -ratio * advantages
        loss_actor2 = -jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()
        total_loss = loss_actor - config["ENT_COEF"] * entropy
        return total_loss, (loss_actor, entropy)

    def train(rng):
        feat_rng, rng = jax.random.split(rng)

        # Feature pipeline (routes via LSTD_FEATURE_TYPE): rnd, tabular, or pretrained.
        get_features_fn, k, rnd_net, rnd_params, target_params = _build_feature_pipeline(
            config, obs_shape, feat_rng
        )

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

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
        initial_sigma_state = {'S': jnp.eye(k) * config['GRAM_REG'], 'N': 1, 't': 1,}

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
                    done,
                    action,
                    ve,
                    vi,
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

            # --- 1. Compute intrinsic rewards in batch ---
            phi = batch_get_features(traj_batch.obs)
            next_phi = batch_get_features(traj_batch.next_obs)
            
            int_rew_from_features = lambda phi: get_scale_free_bonus(sigma_state['S'], phi) 
            # traj_batch, sigma_state, rho = helpers.update_cov_and_get_rho(traj_batch, sigma_state, batch_get_features, int_rew_from_features, alpha_fn)
            rho = int_rew_from_features(next_phi)
            rho = rho - rho.min()
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            
            # --- 4. Optimistic Initialization---
            std = int_rew_from_features(phi) / jnp.sqrt(sigma_state['N'])  # corresponds to estimated standard deviation of least squares estimate at phi (for example, least squares reward prediction)
            PRIOR_SAMPLES = config.get('LSTD_PRIOR_SAMPLES', 1.0)
            scaled_uncertainty = PRIOR_SAMPLES * (std**2)
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
            # lstd_state['Beta'] = helpers.update_beta(lstd_state['Beta'], v_i, traj_batch.value, progress = sigma_state['N'] / config['TOTAL_TIMESTEPS'], update=config['ADAPTIVE_BETA'])
            # lstd_state['Beta'] = helpers.schedule_extrinsic_to_intrinsic_ratio(sigma_state['N'] / config['TOTAL_TIMESTEPS'], config['BONUS_SCALE'])
            rho_scale = lstd_state['Beta'] / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))            
            
            # --- 4. ADVANTAGE CALCULATION ---             
            # scale the intrinsic value and reward before computing advantages.
            last_val_i *= rho_scale
            traj_batch = traj_batch._replace(value=v_e, i_value=v_i * rho_scale, intrinsic_reward=rho * rho_scale)
            # Compute GAEs
            gaes, targets = gae_fn(traj_batch,
                last_val_e,
                last_val_i,
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
            # jax.debug.print('sigma before update {sigma}', sigma = sigma_state['S'][0:10, 0])
            # Update sigma state:
            _, sigma_state, _ = helpers.update_cov_and_get_rho(traj_batch, sigma_state, batch_get_features, int_rew_from_features, alpha_fn)
            # jax.debug.print('sigma after update {sigma}', sigma = sigma_state['S'][0:10, 0])
            
            # METRICS ---
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
                    # rho = get_scale_free_bonus(sigma_state['S'], phi)  * rho_scale
                    rho = int_rew_from_features(phi) * rho_scale
                    # jax.debug.print('unscaled reward for s is {ri}', ri = int_rew_from_features(phi)[0] )
                    return rho
                
                get_vi = lambda obs: batch_get_features(obs) @ lstd_state['w_i'] * rho_scale 
                get_ve = lambda obs: batch_get_features(obs) @ lstd_state['w_e']
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, lstd_state['Beta'], network, train_state, traj_batch, get_vi, get_ve)
            # end metrics
            
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
