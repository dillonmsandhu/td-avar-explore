# Covariance-Based Intrinsic Reward, propagated by an intrinsic value net
# handles absorbing
# intrinsic reward has infintie memory
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
SAVE_DIR = '4_2_cov_net'

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray 
    next_value: jnp.ndarray       # Added for absorbing GAE
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray       # Added for absorbing GAE
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray 
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

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

    # Scan natively over traj_batch
    def _get_advantages(gae_accs, transition):
        gae, i_gae = gae_accs
        
        done = transition.done 
        i_boot_mult = 1.0 - (done * i_mask_boot)
        i_gae_mult  = 1.0 - (done * i_mask_gae)

        # Pull next values directly from the transition tuple!
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
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    k = config.get('RND_FEATURES', 128)
    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    calc_true_values = config.get('CALC_TRUE_VALUES', False)
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        """bonus = x^T Σ^{-1} X, where Σ^{-1} is the empriical second moment inverse."""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    
    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=3)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

        initial_sigma_state = {'S': jnp.eye(k, dtype=jnp.float64),}
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        # batch_get_features = jax.vmap(get_features_fn)

        def batch_get_features(obs): # Scan version.
                if obs.ndim == len(obs_shape) + 2:  # Trajectory Batch [T, B, ...]
                    def scan_fn(carry, obs_step):
                        return None, rnd_net.apply(target_params, obs_step)
                    _, out = jax.lax.scan(scan_fn, None, obs)
                    return out
                return rnd_net.apply(target_params, obs) # Standard Batch [B, ..
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        # -------------------------

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, sigma_state, rnd_state, env_state, last_obs, beta, rng, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value, i_val = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                # GET NEXT VALUES FOR UNIFIED GAE
                true_next_obs = info["real_next_obs"]
                _, next_val, next_i_val = network.apply(train_state.params, true_next_obs)

                # Record
                intrinsic_reward = jnp.zeros_like(reward)  # placeholder, will be filled later
                transition = Transition(
                    done,
                    action,
                    value,
                    next_val,    # Passed cleanly
                    i_val,
                    next_i_val,  # Passed cleanly
                    reward,
                    intrinsic_reward,
                    log_prob,
                    last_obs,
                    true_next_obs,
                    info,
                )

                env_scan_state = (train_state, rnd_state, env_state, obsv, rng)
                return env_scan_state, transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # -------------------------------------------------------------
            # --------- Update Sigma and compute intrinsic reward ---------
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) 
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k))
            
            int_rew_from_features = lambda phi: get_scale_free_bonus(Sigma_inv, phi)

            rho = int_rew_from_features(batch_get_features(traj_batch.next_obs))
            traj_batch = traj_batch._replace(intrinsic_reward=rho)
            
            # --- THE ABSORBING TERMINAL FIX ---
            # The value network never trains on the exact terminal state, making its prediction garbage.
            # In the absorbing formulation, we mathematically know the infinite horizon value of the 
            # terminal state is exactly its intrinsic reward / (1 - gamma_i).
            exact_terminal_i_val = rho / (1.0 - config["GAMMA_i"])
            fixed_next_i_val = jnp.where(
                jnp.logical_and(traj_batch.done, is_absorbing), 
                exact_terminal_i_val, 
                traj_batch.next_i_val
            )
            # Update the batch with the unscaled rho and the fixed terminal values
            traj_batch = traj_batch._replace(
                intrinsic_reward=rho,
                next_i_val=fixed_next_i_val
            )            

            # -------------------------------------------------------------
            # --------- ADVANTAGE CALCULATION (Unified Absorbing) ---------
            gaes, targets = calculate_gae(
                traj_batch, 
                config["GAMMA"], 
                config["GAE_LAMBDA"], 
                is_episodic=is_episodic, 
                is_absorbing=is_absorbing, 
                γi=config["GAMMA_i"], 
                λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            
            rho_scale = config['BONUS_SCALE']
            # Total Advantage = Adv_Extrinsic + (Beta * Adv_Intrinsic)
            advantages = gae_e + (rho_scale * gae_i) # scale the gae.

            # UPDATE NETWORK
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses
            
            # --------- Train the network ---------
            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state

            # UPDATE Covariance
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
            # -------------------------------
            # --------- Update metrics ------

            # The value net is trained on an unscaled rho
            # but implicitly the value prediction is scaled by rho
            # this also makes it comperable to the oracle 

            scaled_reward = traj_batch.intrinsic_reward * rho_scale
            scaled_i_val = traj_batch.i_value * rho_scale
            
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
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

            if evaluator is None: # No way to compute true values, just record the batch average prediction.
                metric.update({
                "vi_pred": scaled_i_val.mean(),
                "v_e_pred": traj_batch.value.mean()
            })
            else:
                def int_rew_from_state(s):
                    phi = batch_get_features(s)
                    rho = int_rew_from_features(phi) * rho_scale
                    return rho
                
                metric = helpers.add_values_to_metric(config, 
                                                    metric, 
                                                    int_rew_from_state, 
                                                    evaluator, 
                                                    beta, 
                                                    network, 
                                                    train_state, 
                                                    traj_batch, 
                                                    rho_scale=rho_scale)
                
            runner_state = (train_state, sigma_state, rnd_state, env_state, last_obs, beta, rng, idx+1)
            return runner_state, metric
            # end update_step

        rng, _rng = jax.random.split(rng)
        init_runner_state = (train_state, initial_sigma_state, rnd_state, env_state, obsv, config['BONUS_SCALE'], _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, init_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
