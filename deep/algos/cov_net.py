# Covariance-Based Intrinsic Reward, propegated by an intrinsic value net
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from envs.deepsea_v import DeepSeaExactValue
from envs.long_chain import LongChainExactValue
SAVE_DIR = 'cov_net'

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
    
def make_train(config):
    assert config.get('EPISODIC', True)
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    k = config.get('RND_FEATURES', 128)
    alpha_fn = lambda t: jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
    calc_true_values = config.get('CALC_TRUE_VALUES', False)
    evaluator = None
    if calc_true_values and config['ENV_NAME'] == 'DeepSea-bsuite':
        evaluator = DeepSeaExactValue(
            size=config['DEEPSEA_SIZE'], unscaled_move_cost=0.01, gamma=config['GAMMA'], episodic=config['EPISODIC']
        )
    if calc_true_values and config['ENV_NAME'] == 'Chain':
        evaluator = LongChainExactValue(config.get('CHAIN_LENGTH', 100), config['GAMMA'], config['EPISODIC'])

    def get_scale_free_bonus(S, features):
        """bonus = x^T Sigma^{-1} X, where Sigma^{-1} is the empriical second moment inverse."""
        Sigma_inv = jnp.linalg.solve(S + config['GRAM_REG'] * jnp.eye(features.shape[-1]), jnp.eye(features.shape[-1]))
        bonus_sq = jnp.einsum('...i,ij,...j->...', features, Sigma_inv, features)
        return jnp.sqrt(bonus_sq)
    
    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
            
        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=3)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

        initial_sigma_state = {
            'S': jnp.eye(k) * config['GRAM_REG'],
            'N': 0, # number of samples
            't': 1, # number of updates
        }

        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)
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
                train_state, rnd_state, env_state, last_obs, rng= env_scan_state

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
                
                # Record
                intrinsic_reward = jnp.zeros_like(reward)  # placeholder, will be filled later
                transition = Transition(
                    done, action, value, i_val, reward, intrinsic_reward, log_prob, last_obs, obsv, info, 
                )

                env_scan_state = (train_state, rnd_state, env_state, obsv, rng)
                return env_scan_state, transition
            
            initial_obs = last_obs # shape: (NUM_ENVS, *obs_shape)
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # -------------------------------------------------------------
            # --------- Update Sigma and compute intrinsic reward ---------
            int_rew_from_features = lambda phi: get_scale_free_bonus(sigma_state['S'], phi) 
            traj_batch, sigma_state, rho = helpers.update_cov_and_get_rho(traj_batch, sigma_state, batch_get_features, int_rew_from_features, alpha_fn)
            
            # Scale rho
            beta = helpers.update_beta(beta, 
                               traj_batch.i_value, 
                               traj_batch.value, 
                               progress = sigma_state['N'] / config['TOTAL_TIMESTEPS'], 
                               update=config['ADAPTIVE_BETA']
            )
            # Final scale of r_i is unscaled rho times 1/sqrt(N) times beta
            rho_scale = beta / jnp.maximum(1.0, jnp.sqrt(sigma_state['N']))

            # Advantage
            _, last_val, last_i_val = network.apply(train_state.params, last_obs)
            # Extrinsic Advantage
            gae_e, target_e = helpers._calculate_gae(traj_batch, last_val, config["GAMMA"], config["GAE_LAMBDA"])
            # B. Intrinsic GAE (Unit Scale)
            gae_i, target_i = helpers._calculate_gae(traj_batch, last_i_val, config["GAMMA_i"], config["GAE_LAMBDA_i"])
            
            # C. Combine using Beta
            # Total Advantage = Adv_Extrinsic + (Beta * Adv_Intrinsic)
            targets = (target_e, rho_scale * target_i) 
            advantages = gae_e + (rho_scale * gae_i)

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
            # -------------------------------
            # --------- Update metrics ------

            scaled_reward = traj_batch.intrinsic_reward * rho_scale
            scaled_i_val = traj_batch.i_value * rho_scale
            
            metric = {k: v.mean() for k, v in traj_batch.info.items()} # performance

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
                
                metric = helpers.add_values_to_metric(config, metric, int_rew_from_state, evaluator, beta, network, train_state, traj_batch)
                
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
