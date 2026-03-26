from core.utils import *
from core.helpers import _calculate_gae, _get_all_traces, Explore_Transition, _loss_fn
import core.helpers as helpers
import core.networks as networks
SAVE_DIR = 'lstd_avar'

def compute_bonus(features, lstd_state, config=None):
    """Computes the state bonus using the sandwich covariance"""
    Sandwich = lstd_state['Sandwich']
    variance_per_state = jnp.einsum('...i,ij,...j->...', features, Sandwich, features)
    B = config['BONUS_SCALE'] * jnp.sqrt(jnp.maximum(variance_per_state, 1e-6))
    return B

def lstd_batch_update( 
                    lstd_state: Dict,
                    transitions, # Explore_Transition
                    features: jnp.ndarray,
                    next_features: jnp.ndarray,
                    traces: jnp.ndarray,
                    config: Dict,
                    α_A: float, 
                    α_S: float,
    ):
    
    # Fix 1: Add 'current_features' as argument
    def lstd(traces, current_features, next_features, transition):
        # Now current_features is (k,) thanks to vmap
        td_features = current_features - config['GAMMA_i'] * (1 - transition.done) * next_features
        # A += z * (φ - γφ')^T
        A_sample = jnp.outer(traces, td_features)
        return A_sample
        
    def per_sample_OPG(z, td_error):
        return jnp.outer(z, z) * (td_error**2)

    # Unpack state (Assuming these are RAW uncorrected EMAs)
    A, S, t = lstd_state['A'], lstd_state['S'], lstd_state['t']
    batch_axes = tuple(range(transitions.done.ndim))
    N = transitions.done.size
    # Fix 1: Pass features to the vmap
    # Vmap structure: (L, B, k) -> we map over axis 0 (L) then axis 1 (B)
    batch_lstd = jax.vmap(jax.vmap(lstd))
    
    # A_update will be (L, B, k, k)
    A_update = batch_lstd(traces, features, next_features, transitions)
    S_update = jax.vmap(jax.vmap(per_sample_OPG))(traces, transitions.td_error)

    # batch averages
    A_b, S_b = jax.tree.map(lambda x: x.mean(axis=batch_axes), (A_update, S_update))
    S_b = 0.5 * (S_b + S_b.T)  # immediately symmetrize for numerical stability

    # EMA
    A = (1-α_A) * A + α_A * A_b
    S = (1-α_S) * S + α_S * S_b
    
    A_view = A + config['A_REGULARIZATION_PER_STEP'] * jnp.eye(A.shape[0])

    # effective sample size of EMA is 2/alpha
    A_inv = jnp.linalg.solve(A_view, jnp.eye(A.shape[0]))
    N_eff = 2.0 / α_A
    cov_w = (1 / N_eff) * (A_inv @ S @ A_inv.T)

    # return {'A': A_view, 'S': S, 'N': N, 't': t+1, 'Sandwich': cov_w, 'A_update': A_b, 'S_update': S_b}
    return {'A': A, 'S': S, 'N': N + lstd_state['N'], 't': t+1, 'Sandwich': cov_w, 'A_update': A_b, 'S_update': S_b}


def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    k = config.get('RND_FEATURES', 128)
    env, env_params = helpers.make_env(config)
    
    obs_shape = env.observation_space(env_params).shape
    # EMA set up:
    GET_ALPHA_FN = lambda t: jnp.maximum(1/10, 1/t)
    prior_t = config["PRIOR_N"] // batch_size

    def compute_sandwich(lstd_state: Dict, α = 1):
        "Computes the sandwich covariance from LSTD state"
        A, S = lstd_state['A'], lstd_state['S']
        A_view = A + config['A_REGULARIZATION_PER_STEP'] * jnp.eye(A.shape[0])
        A_inv = jnp.linalg.pinv(A_view)
        cov_w = (α/2) * (A_inv @ S @ A_inv.T)
        return cov_w

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], k)
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        max_r = 1.0
        
        initial_lstd_state = {
            'A': jnp.eye(k) * config['A_REGULARIZATION'],  # Regularization for numerical stability
            'S': jnp.eye(k) * max_r**2,
            'N': 0, # number of samples
            't': 1, # number of updates
            'A_update': jnp.zeros((k, k)),
            'S_update': jnp.zeros((k, k)),
        }
        initial_lstd_state['Sandwich']= compute_sandwich(initial_lstd_state)
        
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)

        get_features_fn = lambda params, obs: rnd_net.apply(params, obs)
        batch_get_features = jax.vmap(get_features_fn, in_axes=(None, 0))
        
        get_v_features_fn = lambda params, obs: train_state.apply_fn(params, obs, method="get_value_features")[0]
        batch_get_v_features = jax.vmap(get_v_features_fn, in_axes=(None, 0))

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # WARMUP:
        def _warmup_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            
            # Select RANDOM actions (no need for network here, just exploring state space)
            rng, _rng = jax.random.split(rng)
            rng_action = jax.random.split(_rng, config["NUM_ENVS"])
            action = jax.vmap(env.action_space(env_params).sample)(rng_action)
            
            # Step env (wrappers will update their internal mean/std stats automatically)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                rng_step, env_state, action, env_params
            )
            
            return (env_state, obsv, rng), None

        # Run the warmup
        warmup_runner_state = (env_state, obsv, rng)
        (env_state, obsv, rng), _ = jax.lax.scan(
            _warmup_step, warmup_runner_state, None, config["WARMUP"]
        )
        # -------------------------

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng= env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                # Record
                next_pi, next_value = network.apply(train_state.params, obsv)
                δ = reward + config['GAMMA'] * (1-done) * next_value - value
                # δ = reward + config['GAMMA']  * next_value - value
                # δ = jnp.ones_like(reward)

                intrinsic_reward = jnp.zeros_like(reward)
                target_embedding = jnp.zeros_like(reward)
                transition = Explore_Transition(
                    done, action, value, value, reward, intrinsic_reward, log_prob, last_obs, obsv, target_embedding, δ, info
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            # COMPUTE BONUS:
            next_phi = batch_get_features(rnd_state.target_params, traj_batch.next_obs)
            U_prime = compute_bonus(next_phi, lstd_state, config)
            # CALCULATE ADVANTAGE
            _, last_val = network.apply(train_state.params, last_obs)

            gae, targets = _calculate_gae(traj_batch, last_val, config['GAMMA'], config['GAE_LAMBDA'])
            # optimistic advantage! the next state value is uncertainty - so we add a bonus proportional to its stddev
            advantages = gae + config['GAMMA'] * U_prime * (1 - traj_batch.done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                
                def _update_minbatch(minibatch_input, batch_info):
                    train_state, rnd_state, mask_rng = minibatch_input
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    rnd_loss = jnp.zeros_like(total_loss)                    
                    return (train_state, rnd_state, mask_rng), (total_loss, rnd_loss)
                # end update_minibatch

                train_state, rnd_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                rng, mask_rng = jax.random.split(rng)
                (train_state, rnd_state, mask_rng), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, rnd_state, mask_rng), minibatches
                )
                update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # end update_epoch

            # --------- Train the network ---------
            update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["NUM_EPOCHS"]
            )
            train_state, rnd_state, _, _, _, rng = update_state
            # -------------------------------
            # --------- Update LSTD ---------
            new_phi = batch_get_features(rnd_state.target_params, traj_batch.obs)
            new_phi_prime = batch_get_features(rnd_state.target_params, traj_batch.next_obs)
            traces = _get_all_traces(traj_batch, new_phi, config['GAMMA'], config['GAE_LAMBDA'], γi = config["GAMMA_i"])
            lstd_state = lstd_batch_update(
                lstd_state,
                traj_batch,
                new_phi,
                new_phi_prime,
                traces,
                config,
                α_A=GET_ALPHA_FN(lstd_state['t']),
                α_S=GET_ALPHA_FN(lstd_state['t'] + prior_t),
            )
            # -------------------------------
            # --------- Update metrics ------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }            
            # constant obs:
            constant_obs = jnp.zeros_like(traj_batch.obs)
            target_features_const_obs = rnd_net.apply(rnd_state.target_params, constant_obs)
            avg_targ_feat_const_obs = jnp.linalg.norm(target_features_const_obs,axis=-1).mean()

            metric.update({
                "ppo_loss": loss_info[0], 
                "rnd_loss": loss_info[1],
                "bonus_mean": U_prime.mean(),
                "bonus_std": U_prime.std(),
                "bonus_max": U_prime.max(),
                "implied_count": 1.0 / (jnp.square(U_prime).mean() + 1e-8),
                "gae_mean": gae.mean(),
                "gae_std":  gae.std(),
                "gae_max":  gae.max(),
                "lambda_ret_mean": targets.mean(),
                "lambda_ret_std": targets.std(),
                "td_error_mean":traj_batch.td_error.mean(),
                "td_error_std": traj_batch.td_error.std(),
                "avg_targ_feat_const_obs": avg_targ_feat_const_obs,
                "mean_rew": traj_batch.reward.mean(),
            })
            runner_state = (train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric
            # end update_step

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
    
if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)