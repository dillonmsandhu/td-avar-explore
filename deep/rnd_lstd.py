from utils import * 
import helpers
from envs.sparse_mc import SparseMountainCar
import flax
DEFAULT_CONFIG = {
    # "ENV_NAME": "SparseMountainCar-v0",
    "ENV_NAME": "DeepSea-bsuite",
    "LR": 5e-4,
    "LR_END": 5e-4,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 250_000,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.6,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.003,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "NORMALIZE_REWARDS": False,
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,
    "SEED": 42,
    "WARMUP": 200, # warmup steps for running mean/std
    "N_SEEDS": 4,
    # LSTD specific:
    "REGULARIZATION": 1e-4,
    "PER_UPDATE_REGULARIZATION": 1e-4,
    "RND_TRAIN_FRAC": 0.5,
    # deepsea
    "DEEPSEA_SIZE": 20,
}
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
    embedding: jnp.ndarray 
    info: jnp.ndarray

def lstd_i_val(phi_fn, obs, lstd_state):
    """
    phi: (..., k)
    returns: (...)
    """
    features = phi_fn(obs)
    return features @ lstd_state["w_int"]

def lstd_batch_update( 
                    lstd_state: Dict,
                    transitions, # Explore_Transition
                    features: jnp.ndarray,
                    next_features: jnp.ndarray,
                    traces: jnp.ndarray,
                    config: Dict,
                    α: float,
                    α_b: float
    ):
    
    def lstd(traces, features, next_features, done):
        td_features = features - config['GAMMA'] * (1 - done) * next_features
        # A += z * (φ - γφ')^T
        A_sample = jnp.outer(traces, td_features)
        return A_sample
    
    # Unpack state
    A, t = lstd_state['A'], lstd_state['t']
    batch_axes = tuple(range(transitions.done.ndim))
    N = transitions.done.size + lstd_state['N']  # total number of samples seen so far
    batch_lstd = jax.vmap(jax.vmap(lstd))
    
    # A_update will be (L, B, k, k)
    A_update = batch_lstd(traces, features, next_features, transitions.done)

    # Batch average
    A_b = A_update.mean(axis=batch_axes) + config['PER_UPDATE_REGULARIZATION'] * jnp.eye(A.shape[0])
    
    # EMA
    A = (1-α) * A + α * A_b    
    # bias correction
    # bc = 1.0 - (1.0 - α)**t
    # bc = jnp.maximum(bc, 1e-6)
    bc = 1.0
    A_view = A / bc

    # solve LSTD for intrinsic system A^{-1} x = b
    b_int_sample = traces * transitions.intrinsic_reward[..., None]
    b_b = b_int_sample.mean(axis=batch_axes)
    b_new = (1-α_b) * lstd_state['b_int'] + α_b * b_b
    bc = jnp.maximum(1.0 - (1.0 - α_b)**t, 1e-6)
    b_view = b_new / bc
    w_int = jnp.linalg.solve(A_view, b_view)
    
    return {'A': A, 'b_int': b_new, 'w_int': w_int, 'N': N, 't': t+1}

# --- Running Mean Std Helper ---
@flax.struct.dataclass
class RunningMeanStd:
    # keeps tra
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

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] 
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    GET_ALPHA_FN = lambda t: jnp.maximum(1/10, 1/t)
    ALPHA_B = 1/2

    def train(rng):
        # initialize rnd networks
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = initialize_rnd_network(rnd_rng, obs_shape, config)
        _, target_params = initialize_rnd_network(target_rng, obs_shape, config)
            
        # initialize value and policy network
        network, network_params = initialize_actor_critic(rng, obs_shape, n_actions, config, n_heads=2)

        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(config['LR'], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        rnd_state = RNDTrainState.create(
            apply_fn=rnd_net.apply,
            params=rnd_params,
            tx=tx,
            target_params=target_params,
        )
        
        # INIT Running Statistics for Intrinsic Reward
        rnd_ret_rms = RunningMeanStd()
        # Helper to get embedding
        get_rnd_features = lambda obs: rnd_state.apply_fn(target_params, obs)
        batch_get_features = jax.vmap(get_rnd_features)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # WARMUP (Standard env warmup, no changes needed here)
        def _warmup_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            rng, _rng = jax.random.split(rng)
            rng_action = jax.random.split(_rng, config["NUM_ENVS"])
            action = jax.vmap(env.action_space(env_params).sample)(rng_action)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                rng_step, env_state, action, env_params
            )
            return (env_state, obsv, rng), None

        warmup_runner_state = (env_state, obsv, rng)
        (env_state, obsv, rng), _ = jax.lax.scan(
            _warmup_step, warmup_runner_state, None, config["WARMUP"]
        )
        dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
        dummy_phi = rnd_net.apply(target_params, dummy_obs)
        k = dummy_phi.shape[-1]
        initial_lstd_state = {
            'A': jnp.eye(k) * config['REGULARIZATION'],  # Regularization for numerical stability
            'b_int': jnp.zeros(k), 
            'w_int': jnp.zeros(k),
            'N': 0, # number of samples
            't': 1, # number of updates
        }

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # unpack runner state
            train_state, rnd_state, lstd_state, env_state, last_obs, rng, rnd_ret_rms, idx = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                i_val = lstd_i_val(get_rnd_features, jnp.clip(last_obs, -5, 5), lstd_state)
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                # RND Prediction / Intrinsic Reward
                rnd_obs = jnp.clip(obsv, -5, 5)
                target_embedding = rnd_state.apply_fn(target_params, rnd_obs)
                target_embedding = jax.lax.stop_gradient(target_embedding) # Explicit stop grad
                pred = rnd_state.apply_fn(rnd_state.params, rnd_obs)
                # Normalize intrinsic reward
                intrinsic_reward_raw = jnp.mean((pred - target_embedding)**2, axis=-1)
                rnd_ret_rms = rnd_ret_rms.update(intrinsic_reward_raw)
                intrinsic_reward = intrinsic_reward_raw / (jnp.sqrt(rnd_ret_rms.var) + 1e-8)                
                # intrinsic value (dot product with LSTD weights)
                
                transition = Transition(
                    done, action, value, i_val, reward, intrinsic_reward, log_prob, last_obs, obsv, target_embedding, info
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng, rnd_ret_rms)
                return runner_state, transition
            

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms)
            (train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            # Advantage
            _, last_val = network.apply(train_state.params, last_obs)
            last_i_val = lstd_i_val(get_rnd_features, last_obs, lstd_state)
            gaes, targets = helpers.calculate_gae_intrinsic_and_extrinsic(traj_batch, last_val, last_i_val, config["GAMMA"], config["GAE_LAMBDA"])
            advantages = gaes[0] + gaes[1]
            extrinsic_target = targets[0]
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(minibatch_input, batch_info):
                    train_state, rnd_state, mask_rng = minibatch_input
                    traj_batch, advantages, targets = batch_info
                    
                    def rnd_loss_fn(rnd_params, target_embeddings, mask, obs):
                        # Clip obs again here for safety/consistency
                        rnd_obs = jnp.clip(obs, -5, 5)
                        pred = rnd_net.apply(rnd_params, rnd_obs)
                        losses = jnp.mean((pred - target_embeddings)**2, axis=-1)
                        loss = (losses * mask).sum() / (mask.sum() + 1e-8)
                        return loss, _

                    # --- UPDATE PPO ---
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    # --- UPDATE RND ---
                    rnd_grad_fn = jax.value_and_grad(rnd_loss_fn, has_aux=True)
                    mask = jax.random.bernoulli(mask_rng, p=config['RND_TRAIN_FRAC'], shape=(traj_batch.obs.shape[0],))
                    
                    (rnd_loss, _), rnd_grads = rnd_grad_fn(rnd_state.params, traj_batch.embedding, mask, traj_batch.obs)
                    rnd_state = rnd_state.apply_gradients(grads=rnd_grads)
                    
                    return (train_state, rnd_state, mask_rng), (total_loss, rnd_loss)
                # end _update_minbatch

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
            # end _update_epoch
            
            # Train PPO and RND Networks
            initial_update_state = (train_state, rnd_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, rnd_state, _, _, _, rng = update_state
            
            # LSTD update:
            new_phi = batch_get_features(traj_batch.obs)
            new_phi_prime = batch_get_features(traj_batch.next_obs)
            traces = helpers._get_all_traces(traj_batch, new_phi, config['GAMMA'], config['GAE_LAMBDA'])
            lstd_state = lstd_batch_update(
                lstd_state,
                traj_batch,
                new_phi,
                new_phi_prime,
                traces,
                config,
                α=GET_ALPHA_FN(lstd_state['t']),
                α_b=ALPHA_B,
            )
            # Metrics
            metric = {k: v.mean() for k, v in traj_batch.info.items()} 
            
            metric.update({
                "ppo_loss": loss_info[0].mean(), 
                "rnd_loss": loss_info[1].mean(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "raw_intrinsic_rew_mean": traj_batch.intrinsic_reward.mean()  * jnp.sqrt(rnd_ret_rms.var) + 1e-8,
                "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                "rnd_return_mean_est": rnd_ret_rms.mean,
                "rnd_return_std_est": jnp.sqrt(rnd_ret_rms.var),
                "intrinsc_adv_mean": gaes[1].mean(),
            })
            runner_state = (train_state, rnd_state, lstd_state, env_state, last_obs, rng, rnd_ret_rms, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, rnd_state, initial_lstd_state, env_state, obsv, _rng, rnd_ret_rms, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def main():
    import warnings; warnings.simplefilter('ignore')
    import os
    from utils import save_results, save_plot, parse_config_override
    import datetime
    import argparse
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD Explore experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values, e.g. \'{"LR": 0.001, "LAMBDA": 0.0}\'')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help='saves to rnd_lstd/{args.run_suffix}' )
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--save-checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Override with command line config
    config_override = parse_config_override(args.config)
    config.update(config_override)
    rng = jax.random.PRNGKey(config['SEED'])
    # update the network type and learning rate based on the env.
    config = resolve_env_config(config)
        
    def evaluate(config, rng):
        steps_per_pi = config["NUM_ENVS"]*config["NUM_STEPS"]
        run_fn = jax.jit(jax.vmap(make_train(config)))
        rngs = jax.random.split(rng, config['N_SEEDS'])
        out = run_fn(rngs)
        metrics = out["metrics"]

        print("Mean return is " , jnp.mean(metrics['returned_episode_returns']))
        print("(Mean) Max return is " , jnp.max(metrics['returned_episode_returns']))

        run_dir = os.path.join("results", f"rnd_lstd/{args.run_suffix}")
        env_dir = os.path.join(run_dir, config['ENV_NAME'])
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(env_dir, exist_ok=True)
        print(f"Saving {config['ENV_NAME']} results to {run_dir}")

        if args.save_checkpoint:
            save_results(out, config, config['ENV_NAME'], env_dir)
        else:
            save_results(metrics, config, config['ENV_NAME'], env_dir)

        mean_rets = metrics['returned_episode_returns'].mean(0) if config['N_SEEDS'] > 1 else metrics['returned_episode_returns']
        if config['ENV_NAME'] == "SparseMountainCar-v0":
            mean_rets = metrics['returned_discounted_episode_returns'].mean(0) if config['N_SEEDS'] > 1 else metrics['returned_discounted_episode_returns']
        
        intrinsc_adv_mean = metrics['intrinsc_adv_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['ia_mean']
        i_mean = metrics['intrinsic_rew_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['intrinsic_rew_mean']
        rnd_loss = metrics['rnd_loss'].mean(0) if config['N_SEEDS'] > 1 else metrics['rnd_loss']

        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, intrinsc_adv_mean, 'Intrinsic_Adv')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, i_mean, 'Intrinsic_Rew')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, rnd_loss, 'rnd_loss')
        mean_return = float(jnp.mean(metrics['returned_episode_returns']))
        print(f"RESULT mean_return={mean_return}")
    
    evaluate(config, rng)

if __name__ == '__main__':
    main()