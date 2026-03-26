# RND: RND intrinsic reward propagated by a value net.
from core.utils import * 
import core.helpers as helpers
import core.networks as networks
import flax
SAVE_DIR = 'rnd'

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

# --- Running Mean Std Helper ---
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
    config.setdefault("GAMMA_i", 0.99)

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] 
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    k = config.get('RND_FEATURES', 128)
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    
    if config['EPISODIC']: 
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic_episodic
    else:
        gae_fn = helpers.calculate_gae_intrinsic_and_extrinsic
    
    def train(rng):
        # initialize rnd networks
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(rnd_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config['RND_NETWORK_TYPE'], config['NORMALIZE_FEATURES'], config['BIAS'], k)
        
        # initialize value and policy network
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=3)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, rnd_params, target_params)
        
        # INIT Running Statistics for Intrinsic Return (not just reward)
        rnd_ret_rms = RunningMeanStd()
        
        # NEW: Initialize the running return for the filter (RewardForwardFilter state)
        # This tracks the discounted sum of future intrinsic rewards
        rnd_return = jnp.zeros(config["NUM_ENVS"])

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack new state variable `rnd_return`
                train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value, i_value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                # --- RND Logic (MSE + Filter + Normalization) ---
                # Clip obs for RND to prevent outliers
                rnd_obs = jnp.clip(obsv, -5, 5)
                
                # 1. Calculate Raw Intrinsic Reward (MSE)
                target_embedding = rnd_state.apply_fn(rnd_state.target_params, rnd_obs)
                target_embedding = jax.lax.stop_gradient(target_embedding)
                pred = rnd_state.apply_fn(rnd_state.params, rnd_obs)
                intrinsic_reward_raw = jnp.mean((pred - target_embedding)**2, axis=-1)
                
                # 2. Update Reward Forward Filter
                # R_t = R_{t-1} * gamma + r_t
                rnd_return = rnd_return * config["GAMMA_i"] + intrinsic_reward_raw
                
                # 3. Update RMS with the RETURN, not the raw reward
                rnd_ret_rms = rnd_ret_rms.update(rnd_return)
                
                # 4. Normalize the RAW REWARD by the STD of the RETURN
                intrinsic_reward = intrinsic_reward_raw / (jnp.sqrt(rnd_ret_rms.var) + 1e-8)                

                transition = Transition(
                    done, action, value, i_value, reward, intrinsic_reward, log_prob, last_obs, info["real_next_obs"], target_embedding, info
                )
                
                # Pack updated `rnd_return` back into state
                runner_state = (train_state, rnd_state, env_state, obsv, rng, rnd_ret_rms, rnd_return)
                return runner_state, transition
            
            # unpack runner state including `rnd_return`
            train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return, idx = runner_state
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return)
            (train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )

            # scaling for advanrtage
            N = idx *traj_batch.done.shape[0] * traj_batch.done.shape[1]
            beta = helpers.schedule_extrinsic_to_intrinsic_ratio( N / config['TOTAL_TIMESTEPS'], config['BONUS_SCALE'])
            beta = jnp.where(config['ADAPTIVE_BETA'], beta, config['BONUS_SCALE'])
            
            # CALCULATE ADVANTAGE
            _, last_val, last_i_val = network.apply(train_state.params, last_obs)

            gaes, targets = gae_fn(traj_batch, last_val, last_i_val, config["GAMMA"], config["GAE_LAMBDA"], γi = config["GAMMA_i"])
            # Combine advantages
            advantages = gaes[0] + beta * gaes[1] 

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(minibatch_input, batch_info):
                    train_state, rnd_state, mask_rng = minibatch_input
                    traj_batch, advantages, targets = batch_info
                    
                    def rnd_loss_fn(rnd_params, target_embeddings, mask, obs):
                        # Clip obs again here for safety/consistency
                        rnd_obs = jnp.clip(obs, -5, 5)
                        pred = rnd_net.apply(rnd_params, rnd_obs)
                        
                        diff = pred - target_embeddings
                        losses = jnp.mean(jnp.square(diff), axis=-1)
                        
                        loss = (losses * mask).sum() / (mask.sum() + 1e-8)
                        return loss, _

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value, i_val = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        targets, i_targets = targets
                        
                        # Extrinsic VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["VF_CLIP"], config["VF_CLIP"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        
                        # Intrinsic VALUE LOSS
                        value_pred_clipped = traj_batch.i_value + (
                            i_val - traj_batch.i_value
                        ).clip(-config["VF_CLIP"], config["VF_CLIP"])
                        value_losses = jnp.square(i_val - i_targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - i_targets)
                        i_value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            + config["VF_COEF"] * i_value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (i_value_loss, value_loss, loss_actor, entropy)

                    # --- UPDATE PPO ---
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    # --- UPDATE RND ---
                    rnd_grad_fn = jax.value_and_grad(rnd_loss_fn, has_aux=True)
                    mask = jax.random.bernoulli(mask_rng, p=config['RND_TRAIN_FRAC'], shape=(traj_batch.obs.shape[0],))
                    
                    (rnd_loss, _), rnd_grads = rnd_grad_fn(rnd_state.params, traj_batch.embedding, mask, traj_batch.next_obs)
                    rnd_state = rnd_state.apply_gradients(grads=rnd_grads)
                    
                    return (train_state, rnd_state, mask_rng), (total_loss, rnd_loss)

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

            update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["NUM_EPOCHS"]
            )
            train_state, rnd_state, _, _, _, rng = update_state
            
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }            
            metric.update({
                "ppo_loss": loss_info[0].mean(), 
                "rnd_loss": loss_info[1].mean(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                # Note: raw_intrinsic_rew estimate here uses the updated var, but it's an approximation for logging
                "raw_intrinsic_rew_mean": traj_batch.intrinsic_reward.mean()  * jnp.sqrt(rnd_ret_rms.var) + 1e-8,
                "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                "rnd_return_mean_est": rnd_ret_rms.mean,
                "rnd_return_std_est": jnp.sqrt(rnd_ret_rms.var),
                "bonus_mean": gaes[1].mean(),
                "bonus_std": gaes[1].std(),
                "bonus_max": gaes[1].max(),
                "vi_pred": traj_batch.i_value.mean(),
                "v_e_pred": traj_batch.value.mean()
            })
            # Pass new state (rms and rnd_return) forward
            runner_state = (train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, rnd_return, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        
        # Initialize runner_state with the new rnd_return variable
        runner_state = (train_state, rnd_state, env_state, obsv, _rng, rnd_ret_rms, rnd_return, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

<<<<<<< HEAD:deep/algos/rnd.py
=======
def main():
    import warnings; warnings.simplefilter('ignore')
    import os
    from utils import save_results, save_plot, parse_config_override
    import datetime
    import argparse
    import configs

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD Explore experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values, e.g. \'{"LR": 0.001, "LAMBDA": 0.0}\'')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help='saves to rnd/{args.run_suffix}' )
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--base-config', type = str, default = 'mc', choices = ['mc', 'ds'])
    args = parser.parse_args()
    
    if args.base_config == 'mc':
        config = configs.mc_config.copy()
        raise AssertionError('conv_net_v.py only has value solver implemented for DeepSea')
    elif args.base_config == 'ds':
        config = configs.ds_config.copy()
    elif args.base_config  == 'min':
        config = configs.min_config.copy()

    # Override with command line config
    config_override = parse_config_override(args.config)
    config.update(config_override)
    rng = jax.random.PRNGKey(config['SEED'])
        
    def evaluate(config, rng):
        steps_per_pi = config["NUM_ENVS"]*config["NUM_STEPS"]
        run_fn = jax.jit(jax.vmap(make_train(config)))
        rngs = jax.random.split(rng, config['N_SEEDS'])
        out = run_fn(rngs)
        metrics = out["metrics"]

        print("Mean return is " , jnp.mean(metrics['returned_episode_returns']))
        print("(Mean) Max return is " , jnp.max(metrics['returned_episode_returns']))

        run_dir = os.path.join("results", f"rnd/{args.run_suffix}")
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
        
        ia_mean = metrics['ia_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['ia_mean']
        i_mean = metrics['intrinsic_rew_mean'].mean(0) if config['N_SEEDS'] > 1 else metrics['intrinsic_rew_mean']
        rnd_loss = metrics['rnd_loss'].mean(0) if config['N_SEEDS'] > 1 else metrics['rnd_loss']

        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, ia_mean, 'Intrinsic_Adv')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, i_mean, 'Intrinsic_Rew')
        save_plot(env_dir, config['ENV_NAME'], steps_per_pi, rnd_loss, 'rnd_loss')

    
    evaluate(config, rng)

>>>>>>> main:deep/rnd.py
if __name__ == '__main__':
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)