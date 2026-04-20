import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax
import distrax
# from utils import batch_evaluate
from utils import shuffle_and_batch
from typing import Sequence, NamedTuple, Any
import wandb
import envpool
from networks import ActorCriticNet
from atari_wrappers import JaxLogEnvPoolWrapper
import sv_ppo_helpers as helpers
import jax.tree_util as jax_tree # downgraded jax version for env-pool

# PPO
def make_train(config):

    class Transition(NamedTuple):
        done: jnp.ndarray
        action: jnp.ndarray
        value: jnp.ndarray
        reward: jnp.ndarray
        log_prob: jnp.ndarray
        log_probs: jnp.ndarray
        obs: jnp.ndarray
        info: jnp.ndarray
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"] 
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"] # per epoch
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    total_grad_steps = config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
    
    def make_env(num_envs):
        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            num_threads=config['THREADS'],
            **config["ENV_KWARGS"],
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env
    
    env = make_env(config["NUM_ENVS"])
    n_actions = env.single_action_space.n
    obsv, env_state = env.reset()

    def train(rng):
        network = ActorCriticNet(
            action_dim=n_actions,
            norm_type= 'layer_norm' if config["LAYER_NORM"] else None,
        )
        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.single_observation_space.shape), dtype = jnp.float32)
        network_params = network.init(init_rng, init_x)
        
        lr = optax.linear_schedule( # decays during FQE
                init_value=config["LR"],
                end_value=config.get('LR_END', 1e-15),
                transition_steps= total_grad_steps
        )
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(learning_rate=lr, eps=1e-4, weight_decay=1e-4),
        )
        class PPO_FQE_TrainState(TrainState):
            adv_rms: helpers.RunningMeanStd  # <--- Added here

            def policy(self, x):
                return self.apply_fn(self.params, x, method=network.policy)  
        
        # Initialize RMS for normalizing the advantatge
        initial_adv_rms = helpers.RunningMeanStd(
            mean=jnp.array(0.0), 
            var=jnp.array(1.0), 
            count=jnp.array(1e-4)
        )
        train_state = PPO_FQE_TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            adv_rms=initial_adv_rms 
        )
        init_k = config['CONSECUTIVE_STABLE_ITERS']
        fraction_to_one = config.get('K_DECAY_FRACTION', 0.0) # e.g., Hit 1.0 at 50% of training
        
        if fraction_to_one > 0 and init_k > 0: # decay to 1 by K_DECAY_FRACTION of learning
            decay_rate = (1.0 / init_k) ** (1.0 / fraction_to_one)        # Formula: rate = (1/init)^(1/fraction)
            k_schedule = optax.exponential_decay(
                init_value = config['CONSECUTIVE_STABLE_ITERS'],
                end_value = 1.0,
                decay_rate = decay_rate,
                transition_steps=config["NUM_UPDATES"]
            )
        else: 
            k_schedule = lambda t: init_k # constant
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
              
            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng= env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                b, value = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                # all_log_probs = jnp.log(pi.probs())
                log_probs = jax.nn.log_softmax(b.logits)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)

                transition = Transition(
                    done, action, value, reward, log_prob, log_probs, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            # end env_step
            
            pi, train_state, env_state, last_obs, rng, idx, stable_count, pi_idx, pi_num = runner_state
            env_step_state = (train_state, env_state, last_obs, rng)
            (train_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            # CALCULATE ADVANTAGE
            # 1. Get last value (scaled)
            _, last_val_scaled = network.apply(train_state.params, last_obs)
            
            # 2. INVERSE Scale values to REAL domain for GAE
            #    We need real rewards and real values to calculate accurate advantages
            real_traj_values = helpers.muzero_inv_rescale(traj_batch.value)
            real_last_val = helpers.muzero_inv_rescale(last_val_scaled)
            
            # Create a temporary batch with real values for the GAE helper
            # (assuming helpers.get_vtrace_and_gae uses .value attribute)
            real_value_traj_batch = traj_batch._replace(value=real_traj_values)

            # 3. Compute Advantages and Targets (Returns) in REAL space
            advantages, real_targets, target_all_logits = helpers.get_vtrace_and_gae(
                real_value_traj_batch, real_last_val, pi, config['GAMMA'], config['GAE_LAMBDA'],
                min_retrace_ratio = config['MIN_IS_RATIO'], max_retrace_ratio = config['MAX_IS_RATIO']
            )            

            # scale targets
            targets = helpers.muzero_rescale(real_targets)
            # scale advantage
            adv_rms = helpers.update_rms(train_state.adv_rms, advantages)
            advantages = (advantages - adv_rms.mean) / (jnp.sqrt(adv_rms.var) + 1e-5)
            train_state = train_state.replace(adv_rms = adv_rms)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    total_loss, grads = jax.value_and_grad(helpers._loss_fn, has_aux=True)(train_state.params, network, traj_batch, advantages, targets, config)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                minibatches = jax_tree.tree_map(lambda x: shuffle_and_batch(_rng, x, config["NUM_MINIBATCHES"]), (traj_batch, advantages, targets))
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state            
            
            # convergence check
            # Run inference (scan over steps to save memory)
            # def _predict_step(carry, step_obs):
            #     pi, v = network.apply(train_state.params, step_obs)
            #     return carry, (pi.logits, v)

            # Scan over the first dimension (NUM_STEPS)
            # new_behavioral, new_v = jax.vmap()
            new_behavioral, new_v = jax.vmap(network.apply, in_axes=(None,0))(train_state.params, traj_batch.obs)
            # _, (new_logits, new_v) = jax.lax.scan(_predict_step, None, traj_batch.obs)
            # new_behavioral = distrax.Categorical(logits=new_logits)

            update_pi, stable_count, pi_idx, pi_num, v_diff, return_scale, required_k = helpers.determine_convergence(
                new_v,traj_batch, targets, stable_count, pi_idx, pi_num, idx, config, k_schedule
            )
            # Record Metrics
            metric = {k: v.mean() for k, v in traj_batch.info.items()} 
            metric = helpers.compute_sv_metrics( 
                metric, traj_batch, new_behavioral, target_all_logits, update_pi, pi_idx, pi_num, v_diff, return_scale, required_k, targets, new_v, loss_info[1]
            )
            metric['adv_rms_mean'] = train_state.adv_rms.mean
            metric['adv_rms_std'] = jnp.sqrt(train_state.adv_rms.var)
            # (Conditionally) Update the policy, and continue to next iteration
            pi = jax.lax.cond(update_pi, lambda _ : train_state, lambda _: pi, None)
            runner_state = (pi, train_state, env_state, last_obs, rng, idx+1, stable_count, pi_idx, pi_num)

            return runner_state, metric
        # end update_step

        rng, _rng = jax.random.split(rng)
        initial_runner_state = (train_state, train_state, env_state, obsv, _rng, 0, 0, 0, 1)
        runner_state, metrics = jax.lax.scan(
            _update_step, initial_runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}
    # end train

    return train

if __name__ == '__main__':
    import warnings; warnings.simplefilter('ignore')
    import argparse
    import os
    import time
    import jax
    import jax.numpy as jnp
    from utils import save_plot, save_results, load_config_dict

    parser = argparse.ArgumentParser(description="Run SV PPO algorithm")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--envs', type=str, nargs='*', default=[], help="List of environment names to run")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--run-suffix', type=str, default="")
    # Tuner specific arguments
    parser.add_argument('--output-dir', type=str, default=None, help="Force output directory")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default="sv_ppo_tuning")
    
    args = parser.parse_args()
    
    # --- 1. Robust Config Loading ---
    if os.path.isfile(args.config):
        config_path = args.config
    else:
        config_path = os.path.join('configs/sv_ppo/', args.config)
        
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config_dict(config_path)
    
    # --- 2. Apply Overrides ---
    config['ALG'] = 'SV_PPO'
    config['CONFIG'] = os.path.basename(args.config)
    config["NUM_SEEDS"] = args.num_seeds
    config["SEED"] = args.seed
    config["THREADS"] = args.threads
    
    if args.wandb:
        import wandb
        wandb.init(project=args.project, config=config, name=args.run_suffix, group=config['CONFIG'])

    print("Loaded configuration:", config)

    # --- 3. Directory Setup ---
    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join("results", f"sv_ppo/{args.run_suffix}")

    print(f"Saving all results to {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    envs_to_run = args.envs if args.envs else [config['ENV_NAME']]
    
    # Run the algorithm for each environment
    for env_name in envs_to_run:
        print(f"Running environment: {env_name} with Seed {args.seed}")
        config['ENV_NAME'] = env_name
        rng = jax.random.PRNGKey(args.seed)
        
        steps_per_pi = config["NUM_ENVS"] * config["NUM_STEPS"]

        # Run Training
        out = jax.jit(make_train(config))(rng)
        
        # Metrics Processing
        metrics = out["metrics"]
        print(f"Mean return: {jnp.mean(metrics['returned_episode_returns']):.2f}")
        print(f"Max return: {jnp.max(metrics['returned_episode_returns']):.2f}")
        
        # Save results
        env_dir = os.path.join(run_dir, env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        save_plot(env_dir, env_name, steps_per_pi, metrics['returned_episode_returns'])
        
        save_results(out, config, env_name, env_dir)
