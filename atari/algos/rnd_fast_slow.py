# RND exactly as the original paper.
# 1. The observation for RND is the last frame of the *next* observation, which is normalized and clipped.
# 2. "Normalize the intrinsic reward by dividing it by a running estimate of the standard deviations of the intrinsic returns."
# 3. We initialize the normalization parameters by stepping a random agent in the environment for a small number of steps before beginning optimization.
# 4. Initialize two more networks: one target and one trained.
# 5. Atari settings: sticky actions, extrinsic reward clipping, grayscale, episode clipping, no termination on loss of life
from jax import config
from core.imports import *
import core.helpers as helpers
import core.networks as networks
# jax.config.update("jax_enable_x64", True)


SAVE_DIR = "rnd_fast_slow" 

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_rnd_obs: jnp.ndarray
    next_obs: jnp.ndarray
    rnd_target_feats: jnp.ndarray
    info: dict

def make_train(config):
    k_rho = config.get("RND_FEATURES", 128)
    
    # Episodic / Continuing / Absorbing
    is_episodic = config.get("EPISODIC", True)
    is_continuing = (not is_episodic)
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'

    def define_trace_logic(terminals, is_dummy, is_goal, was_goal):
        "Logic for the (intrinsic) GAE depends on whether we have episodic, continuing, or absorbing terminal state"
        if is_episodic: # standard, cut on terminal (also cut dummy transition's trace), and never absorb
            cut_trace = terminals | is_dummy
            absorb_mask = jnp.zeros_like(terminals, dtype=jnp.bool_)
        elif is_continuing: # never cut trace, never absorb.
            cut_trace = jnp.zeros_like(terminals, dtype=jnp.bool_)
            absorb_mask = jnp.zeros_like(terminals, dtype=jnp.bool_)
        elif is_absorbing:
            # Cut on dummy steps (S_T -> S_0) and normal deaths (S_{T-1} -> S_T)
            death = terminals & ~is_goal 
            cut_trace = death | is_dummy 
            # Goals are absorbing.
            absorb_mask = was_goal 
        continue_mask = jnp.logical_not(cut_trace) # 1.0 if continuing, 0.0 if cut    
        return cut_trace, continue_mask, absorb_mask
    
    # Replay Buffer
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    # Env
    env = helpers.make_env(config)
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    
    if config.get('SCHEDULE_BETA', False):
        # goes up until peak and then linearly decays to 0.
        beta_sch = helpers.make_hold_decay_hold_schedule(total_updates = config['NUM_UPDATES'], max_beta=config['BONUS_SCALE']) 
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    # Metrics Function
    def _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale):
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            value_loss, vi_loss, loss_actor, entropy, rnd_mse = loss_info
            metric.update({
                "ppo_actor_loss": loss_actor.mean(),
                "extrinsic_value_loss": value_loss.mean(),
                "vi_loss": vi_loss.mean(),
                "entropy": entropy.mean(),
                "rnd_mse": rnd_mse.mean(),
                "average_obs": jnp.mean(traj_batch.obs, axis=(0,1,2)),
                "bonus_mean": gaes[1].mean(),
                "bonus_std": gaes[1].std(),
                "bonus_max": gaes[1].max(),
                "lambda_ret_mean": targets[0].mean(),
                "lambda_ret_std": targets[0].std(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                "mean_rew": traj_batch.reward.mean(),
                "rho_scale": rho_scale,
                "num_goals": jnp.sum(traj_batch.info.get('is_goal', jnp.zeros_like(traj_batch.done))),
                "vi_pred": traj_batch.i_value.mean(),
                "vi_pred_scaled": traj_batch.i_value.mean() * rho_scale,
                "v_e_pred": traj_batch.value.mean(),
            })
            return metric

    def train(rng):
        obs_rms = helpers.init_rms(shape=(1, 84, 84))
        
        # --- initialize intrinsic reward rho components (Random Nets) ---
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        normalize_rho_features = False
        rnd_net, rho_params = networks.initialize_rnd_network(
            rnd_rng, (1, 84, 84), normalize_rho_features, bias=False, k=k_rho 
        )
        _, target_params = networks.initialize_rnd_network(
            rnd_rng, (1, 84, 84), normalize_rho_features, bias=False, k=k_rho 
        )
        
        # --- initialize PPO network ---
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=3)
        # "target params" will actually denote the slow moving params.
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rho_params, rho_params
        )
        
        # --- initialize the running obs statistics ---
        obsv, env_state = env.reset()
        
        # Initialize pure RMS state
        initial_obs_rms = helpers.init_rms(shape=(1, 84, 84)) 
        initial_obs_rms = helpers.update_rms(initial_obs_rms, 
            obsv[:, 3, :, :].reshape(-1, 1, 84, 84)
        )
        # initialize intrinsic return tracking
        ret_shape = (config["NUM_ENVS"],)
        irets = jnp.zeros(ret_shape)
        iret_rms = helpers.init_rms(shape = ret_shape)

        # --- Warm Up Running Observation Statistics ---
        WARMUP_STEPS = config.get("RND_WARMUP_STEPS", 50) * config["NUM_STEPS"]

        def _warmup_step(warmup_carry, unused):
            env_state, last_obs, obs_rms, rng = warmup_carry
            rng, _rng = jax.random.split(rng)
            action = jax.random.randint(_rng, shape=(last_obs.shape[0],), minval=0, maxval=n_actions)
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(env_state, action)
            # 3. Update RMS Stats
            obs_rms = helpers.update_rms(obs_rms, 
                obsv[:, 3, :, :].reshape(-1, 1, 84, 84)
            )
            return (env_state, obsv, obs_rms, rng), None

        # Execute Warmup Scan
        warmup_carry = (env_state, obsv, initial_obs_rms, rng)
        (env_state, obsv, obs_rms, rng), _ = jax.lax.scan(
            _warmup_step, warmup_carry, None, length=WARMUP_STEPS
        )

        def _update_step(runner_state, unused):
            obs_rms = runner_state['obs_rms']
            train_state = runner_state["train_state"]
            rnd_state = runner_state["rnd_state"]
            env_state = runner_state["env_state"]
            last_obs = runner_state["last_obs"]
            obs_rms = runner_state['obs_rms']
            iret_rms = runner_state['iret_rms']
            irets = runner_state['irets']
            rng = runner_state["rng"]
            idx = runner_state["idx"]

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack the carried features
                train_state, env_state, last_obs, obs_rms, iret_rms, irets, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # pi, v_ext, v_int
                b, ve, vi = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)
                next_ve, next_vi = network.apply(train_state.params, obsv, method=network.value)

                # Compute Intrinsic Reward
                latest_frame = obsv[:, 3, :, :].reshape(-1, 1, 84, 84)
                obs_rms = helpers.update_rms(obs_rms, latest_frame)
                next_rnd_obs = helpers.normalize_obs(obs_rms, latest_frame) # normalizes and clip
                # has shape (64, 1, 84, 84)
                # Forward passes
                rnd_target_feats = rnd_net.apply(target_params, next_rnd_obs)
                rho_feats = rnd_net.apply(rnd_state.params, next_rnd_obs)
                rho_feats_slow = rnd_net.apply(rnd_state.target_params, next_rnd_obs)

                # Calculate MSE for both
                mse_fast = 0.5 * ((rnd_target_feats - rho_feats)**2).sum(-1)
                mse_slow = 0.5 * ((rnd_target_feats - rho_feats_slow)**2).sum(-1)
                # Ri is the learning progress (clamped at 0 to avoid negative rewards)
                ri_raw = jnp.maximum(0.0, mse_slow - mse_fast)
                # normalization
                irets = ri_raw + config['GAMMA_i'] * irets
                iret_rms = helpers.update_rms(iret_rms, irets)

                ri = ri_raw / jnp.sqrt(iret_rms["var"])
                
                # Store transition
                transition = Transition(
                    done, action, ve, next_ve, vi, next_vi, reward, ri, log_prob, last_obs, next_rnd_obs, obsv, rnd_target_feats, info
                )

                runner_state = (train_state, env_state, obsv, obs_rms, iret_rms, irets, rng)
                return runner_state, transition
            # end env_step

            env_step_state = (train_state, env_state, last_obs, obs_rms, iret_rms, irets, rng)
            
            updated_env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )
            (train_state, env_state, last_obs, obs_rms, iret_rms, irets, rng) = updated_env_step_state
            
            # --- 1.a. Done State Handling Post-Processing ---
            terminals = traj_batch.done
            
            is_dummy = traj_batch.info.get("is_dummy", jnp.zeros_like(terminals))
            is_goal = traj_batch.info.get("is_goal", jnp.zeros_like(terminals))
            was_goal = traj_batch.info.get("was_goal", jnp.zeros_like(terminals))
            cut_trace, continue_mask, absorb_mask = define_trace_logic(terminals, is_dummy, is_goal, was_goal)
            
            # --- GAE ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, 
                config["GAMMA"], config["GAE_LAMBDA"], 
                cut_trace, absorb_mask, 
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes

            # --- 6. INTRINSIC vs. EXTRINSIC SCALING ---
            rho_scale = beta_sch(idx) # triangle schedule
            advantages = gae_e + (rho_scale * gae_i)

            # 7. UPDATE NETWORK
            def _update_epoch(update_state, unused):
                # Gradient Step Function (_update_minbatch)
                def _update_minbatch(train_states, batch_info):
                    train_state, rnd_state, mask_rng = train_states
                    # --- UPDATE PPO ---
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers.ppo_loss_two_vals, has_aux=True)
                    (total_loss, aux_losses), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    # --- UPDATE RND ---
                    def rnd_loss_fn(rnd_params):
                        pred = rnd_net.apply(rnd_params, traj_batch.next_rnd_obs)
                        diff = pred - traj_batch.rnd_target_feats
                        losses = jnp.mean(jnp.square(diff), axis=-1)
                        loss = (losses * mask).sum() / (mask.sum() + 1e-8)
                        return loss
                    
                    rnd_grad_fn = jax.value_and_grad(rnd_loss_fn)
                    mask = jax.random.bernoulli(mask_rng, p=config.get('RND_TRAIN_FRAC',0.25), shape=(traj_batch.obs.shape[0],))
                    rnd_loss, rnd_grads = rnd_grad_fn(rnd_state.params)
                    rnd_state = rnd_state.apply_gradients(grads=rnd_grads)
                    rng, mask_rng = jax.random.split(mask_rng)
                    losses = (*aux_losses, rnd_loss)
                    return (train_state, rnd_state, rng), losses
                    # end _update_minibatch
                
                # Shuffle Minibatches
                train_state, rnd_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                rng, mask_rng = jax.random.split(rng)
                train_states = train_state, rnd_state, mask_rng
                # Apply Gradient Steps
                train_states, total_loss = jax.lax.scan(_update_minbatch, train_states, minibatches)
                train_state, rnd_state, _ = train_states
                return (train_state, rnd_state, traj_batch, advantages, targets, rng), total_loss
            # end _update_epoch

            initial_update_state = (train_state, rnd_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, rnd_state, _, _, _, rng = update_state
            # Update the slow params:
            tau = config.get('RND_SLOW_TAU', 0.05)
            rnd_slow_params = jax.tree.map(
                lambda fast, slow: tau * fast + (1.0 - tau) * slow, 
                rnd_state.params, 
                rnd_state.target_params
            )
            rnd_state = rnd_state.replace(target_params=rnd_slow_params)

            # --------- Metrics ---------
            metric = _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale)

            runner_state = {
                "train_state": train_state,
                "env_state": env_state,
                "last_obs": last_obs,
                "rng": rng,
                "rnd_state": rnd_state,
                "obs_rms": obs_rms,
                "iret_rms": iret_rms,
                "irets": irets,
                "idx": idx + 1,
            }
            return runner_state, metric
        # end _update_step

        rng, _rng = jax.random.split(rng)

        initial_runner_state = {
            "train_state": train_state,
            "env_state": env_state,
            "last_obs": obsv,
            "rng": _rng,
            "rnd_state": rnd_state,
            "obs_rms": obs_rms,
            "iret_rms": iret_rms,
            "irets": irets,
            "idx": 1,
        }

        runner_state, metrics = jax.lax.scan(_update_step, initial_runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
