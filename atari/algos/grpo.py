from jax import config

from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.helpers import Transition
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "grpo" 

def make_train(config):
    k_rho = config.get("RND_FEATURES", 128)
    k_lstd = k_rho
    # Episodic / Continuing / Absorbing
    is_episodic = config.get("EPISODIC", True)
    is_continuing = (not is_episodic)
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'

    def define_trace_logic(terminals, is_dummy, is_goal, was_goal):
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
            metric.update({
                "ppo_loss": loss_info[0],
                "rnd_loss": loss_info[1],
                "feat_norm": jnp.linalg.norm(traj_batch.next_rho_feats, axis=-1).mean(),
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
                "v_e_pred": traj_batch.value.mean(),
            })
            return metric

    def train(rng):
        initial_sigma_state = {"S": jnp.eye(k_rho, dtype=jnp.float64)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        # Normalized keeps rho between 0 and 1, bias ensures sigma keeps track of total count.
        rho_net, rho_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["NORMALIZE_RHO_FEATURES"], bias=config['BIAS'], k=k_rho 
        )
        
        def get_rho_feats(obs):
            return rho_net.apply(rho_params, obs)
        
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rho_net, network_params, rho_params
        )
        
        obsv, env_state = env.reset()
        initial_rho_feat = get_rho_feats(obsv)

        def _update_step(runner_state, unused):

            train_state = runner_state["train_state"]
            sigma_state = runner_state["sigma_state"]
            rnd_state = runner_state["rnd_state"]
            env_state = runner_state["env_state"]
            last_obs = runner_state["last_obs"]
            last_rho_feat = runner_state["last_rho_feat"]
            rng = runner_state["rng"]
            idx = runner_state["idx"]

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack the carried features
                train_state, env_state, last_obs, last_rho_feat, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                b, value = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)
                next_val = network.apply(train_state.params, obsv, method=network.value)

                next_rho_feat = get_rho_feats(obsv)

                dummy = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, dummy, dummy, reward, dummy, log_prob, 
                    last_obs, obsv, info, phi=jnp.zeros_like(last_rho_feat), next_phi=jnp.zeros_like(last_rho_feat), 
                    rho_feats=last_rho_feat, next_rho_feats=next_rho_feat
                )

                # Pass the 'next' features forward as the 'last' features for the next step
                runner_state = (train_state, env_state, obsv, next_rho_feat, rng)
                return runner_state, transition
            # end env_step
            env_step_state = (
                train_state, env_state, last_obs, runner_state["last_rho_feat"], rng
            )
            
            (_, env_state, last_obs, last_rho_feat, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # Process batch
            # --- 0. GLOBAL COVARIANCE UPDATE (Pure Accumulation) ---
            sigma_state = helpers.update_cov(sigma_state, traj_batch.rho_feats)            
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_rho))
            
            # --- 1.a. Done State Handling Post-Processing ---
            terminals = traj_batch.done
            
            is_dummy = traj_batch.info.get("is_dummy", jnp.zeros_like(terminals))
            is_goal = traj_batch.info.get("is_goal", jnp.zeros_like(terminals))
            was_goal = traj_batch.info.get("was_goal", jnp.zeros_like(terminals))
            cut_trace, continue_mask, absorb_mask = define_trace_logic(terminals, is_dummy, is_goal, was_goal)

            # --- 5. COMPUTE TARGETS ---
            rho_feats_final = jnp.where(absorb_mask[..., None], traj_batch.rho_feats, traj_batch.next_rho_feats)
            rho = helpers.get_scale_free_bonus(Sigma_inv, rho_feats_final)
    
            # --- Final traj_batch update for GAE ---
            traj_batch = traj_batch._replace(intrinsic_reward=rho)

            gaes, targets = helpers.calculate_gae(
                traj_batch, 
                config["GAMMA"], config["GAE_LAMBDA"], 
                cut_trace, absorb_mask, 
                γi=config["GAMMA_i"], λi=1.0 # Retrace style
            )
            extrinsic_adv, raw_intrinsic_returns = gaes
            extrinsic_target = targets[0]
            
            # --- THE TIME-DEPENDENT BASELINE ---
            # Compute the mean and std across the NUM_ENVS axis (axis=1).
            # This creates a unique baseline for by timestep.
            mean_int_ret_t = jnp.mean(raw_intrinsic_returns, axis=1, keepdims=True)
            # std_int_ret_t = jnp.std(raw_intrinsic_returns, axis=1, keepdims=True)
            # baselined_int_adv = (raw_intrinsic_returns - mean_int_ret_t) / (std_int_ret_t + 1e-8)
            
            # Normalize conditionally based on the timestep index
            baselined_int_adv = (raw_intrinsic_returns - mean_int_ret_t)
            
            # --- 6. INTRINSIC vs. EXTRINSIC SCALING ---
            rho_scale = beta_sch(idx)
            advantages = extrinsic_adv + (rho_scale * baselined_int_adv)

            # 7. UPDATE NETWORK
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
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale)

            runner_state = {
                "train_state": train_state,
                "env_state": env_state,
                "last_obs": last_obs,
                "last_rho_feat": last_rho_feat,  
                "rng": rng,
                "rnd_state": rnd_state,
                "sigma_state": sigma_state,
                "idx": idx + 1,
            }
            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        initial_runner_state = {
            "train_state": train_state,
            "env_state": env_state,
            "last_obs": obsv,
            "rng": _rng,
            "rnd_state": rnd_state,
            "sigma_state": initial_sigma_state,
            "idx": 1,
            "last_rho_feat": initial_rho_feat,  
        }

        runner_state, metrics = jax.lax.scan(_update_step, initial_runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
