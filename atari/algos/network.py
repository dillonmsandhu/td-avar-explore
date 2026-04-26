from jax import config

from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.helpers import Transition
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "network" 

def make_train(config):

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
        next_obs: jnp.ndarray
        rho_feats: jnp.ndarray       # Exploration/Intrinsic features
        next_rho_feats: jnp.ndarray
        info: dict

    k_rho = config.get("RND_FEATURES", 128)
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
    
    # num times to loop
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
            i_value_loss, value_loss, loss_actor, entropy = loss_info
            metric.update({
                "ppo_actor_loss": loss_actor.mean(),
                "extrinsic_value_loss": value_loss.mean(),
                "intrinsic_value_loss": i_value_loss.mean(),
                "entropy": entropy.mean(),
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
        initial_sigma_state = {"S": jnp.eye(k_rho, dtype=jnp.float64)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        # Normalized keeps rho between 0 and 1, bias ensures sigma keeps track of total count.
        rho_net, rho_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["NORMALIZE_RHO_FEATURES"], bias=config['BIAS'], k=k_rho 
        )
        def get_rho_feats(obs):
            return rho_net.apply(rho_params, obs)

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=3)
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
                b, value, i_val = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)
                next_val, next_i_val = network.apply(train_state.params, obsv, method=network.value)

                next_rho_feat = get_rho_feats(obsv)
                
                dummy = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, i_val, next_i_val, reward, dummy, log_prob, 
                    last_obs, obsv, last_rho_feat,next_rho_feat, info
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
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            
            # --- 6. INTRINSIC vs. EXTRINSIC SCALING ---
            rho_scale = beta_sch(idx) # triangle schedule
            advantages = gae_e + (rho_scale * gae_i)

            # 7. UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn_intrinsic_v, has_aux=True)
                    
                    (total_loss, aux_losses), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    return train_state, aux_losses 

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                # loss_info here becomes the batched aux_losses
                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), loss_info
                # end update_epcoh
            
            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
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
