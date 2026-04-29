# PPO
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.buffer import FeatureTraceBufferManagerE, LSTDBufferStateE
from core.lstd import solve_lstd_lambda_from_buffer_extrinsic

SAVE_DIR = "ppo" 

class TransitionPPO(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict

def make_train(config):
    
    # Batching
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    # Env
    env = helpers.make_env(config)
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n

    # Metrics Function
    def _compile_metrics(traj_batch, loss_info, gaes, targets):
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            loss_value, loss_actor, entropy = loss_info
            metric.update({
                "value_loss": loss_value.mean(),
                "ppo_actor_loss": loss_actor.mean(),
                "entropy": entropy.mean(),
                "adv_mean": gaes.mean(),
                "adv_std": gaes.std(),
                "lambda_ret_mean": targets.mean(),
                "lambda_ret_std": targets.std(),
                "mean_rew": traj_batch.reward.mean(),
                "num_goals": jnp.sum(traj_batch.info.get('is_goal', jnp.zeros_like(traj_batch.done))),
                "v_pred": traj_batch.value.mean(),
                "v_std": traj_batch.value.std(),
            })
            return metric

    def train(rng):
        
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=2, cnn_torso='CNN') # Actor net only.

        train_state = networks.basic_flax_train_state(
            config, network, network_params
        )
        
        obsv, env_state = env.reset()

        def _update_step(runner_state, unused):

            train_state = runner_state["train_state"]
            env_state = runner_state["env_state"]
            last_obs = runner_state["last_obs"]
            rng = runner_state["rng"]
            idx = runner_state["idx"]

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack the carried features
                train_state, env_state, last_obs, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                b, val = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)

                # --- IN-LOOP FEATURE EXTRACTION ---
                _, next_val = network.apply(train_state.params, obsv)
                
                dummy = jnp.zeros_like(reward) # Placeholder for is_dummy logic if needed

                transition = TransitionPPO(
                    done, action, val, next_val, reward, log_prob, last_obs, info
                )

                # Pass the 'next' features forward as the 'last' features for the next step
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            # end env_step
            env_step_state = (
                train_state, env_state, last_obs, rng
            )
            
            (_, env_state, last_obs,rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            advantages, target = helpers.calculate_gaeE(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
            )

            # 7. UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, aux_losses), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, aux_losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            initial_update_state = (train_state, traj_batch, advantages, target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = _compile_metrics(traj_batch, loss_info, advantages, target)

            runner_state = {
                "train_state": train_state,
                "env_state": env_state,
                "last_obs": last_obs,            
                "rng": rng,
                "idx": idx + 1,
            }
            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        initial_runner_state = {
            "train_state": train_state,
            "env_state": env_state,
            "last_obs": obsv,
            "rng": _rng,
            "idx": 1,
        }

        runner_state, metrics = jax.lax.scan(_update_step, initial_runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

