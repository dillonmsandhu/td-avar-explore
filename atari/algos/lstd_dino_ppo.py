# PPO using LSTD on DiNO Features. 
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.buffer import FeatureTraceBufferManagerE, LSTDBufferStateE
from core.lstd import solve_lstd_lambda_from_buffer_extrinsic
import jax.image
from transformers import FlaxDinov2Model

# jax.config.update("jax_enable_x64", True)
# DINO_PATH = "/usr/xtmp/ds541/hf_models/dino_v2_flax"
DINO_PATH = "/usr/xtmp/ds541/hf_models/dino_v2_flax_reg"
SAVE_DIR = "lstd_dino_ppo" 

class TransitionE(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    # --- NEW FIELDS ---
    phi: jnp.ndarray            # LSTD features
    next_phi: jnp.ndarray 

def make_train(config):
    dino_model = FlaxDinov2Model.from_pretrained(DINO_PATH)
    k_rho = config.get("RND_FEATURES", 128)
    dino_out_dim = 384 * 2 # 2 frames * 384 dims each
    k_lstd = 128 # bias

    def define_trace_logic(terminals, is_dummy):
        # Purely episodic
        cut_trace = terminals | is_dummy
        absorb_mask = jnp.zeros_like(terminals, dtype=jnp.bool_)
        continue_mask = jnp.logical_not(cut_trace) # 1.0 if continuing, 0.0 if cut    
        return cut_trace, continue_mask, absorb_mask
    
    # Replay Buffer
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    config['CHUNK_SIZE'] =  100_000 + batch_size # chunking for LSTD solver
    buffer_manager = FeatureTraceBufferManagerE(config, k_lstd, k_rho, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE']) # stateless buffer manager.
    config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    config['PADDED_CAPACITY'] = buffer_manager.padded_capacity
    
    # Env
    env = helpers.make_env(config)
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n

    # Metrics Function
    def _compile_metrics(traj_batch, loss_info, gaes, targets):
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            loss_actor, entropy = loss_info
            metric.update({
                "ppo_actor_loss": loss_actor.mean(),
                "entropy": entropy.mean(),
                "feat_norm": jnp.linalg.norm(traj_batch.next_phi, axis=-1).mean(),
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
        
        lstd_params = dino_model.params
        rng, proj_rng = jax.random.split(rng)
        projection_matrix = jax.random.normal(proj_rng, (dino_out_dim, k_lstd-1)) / jnp.sqrt(k_lstd)
        
        initial_lstd_state = {"w": jnp.zeros(k_lstd), }
        initial_buffer_state = buffer_manager.init_state()

        def get_lstd_feats(obs):
            # 1. Extract 2nd and 4th frames -> (B, 2, 84, 84)
            # FIX: Use a standard Python list to avoid TracerArrayConversionError
            x = obs[:, [1, 3], :, :]
            B, T, H, W = x.shape
            
            # 2. Prep for DINO (Fold time, normalize, repeat to RGB, resize)
            x = x.reshape(B * T, 1, H, W).astype(jnp.float32) / 255.0
            x = jnp.repeat(x, 3, axis=1)
            x = jax.image.resize(x, shape=(B * T, 3, 224, 224), method='bilinear')
            
            mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32).reshape(1, 3, 1, 1)
            std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32).reshape(1, 3, 1, 1)
            x = (x - mean) / std
            
            # 3. Get DINO features
            outputs = dino_model(pixel_values=x, params=lstd_params)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            
            # 4. Concatenate the 2 frames -> (B, 768)
            concat_feats = cls_tokens.reshape(B, T * 384)
            
            # 5. FAST RANDOM PROJECTION -> (B, projected_dim)
            projected_feats = concat_feats @ projection_matrix
            
            # 6. ADD BIAS -> Concatenate a 1.0 to the end of each vector -> (B, projected_dim + 1)
            bias = jnp.ones((B, 1), dtype=cls_tokens.dtype)
            projected_feats = jnp.concatenate([projected_feats, bias], axis=-1)
            
            return projected_feats

        # network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=1) # Actor net only.
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=1, cnn_torso=config.get('CNN_TORSO', 'CNN'))

        train_state = networks.basic_flax_train_state(
            config, network, network_params
        )
        
        obsv, env_state = env.reset()
        initial_phi = get_lstd_feats(obsv)

        def _update_step(runner_state, unused):

            train_state = runner_state["train_state"]
            lstd_state = runner_state["lstd_state"]
            buffer_state = runner_state["buffer_state"] 
            env_state = runner_state["env_state"]
            last_obs = runner_state["last_obs"]
            last_phi = runner_state["last_phi"]
            rng = runner_state["rng"]
            idx = runner_state["idx"]

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack the carried features
                train_state, env_state, last_obs, last_phi, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                b = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)

                # --- IN-LOOP FEATURE EXTRACTION ---
                next_phi = get_lstd_feats(obsv)
                
                dummy = jnp.zeros_like(reward) # Placeholder for is_dummy logic if needed

                transition = TransitionE(
                    done, action, dummy, dummy, reward, log_prob, 
                    last_obs, info, phi=last_phi, next_phi=next_phi, 
                )

                # Pass the 'next' features forward as the 'last' features for the next step
                runner_state = (train_state, env_state, obsv, next_phi, rng)
                return runner_state, transition
            # end env_step
            env_step_state = (
                train_state, env_state, last_obs, runner_state["last_phi"], rng
            )
            
            (_, env_state, last_obs, last_phi,rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # Process batch
            # --- 1.a. Done State Handling Post-Processing ---
            terminals = traj_batch.done
            
            is_dummy = traj_batch.info.get("is_dummy", jnp.zeros_like(terminals))
            cut_trace, continue_mask, _ = define_trace_logic(terminals, is_dummy,)
            
            # --- 2. Compute Trace and Add to Buffer ---
            traces = helpers.calculate_traces(traj_batch.phi, cut_trace, config["GAMMA"], config["LSTD_LAMBDA_i"])
            buffer_batch = LSTDBufferStateE(
                traces=traces, 
                features=traj_batch.phi, 
                next_features=traj_batch.next_phi, 
                reward=traj_batch.reward,
                continue_masks=continue_mask, 
                size=jnp.array(batch_size)
            )
            buffer_state = buffer_manager.update_buffer(buffer_state, buffer_batch)
            
            # --- 3. SOLVE LSTD ON BUFFER ---
            lstd_state = solve_lstd_lambda_from_buffer_extrinsic(buffer_state, config)

            # --- 4. EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, prb_rng)
            
            # --- LSTD PREDICTIONS ---
            v = traj_batch.phi @ lstd_state["w"] 
            next_v = traj_batch.next_phi @ lstd_state["w"] 
            V_max_raw = 1.0 / (1.0 - config['GAMMA'])
            v, next_v = jax.tree_util.tree_map(
                lambda x: jnp.clip(x, -V_max_raw, V_max_raw),
                (v, next_v)
            )
            
            # --- traj_batch update for GAE ---
            traj_batch = traj_batch._replace(value=v, next_value=next_v)

            advantages, extrinsic_target = helpers.calculate_gaeE(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"], 
            )

            # 7. UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn_actor, has_aux=True)
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

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = _compile_metrics(traj_batch, loss_info, advantages, extrinsic_target)

            runner_state = {
                "train_state": train_state,
                "env_state": env_state,
                "last_obs": last_obs,
                "last_phi": last_phi,            
                "rng": rng,
                "lstd_state": lstd_state,
                "buffer_state": buffer_state,
                "idx": idx + 1,
            }
            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        initial_runner_state = {
            "train_state": train_state,
            "env_state": env_state,
            "last_obs": obsv,
            "rng": _rng,
            "lstd_state": initial_lstd_state,
            "buffer_state": initial_buffer_state,
            "idx": 1,
            "last_phi": initial_phi,            
        }

        runner_state, metrics = jax.lax.scan(_update_step, initial_runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
