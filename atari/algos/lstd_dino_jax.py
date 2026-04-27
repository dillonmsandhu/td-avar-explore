from jax import config

from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.buffer import FeatureTraceBufferManager, LSTDBufferState
from core.lstd import solve_lstd_lambda_from_buffer
from core.helpers import Transition
import jax.image
from transformers import FlaxDinov2Model

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "cov_lstd_dino" 

def make_train(config):
    config.setdefault("DINO_MODEL", "dinov2_vits14")
    config.setdefault("DINO_DTYPE", "fp16")
    config.setdefault("DINO_BATCH_SIZE", 4096)
    config.setdefault("DINO_FRAME_STRATEGY", "last")

    k_lstd = config.get('LSTD_FEATURES', 128)
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
    
    # Replay Buffer
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    config['CHUNK_SIZE'] =  100_000 + batch_size # chunking for LSTD solver
    buffer_manager = FeatureTraceBufferManager(config, k_lstd, k_rho, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE']) # stateless buffer manager.
    config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    config['PADDED_CAPACITY'] = buffer_manager.padded_capacity
    
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
            value_loss, loss_actor, entropy = loss_info
            metric.update({
                "ppo_actor_loss": loss_actor.mean(),
                "extrinsic_value_loss": value_loss.mean(),
                "entropy": entropy.mean(),
                "feat_norm": jnp.linalg.norm(traj_batch.next_phi, axis=-1).mean(),
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
        dino_model = FlaxDinov2Model.from_pretrained("facebook/dinov2-vits14")
        lstd_params = dino_model.params
        rng, proj_rng = jax.random.split(rng)
        dino_out_dim = 384 * 2 # 2 frames * 384 dims each
        projection_matrix = jax.random.normal(proj_rng, (dino_out_dim, k_lstd)) / jnp.sqrt(k_lstd)
        
        initial_lstd_state = {"w": jnp.zeros(k_lstd), }
        initial_buffer_state = buffer_manager.init_state()
        initial_sigma_state = {"S": jnp.eye(k_rho, dtype=jnp.float64)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        # Normalized keeps rho between 0 and 1, bias ensures sigma keeps track of total count.
        rho_net, rho_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["NORMALIZE_RHO_FEATURES"], bias=config['BIAS'], k=k_rho 
        )
        # 
        lstd_net, lstd_params = networks.initialize_lstd_network( # Or a different architecture
            rnd_rng, obs_shape, config["NORMALIZE_LSTD_FEATURES"], bias=True, k=k_lstd
        ) # will be the same params if the same network

        def get_rho_feats(obs):
            return rho_net.apply(rho_params, obs)

        def get_lstd_feats(obs):
                # 1. Extract 2nd and 4th frames -> (B, 2, 84, 84)
                x = obs[:, jnp.array([1, 3]), :, :]
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
                
                # 5. FAST RANDOM PROJECTION -> (B, 128)
                projected_feats = concat_feats @ projection_matrix
                
                return projected_feats

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rho_net, network_params, rho_params
        )
        
        obsv, env_state = env.reset()
        initial_phi = get_lstd_feats(obsv)
        initial_rho_feat = get_rho_feats(obsv)

        def _update_step(runner_state, unused):

            train_state = runner_state["train_state"]
            lstd_state = runner_state["lstd_state"]
            sigma_state = runner_state["sigma_state"]
            buffer_state = runner_state["buffer_state"] 
            rnd_state = runner_state["rnd_state"]
            env_state = runner_state["env_state"]
            last_obs = runner_state["last_obs"]
            last_phi = runner_state["last_phi"]
            last_rho_feat = runner_state["last_rho_feat"]
            rng = runner_state["rng"]
            idx = runner_state["idx"]

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack the carried features
                train_state, env_state, last_obs, last_phi, last_rho_feat, rng = env_scan_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                b, value = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)
                next_val = network.apply(train_state.params, obsv, method=network.value)

                # --- NEW: IN-LOOP FEATURE EXTRACTION ---
                next_phi = get_lstd_feats(obsv)
                next_rho_feat = get_rho_feats(obsv)

                dummy = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, dummy, dummy, reward, dummy, log_prob, 
                    last_obs, obsv, info, phi=last_phi, next_phi=next_phi, 
                    rho_feats=last_rho_feat, next_rho_feats=next_rho_feat
                )

                # Pass the 'next' features forward as the 'last' features for the next step
                runner_state = (train_state, env_state, obsv, next_phi, next_rho_feat, rng)
                return runner_state, transition
            # end env_step
            env_step_state = (
                train_state, env_state, last_obs, 
                runner_state["last_phi"], runner_state["last_rho_feat"], rng
            )
            
            (_, env_state, last_obs, last_phi, last_rho_feat,rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

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
            
            # --- 2. Compute Trace and Add to Buffer ---
            traces = helpers.calculate_traces(traj_batch.phi, cut_trace, config["GAMMA_i"], config["LSTD_LAMBDA_i"])
            buffer_batch = LSTDBufferState(
                traces=traces, 
                features=traj_batch.phi, 
                next_features=traj_batch.next_phi, 
                rho_features=traj_batch.rho_feats,
                next_rho_features=traj_batch.next_rho_feats,
                continue_masks=continue_mask, 
                absorb_masks=absorb_mask, 
                size=jnp.array(batch_size)
            )
            buffer_state = buffer_manager.update_buffer(buffer_state, buffer_batch)
            
            # --- 3. SOLVE LSTD ON BUFFER ---
            lstd_state = solve_lstd_lambda_from_buffer(buffer_state, Sigma_inv, config)

            # --- 4. EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, prb_rng)
            
            # --- 5. COMPUTE TARGETS ---
            rho_feats_final = jnp.where(absorb_mask[..., None], traj_batch.rho_feats, traj_batch.next_rho_feats)
            rho = helpers.get_scale_free_bonus(Sigma_inv, rho_feats_final)
            
            # --- LSTD PREDICTIONS ---
            v_i = traj_batch.phi @ lstd_state["w"] 
            next_v_i = traj_batch.next_phi @ lstd_state["w"] 
            
            # --- Clip ---
            V_max_raw = 1.0 / (1.0 - config['GAMMA_i'])
            v_i, next_v_i = jax.tree_util.tree_map(lambda x: jnp.clip(x, 0, V_max_raw), (v_i, next_v_i))
            
            # --- Final traj_batch update for GAE ---
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho, next_i_val=next_v_i)

            gaes, targets = helpers.calculate_gae(
                traj_batch, 
                config["GAMMA"], config["GAE_LAMBDA"], 
                cut_trace, absorb_mask, 
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            # center just the intrinsic advantage. i.e. treat the whole thing as a baseline and give it mean 0.
            gae_i = jnp.where(config.get('GLOBAL_ADVANTAGE_CENTERING', False),
                             gae_i - gae_i.mean(),
                             gae_i)
            # --- 6. INTRINSIC vs. EXTRINSIC SCALING ---
            rho_scale = beta_sch(idx) # triangle schedule
            advantages = gae_e + (rho_scale * gae_i)
            extrinsic_target = targets[0]

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

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale)

            runner_state = {
                "train_state": train_state,
                "env_state": env_state,
                "last_obs": last_obs,
                "last_phi": last_phi,            
                "last_rho_feat": last_rho_feat,  
                "rng": rng,
                "lstd_state": lstd_state,
                "rnd_state": rnd_state,
                "sigma_state": sigma_state,
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
            "rnd_state": rnd_state,
            "sigma_state": initial_sigma_state,
            "buffer_state": initial_buffer_state,
            "idx": 1,
            "last_phi": initial_phi,            
            "last_rho_feat": initial_rho_feat,  
        }

        runner_state, metrics = jax.lax.scan(_update_step, initial_runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
