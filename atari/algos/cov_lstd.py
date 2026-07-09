from jax import config

from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.buffer import FeatureTraceBufferManager, LSTDBufferState
from core.lstd import solve_lstd_lambda_from_buffer
from core.helpers import Transition
from core.dino_features import get_dino_features_on_atari_obs_grid
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "cov_lstd" 

def make_train(config):
    k_rho = config.get("RND_FEATURES", 512)
    normalize_rho_obs = config.get("NORMALIZE_RHO_OBS", True)
    normalize_rho = config.get("NORMALIZE_RHO", True)
    normalize_lstd_obs = config.get("NORMALIZE_LSTD_OBS", True)

    config['COV_LEAK'] = config.get('COV_LEAK', 1 - 1e-5)
    
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
    # buffer_manager = FeatureTraceBufferManager(config, k_lstd, k_rho, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE']) # stateless buffer manager.
    # config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    # config['PADDED_CAPACITY'] = buffer_manager.padded_capacity
    
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
    def _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale, ret_std, lstd_state, sigma_state):
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            w_norm = lstd_state["w_norm"]
            A_trace = lstd_state["A_trace"]
            value_loss, loss_actor, entropy = loss_info
            metric.update({
                "ppo_actor_loss": loss_actor.mean(),
                "extrinsic_value_loss": value_loss.mean(),
                "entropy": entropy.mean(),
                "feat_norm": jnp.linalg.norm(traj_batch.next_phi, axis=-1).mean(),
                "feat_var": jnp.var(traj_batch.phi, axis=0).mean(),
                "rho_feat_var": jnp.var(traj_batch.rho_feats, axis=0).mean(),
                "average_obs": jnp.mean(traj_batch.obs, axis=(0,1,2)),
                "median_obs": jnp.median(traj_batch.obs, axis=(0,1,2)),
                "obs": traj_batch.obs[0,0,0,:,:],
                "bonus_mean": gaes[1].mean(),
                "bonus_std": gaes[1].std(),
                "bonus_max": gaes[1].max(),
                "lambda_ret_mean": targets[0].mean(),
                "lambda_ret_std": targets[0].std(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                "mean_rew": traj_batch.reward.mean(),
                "rho_scale": rho_scale,
                "ret_std": ret_std,
                "num_goals": jnp.sum(traj_batch.info.get('is_goal', jnp.zeros_like(traj_batch.done))),
                "vi_pred": traj_batch.i_value.mean(),
                "vi_pred_scaled": traj_batch.i_value.mean() * rho_scale / (ret_std + 1e-8),
                "v_e_pred": traj_batch.value.mean(),
                "val_loss_ratio": value_loss / (loss_actor + 1e-8),
            })
            metric.update({
                # 1. Vanishing Bonus Check
                "ri_mean": traj_batch.intrinsic_reward.max(),
                "ri_max": traj_batch.intrinsic_reward.max(),
                "ri_min": traj_batch.intrinsic_reward.min(),
                
                # 2. LSTD Explosion Check
                "vi_pred_max": traj_batch.i_value.max(),
                "vi_pred_min": traj_batch.i_value.min(),
                "lstd_w_norm": w_norm,
                "lstd_A_trace": A_trace,
                
                # 3. Advantage Domination Check
                "gae_i_mean": jnp.mean(jnp.abs(gaes[1])), # Mean absolute advantage
                "gae_e_mean": jnp.mean(jnp.abs(gaes[0])),
                "gae_scale_ratio": jnp.mean(jnp.abs(gaes[1])) / (jnp.mean(jnp.abs(gaes[0])) + 1e-8),
                "gae_ratio": jnp.mean( jnp.abs(gaes[1]) / (jnp.abs(gaes[0])+ 1e-8) ) ,
                "gae_intrinsic_frac": jnp.mean(jnp.abs(gaes[1]) / (jnp.abs(gaes[0]) + jnp.abs(gaes[1]) + 1e-8)),
                
                # intrinsic value accuracy:
                "i_target_mean": targets[1].mean(),
                "i_value_error": jnp.mean(jnp.square(targets[1] - traj_batch.i_value)),
                
                
                # 4. Feature Health
                "phi_max": traj_batch.phi.max(),
                "sigma_trace": jnp.trace(sigma_state["S"]), # Is the covariance exploding?
            })
            
            return metric

    def train(rng):
        obs_rms = helpers.init_rms(shape=(1, 84, 84))
        
        # --- initialize intrinsic reward rho components ---
        initial_sigma_state = {"S": jnp.eye(k_rho, dtype=jnp.float64)} # global accumulation
        rnd_rng, rng = jax.random.split(rng)
        # Normalized keeps rho between 0 and 1, bias ensures sigma keeps track of total count.
        # rho_net, rho_params = networks.initialize_rnd_network(
        #     rnd_rng, obs_shape, config["NORMALIZE_RHO_FEATURES"],
        #      bias=config['BIAS'], k=k_rho 
        # )

        # --- initialize intrinsic reward rho components (Random Nets) ---
        # Change: rho is only based on the final frame (of the next observation)
        rnd_rng, rng = jax.random.split(rng)
        rho_net = networks.RND_Target(k=k_rho)        
        dummy_obs = jnp.zeros((1, 1, 84, 84)) # Dummy input to trace shapes
        rho_params = rho_net.init(rnd_rng, dummy_obs)

        def get_rho_feats(obs):
            return rho_net.apply(rho_params, obs)
        
        # --- initialize LSTD components---
        if config.get('LSTD_DINO', False):
            get_lstd_feats = get_dino_features_on_atari_obs_grid
            k_lstd = 385

        else:
            k_lstd = config.get("LSTD_FEATURES", 512)
            lstd_net, lstd_params = networks.initialize_lstd_network( # Or a different architecture
                rnd_rng, obs_shape, config["NORMALIZE_LSTD_FEATURES"], bias=True, k=k_lstd, pool=config['POOL_LSTD_NET'], layer_norm = config['LSTD_LAYERNORM']
            ) # will be the same params if the same network
            def get_lstd_feats(obs):
                return lstd_net.apply(lstd_params, obs)
        
        buffer_manager = FeatureTraceBufferManager(config, k_lstd, k_rho, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE']) # stateless buffer manager.
        config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
        config['PADDED_CAPACITY'] = buffer_manager.padded_capacity
        initial_buffer_state = buffer_manager.init_state()
        initial_lstd_state = {"w": jnp.zeros(k_lstd), "w_norm": 0, "A_trace": 0}

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=2, shared_torso = True)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rho_net, network_params, rho_params
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
        iret_rms = helpers.init_rms(shape=())
        # iret_rms = helpers.init_rms(shape = ret_shape)

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
        # normalize initial observation
        obs_last_frame = obsv[:, 3, :, :].reshape(-1, 1, 84, 84)
        initial_phi =  get_lstd_feats(helpers.normalize_obs(obs_rms, obsv) if normalize_lstd_obs else obsv )
        initial_rho_feat = get_rho_feats(
            helpers.normalize_obs(obs_rms, obs_last_frame) if normalize_rho_obs else obs_last_frame
        )

        def _update_step(runner_state, unused):
            obs_rms = runner_state['obs_rms']
            iret_rms = runner_state['iret_rms']
            irets = runner_state['irets']
            train_state = runner_state["train_state"]
            lstd_state = runner_state["lstd_state"]
            sigma_state = runner_state["sigma_state"]
            buffer_state = runner_state["buffer_state"] 
            rnd_state = runner_state["rnd_state"]
            env_state = runner_state["env_state"]
            last_obs = runner_state["last_obs"]
            last_phi = runner_state["last_phi"]
            last_rho_feat = runner_state["last_rho_feat"]
            obs_rms = runner_state['obs_rms']
            rng = runner_state["rng"]
            idx = runner_state["idx"]

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                # Unpack the carried features
                train_state, env_state, last_obs, last_phi, last_rho_feat, obs_rms, rng = env_scan_state

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
                s_prime_last_frame = obsv[:, 3, :, :].reshape(-1, 1, 84, 84)
                obs_rms = helpers.update_rms(obs_rms, s_prime_last_frame.reshape(-1, 1, 84, 84)) # update with new obs

                # Change: LSTD uses all four stacked frames, rho uses just the final frame (matching RND's obs)
                next_obs_rho = helpers.normalize_obs(obs_rms, s_prime_last_frame) if normalize_rho_obs else s_prime_last_frame
                next_obs_lstd = helpers.normalize_obs(obs_rms, obsv) if normalize_lstd_obs else obsv
                next_phi = get_lstd_feats(next_obs_lstd)
                next_rho_feat = get_rho_feats(next_obs_rho)

                dummy = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, dummy, dummy, reward, dummy, log_prob, 
                    last_obs, obsv, info, phi=last_phi, next_phi=next_phi, 
                    rho_feats=last_rho_feat, next_rho_feats=next_rho_feat
                )

                # Pass the 'next' features forward as the 'last' features for the next step
                runner_state = (train_state, env_state, obsv, next_phi, next_rho_feat, obs_rms, rng)
                return runner_state, transition
            
            # end env_step
            env_carry = (
                train_state, env_state, last_obs, 
                runner_state["last_phi"], runner_state["last_rho_feat"], obs_rms, rng
            )
            
            env_carry, traj_batch = jax.lax.scan(
                _env_step, env_carry, None, config["NUM_STEPS"]
                )

            (_, env_state, last_obs, last_phi, last_rho_feat, obs_rms, rng) = env_carry

            # Process batch
            # --- 0. GLOBAL COVARIANCE UPDATE (Pure Accumulation) ---
            sigma_state = helpers.update_cov(sigma_state, 
                        traj_batch.rho_feats, 
                        leak = config['COV_LEAK'],
                        bonus_type = config['BONUS_TYPE']
            )            
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_rho))
            
            # --- 1.a. Done State Handling Post-Processing ---
            terminals = traj_batch.done
            
            is_dummy = traj_batch.info.get("is_dummy", jnp.zeros_like(terminals))
            is_goal = traj_batch.info.get("is_goal", jnp.zeros_like(terminals))
            was_goal = traj_batch.info.get("was_goal", jnp.zeros_like(terminals))
            cut_trace, continue_mask, absorb_mask = define_trace_logic(terminals, is_dummy, is_goal, was_goal)
            
            # --- 2. Compute Rho ---
            rho_feats_final = jnp.where(absorb_mask[..., None], traj_batch.rho_feats, traj_batch.next_rho_feats)
            rho = helpers.get_scale_free_bonus(Sigma_inv, rho_feats_final)

            # --- Standardize Rho --- 
            def compute_intrinsic_ret(current_irets, raw_rho, continue_mask, gamma_i):
                """
                Applies the standard RND forward exponential filter to the rewards.
                raw_rho expected shape: (NUM_STEPS, NUM_ENVS)
                """
                def _forward_step(carry, step_data):
                    r_t, cont_mask = step_data
                    # Multiply the running return (carry) by gamma and the mask
                    next_ret = r_t + (gamma_i * carry * cont_mask)
                    return next_ret, next_ret
                
                c_mask = continue_mask.squeeze(-1) if continue_mask.ndim == 3 else continue_mask
                scan_inputs = (raw_rho, c_mask)
                final_irets, per_timestep_irets = jax.lax.scan(_forward_step, current_irets, scan_inputs)
                return final_irets, per_timestep_irets
            
            irets, per_timestep_irets = compute_intrinsic_ret(irets, rho, continue_mask, config["GAMMA_i"])
            iret_rms = helpers.update_rms(iret_rms, per_timestep_irets.reshape(-1))
            # iret_rms = helpers.update_ema_rms(iret_rms, per_timestep_irets.reshape(-1))
            
            scaling_factor = jnp.sqrt(iret_rms["var"] + 1e-8) if normalize_rho else 1.0
            rho = rho / scaling_factor
            
            # --- 3. Compute Trace and Add to Buffer ---
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
            lstd_state = solve_lstd_lambda_from_buffer(buffer_state, Sigma_inv, config, scaling_factor = 1.0)

            # --- 4. EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, prb_rng)
            
            # --- 5. COMPUTE TARGETS ---
            
            # --- LSTD PREDICTIONS ---
            v_i = traj_batch.phi @ lstd_state["w"] / scaling_factor
            next_v_i = traj_batch.next_phi @ lstd_state["w"] / scaling_factor
            
            # --- Clip ---
            # V_max_raw = 1.0 / (1.0 - config['GAMMA_i'])
            # v_i, next_v_i = jax.tree_util.tree_map(lambda x: jnp.clip(x, 0, V_max_raw), (v_i, next_v_i))
            
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho, next_i_val=next_v_i)
            # --- GAE ---
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
            advantages = jnp.where(config.get('CENTER_ADVANTAGES', False),
                (advantages - advantages.mean()) / (advantages.std() + 1e-8),
                advantages
            )
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
            metric = _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale, scaling_factor, lstd_state, sigma_state)

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
                "obs_rms": obs_rms,
                "iret_rms": iret_rms,
                "irets": irets,
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
            "last_phi": initial_phi,            
            "last_rho_feat": initial_rho_feat,  
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
