# adds target network for the features
# also adds value network loss clipping
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math 
from core.imports import *
from gymnax.environments import spaces
import core.helpers as helpers
import core.networks as networks
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
SAVE_DIR = "4_16_lstd_learned_feats"
# jax.config.update("jax_enable_x64", True)
from flax.training.train_state import TrainState
from core.buffer import DynamicFeatureBufferManager, DynamicBufferState

class Transition(NamedTuple):
    done: jnp.ndarray
    goal: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray
    i_val_net: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

class TargetTrainState(TrainState):
    """Extension of TrainState to include EMA/Target parameters."""
    target_params: flax.core.FrozenDict  # or flax.core.FrozenDict
    ema_decay: float = 0.95

    def apply_ema(self):
        """Perform the convex combination update: target = d*target + (1-d)*online."""
        new_target = jax.tree_util.tree_map(
            lambda t, o: self.ema_decay * t + (1.0 - self.ema_decay) * o,
            self.target_params,
            self.params
        )
        return self.replace(target_params=new_target)

class FeatureNet(nn.Module):
    """
    Dedicated network for Expressive LSTD and Intrinsic Value computation.
    Returns: (v_int, phi_lstd, current_rnd_pred, next_rnd_pred, inv_logits)
    """
    network_type: str
    n_actions: int        
    k_rnd: int = 128      
    k_lstd: int = 64      
    lstd_bias: bool = True

    def setup(self):
        # 1. Dedicated Intrinsic Torso (May output strictly positive values if it ends in ReLU)
        self.torso = networks.make_torso(self.network_type, out_dim=128)

        # --- LSTD FEATURE COMPONENT (Stable Isotropic Projection) ---
        # LayerNorm forces the torso output to be zero-mean before the linear projection
        self.lstd_norm = nn.LayerNorm()
        learned_dim = self.k_lstd - 1 if self.lstd_bias else self.k_lstd
        
        # Use orthogonal(1.0) because there is no ReLU destroying half the variance after this layer
        self.lstd_feature_layer = nn.Dense(learned_dim, use_bias=False, kernel_init=orthogonal(1.0))
        self.v_int_head = nn.Dense(1, use_bias=False, kernel_init=orthogonal(1.0))

        # --- AUXILIARY TASK HEADS ---
        # Create a shared nonlinear bottleneck for auxiliary tasks to isolate 
        # their gradients slightly from the LSTD projection space.
        self.aux_bottleneck = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)))
        
        self.current_rnd_head = nn.Dense(self.k_rnd, kernel_init=orthogonal(1.0))
        
        self.delta_rnd_head = nn.Sequential([
            nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2))),
            nn.relu,
            nn.Dense(self.k_rnd, kernel_init=orthogonal(0.01)) 
        ])
        
        self.inverse_head = nn.Sequential([
            nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2))),
            nn.relu,
            nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))
        ])

    def _get_normalized_lstd_features(self, x):
        """Internal helper to guarantee stable LSTD geometry."""
        torso_out = nn.remat(lambda model, val: model(val))(self.torso, x)
        
        # 1. Recenter to zero-mean to avoid collinear "cones" of features
        z_centered = self.lstd_norm(torso_out)
        
        # 2. Orthogonal projection into LSTD space
        phi_raw = self.lstd_feature_layer(z_centered) 
        
        # 3. Spherical normalization
        phi_norm = phi_raw / (jnp.linalg.norm(phi_raw, axis=-1, keepdims=True) + 1e-8)
        
        if self.lstd_bias:
            bias = jnp.ones(phi_norm.shape[:-1] + (1,))
            phi_lstd = jnp.concatenate([phi_norm, bias], axis=-1)
        else:
            phi_lstd = phi_norm
            
        return phi_lstd, torso_out

    def get_lstd_features(self, x):
        """Fast-path for LSTD matrix construction."""
        phi_lstd, _ = self._get_normalized_lstd_features(x)
        return phi_lstd

    def i_value(self, x):
        """Fast-path for getting the network's intrinsic value prediction (PPO Anchor)."""
        phi_lstd, _ = self._get_normalized_lstd_features(x)
        return self.v_int_head(phi_lstd).squeeze(-1)

    def __call__(self, x, next_x=None):
        phi_lstd, torso_out = self._get_normalized_lstd_features(x)
        v_int = self.v_int_head(phi_lstd).squeeze(-1)

        # Route auxiliary tasks through the nonlinear bottleneck
        z_aux = nn.relu(self.aux_bottleneck(torso_out))

        # --- Residual Forward Prediction ---
        current_rnd_pred = self.current_rnd_head(z_aux)
        delta_pred = self.delta_rnd_head(z_aux)
        next_rnd_pred = current_rnd_pred + delta_pred

        # --- Inverse Dynamics Prediction ---
        inv_logits = None
        if next_x is not None:
            phi_next_lstd, _ = self._get_normalized_lstd_features(next_x)
            # Stop gradient on next_state features to prevent representation collapse
            g_input = jnp.concatenate([phi_lstd, jax.lax.stop_gradient(phi_next_lstd)], axis=-1)
            inv_logits = self.inverse_head(g_input)

        return v_int, phi_lstd, current_rnd_pred, next_rnd_pred, inv_logits

# Inverse loss (flexible to continuous / discrete action types)
def compute_inv_loss(inv_pred, actions, is_continuous: bool):
    """
    Computes inverse dynamics loss.
    Discrete: Cross-entropy over action categories.
    Continuous: Combined MSE and Cosine Distance for directional robustness.
    """
    if is_continuous:
        # 1. Coordinate-wise error (Magnitude)
        mse_loss = jnp.square(inv_pred - actions).mean()
        
        # 2. Directional error (Cosine Distance)
        # We normalize to unit vectors to ensure the loss focuses on 
        # how well the features predict the 'direction' of the action.
        inv_unit = inv_pred / (jnp.linalg.norm(inv_pred, axis=-1, keepdims=True) + 1e-8)
        act_unit = actions / (jnp.linalg.norm(actions, axis=-1, keepdims=True) + 1e-8)
        
        # Cosine similarity is 1.0 when perfect, -1.0 when opposite.
        cosine_sim = jnp.sum(inv_unit * act_unit, axis=-1)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Summing them ensures features capture both the "what" and "how much"
        return mse_loss + cosine_loss
    
    else:
        # Discrete Case: Standard Categorical Cross-Entropy
        # Note: actions is expected to be a batch of integer indices
        n_actions = inv_pred.shape[-1]
        labels = jax.nn.one_hot(actions, n_actions)
        return optax.softmax_cross_entropy(logits=inv_pred, labels=labels).mean()


def make_train(config):
    # Episodic / Continuing / Absorbing
    is_episodic = config.get("EPISODIC", True)
    is_continuing = (not is_episodic)
    is_absorbing = config.get("ABSORBING_GOAL_STATE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'
    
    k_base = config.get("RND_FEATURES", 128)
    k_lstd = config.get("LSTD_FEATURES", 64) # when using tabular features, identity
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    action_space = env.action_space(env_params)
    
    if isinstance(action_space, spaces.Discrete):
        n_actions = action_space.n
        is_continuous = False
    elif isinstance(action_space, spaces.Box):
        # For Box spaces, action_dim is the size of the continuous vector
        n_actions = action_space.shape[0] if len(action_space.shape) > 0 else 1
        is_continuous = True
    else:
        raise ValueError(f"Unsupported Gymnax action space: {type(action_space)}")
    
    evaluator = helpers.initialize_evaluator(config)

    # Replay Buffer
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    mb_size = config["MINIBATCH_SIZE"]
    # 2. Round the buffer size down to the nearest perfect multiple of mb_size (gives e.g. 100,000 -> 99,840)
    aligned_base = (BUFFER_CAPACITY // mb_size) * mb_size 
    # 3. Add your batch size. Because both numbers are perfect multiples of mb_size,
    # the total CHUNK_SIZE is mathematically guaranteed to reshape perfectly.
    config['CHUNK_SIZE'] = aligned_base + batch_size

    buffer_manager = DynamicFeatureBufferManager(
        config, k_base, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE'], obs_shape
    )
    config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    config['PADDED_CAPACITY'] = buffer_manager.padded_capacity
    
    if config.get('SCHEDULE_BETA', False):
        # goes up until peak and then linearly decays to 0.
        beta_sch = helpers.make_triangle_schedule(total_updates = config['NUM_UPDATES'], max_beta=config['BONUS_SCALE'], peak_at=0.01) 
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    def solve_lstd_buffer(buffer_state: DynamicBufferState, Sigma_inv, lstd_state, get_phi_lstd, config):
        # Load observationsfor the LSTD projection.
        # chunked for low memory inference.

        N = buffer_state.size
        # Reshape buffer into chunks
        num_chunks = config['NUM_CHUNKS']
        chunk_size = config['CHUNK_SIZE']
        padded_capacity = config['PADDED_CAPACITY']
        
        chunked_obs = buffer_state.observations.reshape((num_chunks, chunk_size) + obs_shape)
        chunked_next_obs = buffer_state.next_observations.reshape((num_chunks, chunk_size) + obs_shape)
        chunked_next_phi_base = buffer_state.next_features.reshape(num_chunks, chunk_size, -1)
        chunked_terminals = buffer_state.terminals.reshape(num_chunks, chunk_size, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(num_chunks, chunk_size, 1)
        valid_mask = (jnp.arange(padded_capacity) < buffer_state.size)[..., None]
        chunked_mask = valid_mask.reshape(num_chunks, chunk_size, 1)
        gamma_i = config["GAMMA_i"]

        def process_chunk(carry, chunk_data):
            A_acc, b_acc = carry
            obs_c, next_obs_c, next_phi_b, term, absorb, mask = chunk_data
            
            next_rho = helpers.get_scale_free_bonus(Sigma_inv, next_phi_b)

            # vmap the feature map over the chunked observations
            phi_lstd = get_phi_lstd(obs_c)
            next_phi_lstd = get_phi_lstd(next_obs_c)
            traces_lstd = phi_lstd # LSTD(0)

            delta_Phi = phi_lstd - gamma_i * (1 - term) * next_phi_lstd
            A_batch = jnp.einsum("ni, nj -> ij", traces_lstd, delta_Phi)
            b_batch = jnp.einsum("ni, n -> i", traces_lstd, next_rho * mask.squeeze())
            
            abs_features = next_phi_lstd * absorb
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", abs_features, abs_features)
            b_abs = jnp.einsum("ni, n -> i", abs_features, next_rho * absorb.squeeze() * mask.squeeze())
            
            return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs), None

        # Construct LSTD system by scanning over chunks.
        init_A = jnp.zeros((k_lstd, k_lstd))
        init_b = jnp.zeros(k_lstd)
        
        (final_A, final_b), _ = jax.lax.scan(
            process_chunk, 
            (init_A, init_b), 
            (chunked_obs, chunked_next_obs, chunked_next_phi_base, chunked_terminals, chunked_absorb, chunked_mask)
        )
        
        reg = jnp.eye(k_lstd) * config["LSTD_L2_REG"] * buffer_state.size
        A_view = final_A + reg
        w_i = jnp.linalg.solve(A_view, final_b) 
        
        # tracking:
        singular_values = jnp.linalg.svd(A_view, compute_uv=False)
        cond_number = singular_values[0] / (singular_values[-1] + 1e-8)
        
        return {
            "w": w_i,
            "cond_number": cond_number
        }

    def train(rng):
        initial_lstd_state = {
                    "w": jnp.zeros(k_lstd), 
                    "cond_number": 0.0
        }
            
        initial_buffer_state = buffer_manager.init_state()
        initial_sigma_state = {"S": jnp.eye(k_base, dtype=jnp.float64)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], bias = False, k=k_base
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], bias = False, k=k_base
        )
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        # batch_get_features = jax.vmap(get_features_fn)

        # For RND, fixed function.
        def batch_get_features(obs_full):
            # Identify how many dimensions belong to the observation itself
            num_obs_dims = len(obs_shape)
            batch_dims = obs_full.shape[:-num_obs_dims]
            
            # Flatten all batch dimensions into a single flat batch
            obs_flat = obs_full.reshape((-1,) + obs_shape)
            total_steps = obs_flat.shape[0]
            net_chunk = config["MINIBATCH_SIZE"] 
            n_chunks = total_steps // net_chunk
            obs_reshaped = obs_flat.reshape((n_chunks, net_chunk) + obs_shape)
            
            def _base_scan_step(unused, x_chunk):
                return None, get_features_fn(x_chunk)

            _, phi_out = jax.lax.scan(_base_scan_step, None, obs_reshaped)
            
            # Dynamically restore whatever batch dimensions we started with
            return phi_out.reshape(batch_dims + (-1,))

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        
        # initialize feature network for the LSTD features
        feature_net = FeatureNet(
            network_type=config["RND_NETWORK_TYPE"], 
            n_actions=n_actions, 
            k_rnd=k_base, 
            k_lstd=k_lstd
        )
        rng, feat_rng = jax.random.split(rng)
        feature_params = feature_net.init(feat_rng, jnp.zeros(obs_shape), jnp.zeros(obs_shape))
        # feature_params = feature_net.init(feat_rng, jnp.ones((1,) + obs_shape))
        # Standard Adam (no weight decay), Zero Momentum (b1=0), High Epsilon
        feat_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=1e-5)
        )
        feat_train_state = TargetTrainState.create(
            apply_fn=feature_net.apply,
            params=feature_params,
            target_params=feature_params, # Initialize target = online
            tx=feat_tx,
            ema_decay=config.get("EMA_FEAT_NET", 0.95)
        )
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            feat_train_state, train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                is_goal = info['is_goal']
                target_next_obs = info["real_next_obs"].reshape(last_obs.shape)
                dummy = jnp.zeros_like(reward)

                next_val = network.apply(train_state.params, target_next_obs, method=network.value)
                i_val_net = feature_net.apply(feat_train_state.params, last_obs, method = feature_net.i_value)

                transition = Transition(
                    done, is_goal, action, value, next_val, dummy, dummy, i_val_net, reward, dummy, log_prob, last_obs, target_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])
            
            # Intrinsic reward using Base features (Target params) 
            phi_base = batch_get_features(traj_batch.obs)
            next_phi_base = batch_get_features(traj_batch.next_obs)
            # --- GLOBAL COVARIANCE UPDATE (Pure Accumulation) --
            sigma_state = helpers.update_cov(traj_batch, sigma_state, phi_base, next_phi_base)            

            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_base))
            
            # --- 2. Update Buffer ---
            terminals = jnp.where(not is_continuing, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.goal, 0) # Fixed from .done to .goal
            
            # Package the batch into the NamedTuple
            buffer_batch = DynamicBufferState(
                observations=traj_batch.obs,
                next_observations=traj_batch.next_obs,
                features=phi_base,
                next_features=next_phi_base,
                terminals=terminals,
                absorb_masks=absorb_masks,
                size=jnp.array(0) # Dummy size for the batch tuple
            )
            
            # Call the inherited method
            buffer_state = buffer_manager.update_buffer(buffer_state, buffer_batch)

            def get_phi_lstd(obs):
                # Use target_params for the stable manifold
                return feature_net.apply(
                    feat_train_state.target_params, 
                    obs, 
                    method=feature_net.get_lstd_features
                )
            # batch_get_phi_lstd = jax.vmap(get_phi_lstd)
            
            def batch_get_phi_lstd(obs_full):
                num_obs_dims = len(obs_shape)
                batch_dims = obs_full.shape[:-num_obs_dims]
                obs_flat = obs_full.reshape((-1,) + obs_shape)
                total_steps = obs_flat.shape[0]
                net_chunk = config["MINIBATCH_SIZE"] 
                n_chunks = total_steps // net_chunk
                obs_reshaped = obs_flat.reshape((n_chunks, net_chunk) + obs_shape)

                def _feat_scan_step(unused, x_chunk):
                    return None, get_phi_lstd(x_chunk)

                _, phi_out = jax.lax.scan(_feat_scan_step, None, obs_reshaped)
                
                return phi_out.reshape(batch_dims + (-1,))
            
            phi_lstd = batch_get_phi_lstd(traj_batch.obs)
            next_phi_lstd = batch_get_phi_lstd(traj_batch.next_obs)

            # solve LSTD on the buffer
            lstd_state = solve_lstd_buffer(buffer_state, Sigma_inv, lstd_state, batch_get_phi_lstd, config)

            # --- 3 & 4. SCORE AND EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, batch_get_phi_lstd, prb_rng)

            # --- BATCH INTRINSIC VALUES FOR GAE ---
            rho = helpers.get_scale_free_bonus(Sigma_inv, next_phi_base)

            # --- 1. RAW LSTD PREDICTIONS ---
            # Do NOT apply rho_scale here. 
            v_i = phi_lstd @ lstd_state["w"] 
            next_v_i = next_phi_lstd @ lstd_state["w"] 

            # --- 2. RAW CLIPPING ---
            # Clip to the pure mathematical bound (no rho_scale needed)
            V_max_raw = 1.0 / (1.0 - config['GAMMA_i'])
            v_i, next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, V_max_raw), (v_i, next_v_i))

            # --- 3. EXACT RAW ABSORBING OVERRIDE ---
            # --- Absorbing overwrite ---
            # exact_terminal_i_val = rho / (1.0 - config["GAMMA_i"])
            # overwrite_val = jnp.logical_and(traj_batch.goal, is_absorbing)
            # fixed_next_i_val = jnp.where(overwrite_val, exact_terminal_i_val, next_v_i)
            # traj_batch = traj_batch._replace(
            #     i_value=v_i, 
            #     intrinsic_reward=rho, 
            #     next_i_val=fixed_next_i_val
            # )

            # --- Final traj_batch update for GAE ---
            traj_batch = traj_batch._replace(
                i_value=v_i, 
                intrinsic_reward=rho, 
                next_i_val=next_v_i
            )

            # -------------------------------------------------------------
            # --------- ADVANTAGE CALCULATION (Unified Absorbing) ---------
            gaes, targets = helpers.calculate_gae(
                traj_batch, 
                config["GAMMA"], 
                config["GAE_LAMBDA"], 
                is_continuing,
                γi=config["GAMMA_i"], 
                λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes
            
            # --- 5. POST-GAE SCALING ---
            rho_scale = beta_sch(idx) # triangle schedule
            advantages = gae_e + (rho_scale * gae_i)
            (extrinsic_target, distilled_i_target) = targets
                    
            # Loss function for the feature module, called in update-epoch
            def _feat_loss_fn(feat_params, obs, next_obs, actions, target_i, i_val_net, true_c_rnd, true_n_rnd):
                # Pass both obs and next_obs to trigger the inverse head
                v_int, _, p_c_rnd, p_n_rnd, inv_pred = feature_net.apply(feat_params, obs, next_obs)
                
                # --- 3(a) Distillation with PPO-Style Clipping ---
                clip_eps = config.get("FEAT_CLIP_EPS", 0.2)
                
                # 1. Unclipped squared error
                v_loss_unclipped = jnp.square(v_int - jax.lax.stop_gradient(target_i))
                
                # 2. Clipped prediction (bounded to a trust region around the old prediction)
                v_clipped = i_val_net + jnp.clip(v_int - i_val_net, -clip_eps, clip_eps)
                
                # 3. Clipped squared error
                v_loss_clipped = jnp.square(v_clipped - jax.lax.stop_gradient(target_i))
                
                # 4. Take the max (pessimistic bound) to clip the gradient
                val_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
                
                # 3(b) RND matching anchors the rank of the representation (with Delta Residual)
                c_rnd_loss = jnp.mean(jnp.square(p_c_rnd - jax.lax.stop_gradient(true_c_rnd)))
                n_rnd_loss = jnp.mean(jnp.square(p_n_rnd - jax.lax.stop_gradient(true_n_rnd)))

                # 3(c) Inverse Dynamics forces features to capture controllable dynamics
                inv_loss = compute_inv_loss(inv_pred, actions, is_continuous)

                # Weighted composition based on ablation config
                total_feat_loss = (config.get("VAL_LOSS_WEIGHT", 1.0) * val_loss) + \
                                  (config.get("RND_LOSS_WEIGHT", 1.0) * (c_rnd_loss + n_rnd_loss)) + \
                                  (config.get("INV_LOSS_WEIGHT", 1.0) * inv_loss)
                                  
                return total_feat_loss, (val_loss, c_rnd_loss, n_rnd_loss, inv_loss)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(states, batch_info):
                    train_state, feat_train_state = states
                    mb_traj, mb_adv, mb_ext_target, mb_int_target, mb_true_c_rnd, mb_true_n_rnd = batch_info
                    
                    # --- 1. Actor-Critic Update ---
                    ac_grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (ac_loss, ac_aux_losses), ac_grads = ac_grad_fn(
                        train_state.params, network, mb_traj, mb_adv, mb_ext_target, config
                    )
                    train_state = train_state.apply_gradients(grads=ac_grads)

                    # --- 2. FeatureNet Update ---
                    feat_grad_fn = jax.value_and_grad(_feat_loss_fn, has_aux=True)
                    (feat_loss, feat_aux_losses), feat_grads = feat_grad_fn(
                        feat_train_state.params, 
                        mb_traj.obs, 
                        mb_traj.next_obs,  
                        mb_traj.action,    
                        mb_int_target, 
                        mb_traj.i_val_net,
                        mb_true_c_rnd, 
                        mb_true_n_rnd
                    )
                    feat_train_state = feat_train_state.apply_gradients(grads=feat_grads)

                    return (train_state, feat_train_state), (ac_loss, ac_aux_losses, feat_loss, feat_aux_losses)
                
                # Unpack the joint update state
                train_state, feat_train_state, traj_batch, advantages, ext_targets, int_targets, true_c_rnd, true_n_rnd, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Bundle all necessary data for the dual update
                batch = (traj_batch, advantages, ext_targets, int_targets, true_c_rnd, true_n_rnd)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                (train_state, feat_train_state), losses = jax.lax.scan(_update_minbatch, (train_state, feat_train_state), minibatches)
                
                update_state = (train_state, feat_train_state, traj_batch, advantages, ext_targets, int_targets, true_c_rnd, true_n_rnd, rng)
                return update_state, losses
            # end update_epoch

            # Initialize update with BOTH train states and the base RND targets
            initial_update_state = (
                train_state, feat_train_state, traj_batch, advantages, extrinsic_target, distilled_i_target, phi_base, next_phi_base, rng
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, feat_train_state, _, _, _, _, _, _, rng = update_state
            ac_loss, ac_aux_info, feat_loss, feat_aux_info = loss_info
            feat_train_state = feat_train_state.apply_ema()
  
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            # Shared Metrics
            metric.update(
                {
                    "ppo_loss": loss_info[0].mean(),
                    "e_value_loss": ac_aux_info[0].mean(),
                    "pi_loss": ac_aux_info[1].mean(),
                    "entropy": ac_aux_info[2].mean(),
                    "feat_total_loss": feat_loss.mean(),
                    "feat_distill_loss": feat_aux_info[0].mean(),
                    "inv_loss": feat_aux_info[3].mean(),
                    "rnd_c_loss": feat_aux_info[1].mean(),
                    "rnd_n_loss": feat_aux_info[2].mean(),            
                    "feat_norm": jnp.linalg.norm(next_phi_lstd, axis=-1).mean(),
                    "bonus_mean": gaes[1].mean(),
                    "bonus_std": gaes[1].std(),
                    "bonus_max": gaes[1].max(),
                    "lambda_ret_mean": targets[0].mean(),
                    "lambda_ret_std": targets[0].std(),
                    "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                    "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                    "mean_rew": traj_batch.reward.mean(),
                    "rho_scale": rho_scale,
                    'condition_number': lstd_state['cond_number']
                }
            )

            if evaluator is None:  # No way to compute true values, just record the batch average prediction.
                metric.update(
                    {
                        "vi_pred": traj_batch.i_value.mean(),
                        "v_e_pred": traj_batch.value.mean(),
                    }
                )
            else:
                def int_rew_from_state(s,):  # for computing the intrinsic reward given an arbitrary state
                    phi = batch_get_features(s)
                    rho = helpers.get_scale_free_bonus(Sigma_inv, phi) * rho_scale
                    return rho

                def get_vi(obs):
                    phi_lstd = batch_get_phi_lstd(obs)
                    return  phi_lstd @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(
                    config,
                    metric,
                    int_rew_from_state,
                    evaluator,
                    rho_scale,
                    network,
                    train_state,
                    traj_batch,
                    get_vi,
                )

            runner_state = (feat_train_state, train_state, lstd_state, sigma_state, buffer_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (feat_train_state, train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
