# adds target network for the features
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math 
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
SAVE_DIR = "4_14_lstd_learned_feats"

from flax.training.train_state import TrainState

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
        torso_out = self.torso(x)
        
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
    info: jnp.ndarray

class LSTDBufferState(NamedTuple):
    observations: jnp.ndarray
    next_observations: jnp.ndarray
    features: jnp.ndarray          
    next_features: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    size: jnp.ndarray

def make_train(config):
    is_episodic = config.get("EPISODIC", True)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    terminate_lstd_bootstrap = jnp.logical_and(is_episodic, not(is_absorbing))
    
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    
    # --- Buffer Padding for Extended Collection and Chunking ---
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    CHUNK_SIZE = config.get('CHUNK_SIZE', 3 * config['NUM_ENVS'])
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    
    if batch_size % CHUNK_SIZE != 0:
        CHUNK_SIZE = math.gcd(batch_size, CHUNK_SIZE)

    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    NUM_CHUNKS_BASE = (BUFFER_CAPACITY + CHUNK_SIZE - 1) // CHUNK_SIZE
    BUFFER_CAPACITY = NUM_CHUNKS_BASE * CHUNK_SIZE # Adjusted for alignment
    
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    NUM_CHUNKS = EXTENDED_CAPACITY // CHUNK_SIZE
    PADDED_CAPACITY = NUM_CHUNKS * CHUNK_SIZE
    
    k_base = config.get("RND_FEATURES", 128)
    k_lstd = config.get("LSTD_FEATURES", 64) # when using tabular features, identity
    
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n  # <-- Needed for inverse model
    evaluator = helpers.initialize_evaluator(config)

    def get_scale_free_bonus(S_inv, features):
        """bonus = sqrt(x^T Σ^{-1} x)"""
        bonus_sq = jnp.einsum("...i,ij,...j->...", features, S_inv, features)
        return jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
    
    def update_phi_precision(lstd_state, features, next_features, done):
        phi_diag_precision = lstd_state['phi_diag_counts']
        absorb_mask = jnp.where(is_absorbing, done, 0)
        # Calculate precision for this batch
        batch_phi_prec = jnp.sum(features**2, axis=(0, 1)) 
        # Calculate precision for absorbing transitions
        absorbing_features = next_features * absorb_mask[..., None]
        abs_phi_prec = jnp.sum(absorbing_features**2, axis=(0, 1)) 
        new_counts = batch_phi_prec + abs_phi_prec
        return {**lstd_state, "phi_diag_counts": phi_diag_precision + new_counts}
    
    def update_buffer(buffer_state: LSTDBufferState, obs, next_obs, features, next_features, terminals, absorb_masks):
        # Dynamically reshape based on the true observation shape
        obs = obs.reshape((-1,) + obs_shape).astype(jnp.float32)
        next_obs = next_obs.reshape((-1,) + obs_shape).astype(jnp.float32)
        features = features.reshape(-1, k_base).astype(jnp.float32)
        next_features = next_features.reshape(-1, k_base).astype(jnp.float32)
        terminals = terminals.reshape(-1, 1).astype(jnp.float32)
        absorb_masks = absorb_masks.reshape(-1, 1).astype(jnp.float32)
        
        B = obs.shape[0]  # <-- Fixed
        start_idx = buffer_state.size
        
        new_obs = jax.lax.dynamic_update_slice(buffer_state.observations, obs, (start_idx,) + (0,)*len(obs_shape))
        new_next_obs = jax.lax.dynamic_update_slice(buffer_state.next_observations, next_obs, (start_idx,) + (0,)*len(obs_shape))
        new_features = jax.lax.dynamic_update_slice(buffer_state.features, features, (start_idx, 0))
        new_next_features = jax.lax.dynamic_update_slice(buffer_state.next_features, next_features, (start_idx, 0))
        new_terminals = jax.lax.dynamic_update_slice(buffer_state.terminals, terminals, (start_idx, 0))
        new_absorb_masks = jax.lax.dynamic_update_slice(buffer_state.absorb_masks, absorb_masks, (start_idx, 0))
        
        return LSTDBufferState(
            observations=new_obs, next_observations=new_next_obs, 
            features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=start_idx + B
        )
            
    def solve_lstd_buffer(buffer_state: LSTDBufferState, Sigma_inv, lstd_state, get_phi_lstd, config):
        # Load observationsfor the LSTD projection.
        # chunked for low memory inference.
        chunked_obs = buffer_state.observations.reshape((NUM_CHUNKS, CHUNK_SIZE) + obs_shape)
        chunked_next_obs = buffer_state.next_observations.reshape((NUM_CHUNKS, CHUNK_SIZE) + obs_shape)
        chunked_next_phi_base = buffer_state.next_features.reshape(NUM_CHUNKS, CHUNK_SIZE, -1)
        chunked_terminals = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        chunked_absorb = buffer_state.absorb_masks.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        valid_mask = (jnp.arange(PADDED_CAPACITY) < buffer_state.size)[..., None]
        chunked_mask = valid_mask.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        gamma_i = config["GAMMA_i"]

        def process_chunk(carry, chunk_data):
            A_acc, b_acc = carry
            obs_c, next_obs_c, next_phi_b, term, absorb, mask = chunk_data
            
            next_rho = get_scale_free_bonus(Sigma_inv, next_phi_b)

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

        # --- Bayesian Optimistic Prior (Diagonal) ---
        PRIOR_SAMPLES = config.get("LSTD_PRIOR_SAMPLES", 1.0)
        new_phi_diag_counts = lstd_state["phi_diag_counts"]
        lambda_k = PRIOR_SAMPLES / (PRIOR_SAMPLES + new_phi_diag_counts)
        Lambda_mat = jnp.diag(lambda_k)
        prior_b = lambda_k * lstd_state["V_max"]
        
        reg = jnp.eye(k_lstd) * config["A_REGULARIZATION_PER_STEP"]
        A_view = final_A + Lambda_mat + reg
        b_view = final_b + prior_b
        w_i = jnp.linalg.pinv(A_view, rtol=1e-8) @ b_view
        
        # tracking:
        singular_values = jnp.linalg.svd(A_view, compute_uv=False)
        cond_number = singular_values[0] / (singular_values[-1] + 1e-8)
        
        return {
            "w": w_i,
            "Beta": lstd_state["Beta"],
            "V_max": lstd_state["V_max"],
            "phi_diag_counts": new_phi_diag_counts,
            "cond_number": cond_number
        }
        
    def evict_buffer(buffer_state: LSTDBufferState, get_phi_lstd, config, rng):
        """Computes IV Trace Leverage scores and evicts the lowest-scoring points iteratively."""
        size = buffer_state.size
        obs_chunks = buffer_state.observations.reshape((NUM_CHUNKS, CHUNK_SIZE) + obs_shape)
        next_obs_chunks = buffer_state.next_observations.reshape((NUM_CHUNKS, CHUNK_SIZE) + obs_shape)
        next_phi_base = buffer_state.next_features
        terminals_chunks = buffer_state.terminals.reshape(NUM_CHUNKS, CHUNK_SIZE, 1)
        # chunked netowrk inference to get features
        def _compute_chunk_features(carry, obs_chunk):
            return None, get_phi_lstd(obs_chunk)

        _, phi_chunks = jax.lax.scan(_compute_chunk_features, None, obs_chunks)
        _, next_phi_chunks = jax.lax.scan(_compute_chunk_features, None, next_obs_chunks)
        
        # LSTD matrices:
        Z_chunks = phi_chunks
        X_chunks = phi_chunks - config["GAMMA_i"] * (1 - terminals_chunks) * next_phi_chunks
        k_val = Z_chunks.shape[-1]
        
        # --- STATIC COMPILATION FIX ---
        # Calculate exactly how many drops are needed mathematically as pure Python ints.
        # We know the buffer always overshoots by exactly `batch_size` when it gets full.
        static_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        percent_fifo = config.get("PERCENT_FIFO", 0.2)
        NUM_CUTS = config.get("EVICTION_CUTS", 4) 
        
        static_fifo_drops = int(static_batch_size * percent_fifo)
        static_prb_drops = static_batch_size - static_fifo_drops
        static_drops_per_cut = static_prb_drops // NUM_CUTS
        
        # Dynamic check: Are we actually full yet?
        buffer_is_full = size > BUFFER_CAPACITY
        
        # --- PHASE 1: FIFO Masking ---
        indices = jnp.arange(PADDED_CAPACITY)
        
        # Valid data only goes up to the current size
        valid_mask = indices < size
        
        # If the buffer is full, mark the oldest 'static_fifo_drops' as False to kill them immediately
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))
        
        def cut_step(carry, step_idx):
            mask_curr = carry
            mask_chunks = mask_curr.reshape(NUM_CHUNKS, CHUNK_SIZE)
            
            # 1. Chunked A_curr computation (Avoids materializing massive valid_Z)
            def A_chunk(carry_A, chunk_data):
                z_c, x_c, m_c = chunk_data
                # Multiply mask into one operand, saving ~1.6GB of VRAM
                A_c = jnp.einsum("ni, nj -> ij", z_c, x_c * m_c[:, None])
                return carry_A + A_c, None
                
            A_curr, _ = jax.lax.scan(A_chunk, jnp.zeros((k_val, k_val)), (Z_chunks, X_chunks, mask_chunks))
            A_curr += jnp.eye(k_val) * config.get("A_REGULARIZATION_PER_STEP", 1e-3)
            A_inv_curr = jnp.linalg.pinv(A_curr, rtol=1e-8)
            
            # 2. Chunked Score computation
            def score_chunk(carry_unused, chunk_data):
                z_c, x_c = chunk_data
                u_c = z_c @ A_inv_curr.T
                v_c = x_c @ A_inv_curr
                w_c = v_c @ A_inv_curr.T
                
                c = 1.0 - jnp.sum(x_c * u_c, axis=-1)
                c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
                
                u_norm = jnp.sum(u_c * u_c, axis=-1)
                v_norm = jnp.sum(v_c * v_c, axis=-1)
                cross = jnp.sum(u_c * w_c, axis=-1)
                
                s_c = (2.0 * cross / c) + (u_norm * v_norm) / (c * c)
                return None, s_c
                
            _, chunked_scores = jax.lax.scan(score_chunk, None, (Z_chunks, X_chunks))
            scores = chunked_scores.flatten()
            
            # Stochastic Sampling (Gumbel-Max Trick)
            drop_logits = -scores / config.get("STOCHASTIC_TEMP", 1.0)
            drop_logits = jnp.where(mask_curr, drop_logits, -jnp.inf)
            
            rng_key = jax.random.fold_in(rng, step_idx) 
            gumbel_noise = jax.random.gumbel(rng_key, drop_logits.shape)
            noisy_logits = drop_logits + gumbel_noise
            
            _, drop_indices = jax.lax.top_k(noisy_logits, static_drops_per_cut)
            
            mask_next_candidate = mask_curr.at[drop_indices].set(False)
            mask_next = jnp.where(buffer_is_full, mask_next_candidate, mask_curr)
            
            return mask_next, None

        # Run the C-cut loop (statically defined)
        final_mask, _ = jax.lax.scan(cut_step, initial_mask, jnp.arange(NUM_CUTS))
        
        # --- PHASE 3: Compaction & Remainder Trimming ---
        # selection_scores ensures surviving True masks = ~1.0, False masks = ~0.0.
        # The tie-breaker (+ indices) prefers keeping NEWER items when resolving remainders.
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, BUFFER_CAPACITY)

        # Trim buffer based on keep indicies. 
        new_obs = jnp.zeros_like(buffer_state.observations).at[:BUFFER_CAPACITY].set(buffer_state.observations[keep_indices])
        new_next_obs = jnp.zeros_like(buffer_state.next_observations).at[:BUFFER_CAPACITY].set(buffer_state.next_observations[keep_indices])
        new_features = jnp.zeros_like(buffer_state.features).at[:BUFFER_CAPACITY].set(buffer_state.features[keep_indices])
        new_next_features = jnp.zeros_like(buffer_state.next_features).at[:BUFFER_CAPACITY].set(buffer_state.next_features[keep_indices])
        new_terminals = jnp.zeros_like(buffer_state.terminals).at[:BUFFER_CAPACITY].set(buffer_state.terminals[keep_indices])
        new_absorb_masks = jnp.zeros_like(buffer_state.absorb_masks).at[:BUFFER_CAPACITY].set(buffer_state.absorb_masks[keep_indices])
        new_size = jnp.minimum(size, BUFFER_CAPACITY)

        return LSTDBufferState(
            observations=new_obs, next_observations=new_next_obs, 
            features=new_features, next_features=new_next_features,
            terminals=new_terminals, absorb_masks=new_absorb_masks,
            size=new_size
        )


    V_max = 1.0 / (1 - config["GAMMA_i"]) # maximum intrinsic values
    
    if config["NORMALIZE_FEATURES"]:
        V_max /= jnp.sqrt(k_lstd)

    def train(rng):
        initial_lstd_state = {
                    "w": jnp.zeros(k_lstd), 
                    "Beta": config["BONUS_SCALE"],
                    "V_max": V_max,
                    "phi_diag_counts": jnp.zeros(k_lstd),
                    "cond_number": 0.0
                }
            
        initial_buffer_state = LSTDBufferState(
            observations = jnp.zeros((PADDED_CAPACITY,) + obs_shape, dtype=jnp.float32), 
            next_observations = jnp.zeros((PADDED_CAPACITY,) + obs_shape, dtype=jnp.float32), 
            features = jnp.zeros((PADDED_CAPACITY, k_base), dtype=jnp.float32),
            next_features = jnp.zeros((PADDED_CAPACITY, k_base), dtype=jnp.float32),
            terminals = jnp.zeros((PADDED_CAPACITY, 1), dtype=jnp.float32),
            absorb_masks = jnp.zeros((PADDED_CAPACITY, 1), dtype=jnp.float32),
            size = jnp.array(0, dtype=jnp.int32)
        )
        
        initial_sigma_state = {"S": jnp.eye(k_base, dtype=jnp.float64)} # global accumulation

        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], bias = False, k=k_base
        )
        _, target_params = networks.initialize_rnd_network(
            target_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], bias = False, k=k_base
        )
        
        def get_lstd_feature_map(params, bias = (k_lstd > k_base)): 
            if bias:
                def _get_features(obs):
                    # Use rnd_net
                    phi_base = rnd_net.apply(params, obs) 
                    bias = jnp.ones(phi_base.shape[:-1] + (1,))
                    phi_lstd = jnp.concatenate([bias, phi_base], axis=-1)
                    return phi_lstd
            else: 
                def _get_features(obs):
                    return rnd_net.apply(params, obs) 
            
            return _get_features
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

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
            feat_train_state, train_state, lstd_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                next_i_val = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, rnd_state, env_state, obsv, rng), transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])
            
            # Intrinsic reward using Base features (Target params) 
            phi_base = batch_get_features(traj_batch.obs)
            next_phi_base = batch_get_features(traj_batch.next_obs)
            # --- GLOBAL COVARIANCE UPDATE (Pure Accumulation) --
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"]) # Cholesky solver
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_base))
            # Sigma_inv = jnp.linalg.solve(sigma_state["S"], jnp.eye(k_lstd))
            int_rew_from_features = lambda feats: get_scale_free_bonus(Sigma_inv, feats)

            # --- 2. SOLVE LSTD OVER EXTENDED BUFFER ---
            # update buffer
            terminals = jnp.where(terminate_lstd_bootstrap, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.done, 0)
            buffer_state = update_buffer(
                buffer_state, traj_batch.obs, traj_batch.next_obs, 
                phi_base, next_phi_base, terminals, absorb_masks
            )

            def get_phi_lstd(obs):
                # Use target_params for the stable manifold
                return feature_net.apply(
                    feat_train_state.target_params, 
                    obs, 
                    method=feature_net.get_lstd_features
                )
            
            batch_get_phi_lstd = jax.vmap(get_phi_lstd)
            phi_lstd = batch_get_phi_lstd(traj_batch.obs)
            next_phi_lstd = batch_get_phi_lstd(traj_batch.next_obs)

            # update precision for optimism
            lstd_state = update_phi_precision(lstd_state, phi_lstd, next_phi_lstd, traj_batch.done)
            # solve LSTD on the buffer
            lstd_state = solve_lstd_buffer(buffer_state, Sigma_inv, lstd_state, get_phi_lstd, config)

            # --- 3 & 4. SCORE AND EVICT BUFFER ---
            rng, prb_rng = jax.random.split(rng)
            buffer_state = evict_buffer(buffer_state,get_phi_lstd, config, prb_rng)

            # --- BATCH INTRINSIC VALUES FOR GAE ---
            batch_next_rho = get_scale_free_bonus(Sigma_inv, next_phi_base)

            rho_scale = lstd_state["Beta"]
            v_i = phi_lstd @ lstd_state["w"] * rho_scale
            next_v_i = next_phi_lstd @ lstd_state["w"] * rho_scale

            v_i, next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, lstd_state['V_max']) , (v_i, next_v_i) )
            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=batch_next_rho * rho_scale, next_i_val=next_v_i)

            # --- ADVANTAGE CALCULATION ---
            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"],
                is_episodic=is_episodic, is_absorbing=is_absorbing,
                γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"]
            )
            advantages = gaes[0] + gaes[1]
            (extrinsic_target, distilled_i_target) = targets
            
            # Loss function for the feature module, called in update-epoch
            def _feat_loss_fn(feat_params, obs, next_obs, actions, target_i, true_c_rnd, true_n_rnd):
                # Pass both obs and next_obs to trigger the inverse head
                v_int, _, p_c_rnd, p_n_rnd, inv_logits = feature_net.apply(feat_params, obs, next_obs)
                
                # 3(a) Distillation pulls the representation toward the LSTD fixed-point
                val_loss = 0.5 * jnp.mean(jnp.square(v_int - jax.lax.stop_gradient(target_i)))
                
                # 3(b) RND matching anchors the rank of the representation (with Delta Residual)
                c_rnd_loss = jnp.mean(jnp.square(p_c_rnd - jax.lax.stop_gradient(true_c_rnd)))
                n_rnd_loss = jnp.mean(jnp.square(p_n_rnd - jax.lax.stop_gradient(true_n_rnd)))
                
                # 3(c) Inverse Dynamics forces features to capture controllable dynamics
                action_one_hot = jax.nn.one_hot(actions, n_actions)
                inv_loss = optax.softmax_cross_entropy(logits=inv_logits, labels=action_one_hot).mean()
                
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
  
            # update covariance to incorporate new data
            sigma_state = helpers.update_cov(traj_batch, sigma_state, batch_get_features)
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
                    "lambda_k": lstd_state['phi_diag_counts'],
                    "beta": lstd_state["Beta"],
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
                    rho = int_rew_from_features(phi) * rho_scale
                    return rho

                def get_vi(obs):
                    phi_lstd = get_phi_lstd(obs)
                    return  phi_lstd @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(
                    config,
                    metric,
                    int_rew_from_state,
                    evaluator,
                    lstd_state["Beta"],
                    network,
                    train_state,
                    traj_batch,
                    get_vi,
                )

            runner_state = (feat_train_state, train_state, lstd_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (feat_train_state, train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
