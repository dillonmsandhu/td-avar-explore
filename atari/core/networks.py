from core.imports import *
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import warnings

class ImpalaStack(nn.Module):
    channels: int
    @nn.compact
    def __call__(self, x):
        # 1. Initial Processing & Downsampling
        x = nn.Conv(self.channels, kernel_size=(3, 3), strides=(1, 1),
                    kernel_init=nn.initializers.orthogonal(1.0))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        
        # 2. Two Residual Blocks
        for _ in range(2):
            residual = x
            x = nn.relu(x)
            x = nn.Conv(self.channels, (3, 3), padding="SAME",
                        kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.relu(x)
            x = nn.Conv(self.channels, (3, 3), padding="SAME",
                        kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = x + residual
        return x

class ImpalaCNN(nn.Module):
    out_dim: int = 256
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if x.ndim == 3:  # Shape (H, W, C) -> Add batch dimension
            x = x[None, ...]  # Shape becomes (1, H, W, C)
        
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        
        # Stack 1: [16 channels] -> Result: 42x42x16
        x = ImpalaStack(channels=16)(x)
        
        # Stack 2: [32 channels] -> Result: 21x21x32
        x = ImpalaStack(channels=32)(x)
        
        # Stack 3: [32 channels] -> Result: 11x11x32
        x = ImpalaStack(channels=32)(x)
        
        x = nn.relu(x) # Final activation before flattening
        x = x.reshape((x.shape[0], -1)) # Flatten
        
        # Finally linear layer
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        
        return x

class ConvTorso(nn.Module): 
    # Just the convolutions and the flatten. No dense layers. No division by 255.
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if x.ndim == 3:  
            x = x[None, ...]  
        
        # channel last for FLAX
        x = jnp.transpose(x, (0, 2, 3, 1)) 
        
        x = nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)))(x)
        
        return x.reshape((x.shape[0], -1))

# Nature DQN with a layer norm. Used for Rho.
class CNN(nn.Module): 
    out_dim: int = 448
    divide_by_255: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.divide_by_255:
            x = x / 255.0
            
        x = ConvTorso()(x)
        x = nn.activation.leaky_relu(x) # <-- Restored!

        
        # Standard PPO visual extraction
        x = nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.activation.relu(x)
        
        # Linear output for LSTD precision matrix!
        x = nn.Dense(self.out_dim, kernel_init=orthogonal(jnp.sqrt(2)), use_bias=False)(x)
        return x


# # Nature-like DQN with pooling, used for LSTD featuers.
# class LSTD_CNN(nn.Module): 
#     out_dim: int = 128
#     pool: bool = False

#     @nn.compact
#     def __call__(self, x: jnp.ndarray):

#         if x.ndim == 3:  # Shape (H, W, C) -> Add batch dimension
#             x = x[None, ...]  # Shape becomes (1, H, W, C)
        
#         x = jnp.transpose(x, (0, 2, 3, 1))
#         x = x / 255.0

#         # Input: [Batch, 84, 84, Channels]
        
#         # 1. First Layer (Nature DQN style)
#         # Formula: floor((W - K) / S) + 1
#         # floor((84 - 8) / 4) + 1 = floor(19) + 1 = 20
#         # Result: [Batch, 20, 20, 32]
#         x = nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x)
#         x = nn.activation.leaky_relu(x)
        
#         # 2. Second Layer
#         # floor((20 - 4) / 2) + 1 = floor(8) + 1 = 9
#         # Result: [Batch, 9, 9, 64]
#         x = nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x)
#         x = nn.activation.leaky_relu(x)

#         # 3. Hybrid Pooling (Smoothing)
#         # stride=(1, 1) preserves resolution but smears features across 2x2 neighbors
#         # Result: [Batch, 9, 9, 64]
#         if self.pool:
#             x = nn.avg_pool(x, window_shape=(2, 2), strides=(1, 1), padding="SAME")
        
#         # 4. Final Convolutional Compression
#         # floor((9 - 3) / 1) + 1 = 7
#         # Result: [Batch, 7, 7, 64]
#         x = nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x)
#         x = nn.activation.leaky_relu(x)
        
#         # 5. Flatten & Projection
#         # 7 * 7 * 64 = 3,136
#         x = x.reshape((x.shape[0], -1))
#         x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        
#         # 3,136 -> 128
#         x = nn.Dense(self.out_dim, use_bias=True)(x)
        
#         return x

class LSTD_Feature_Net(nn.Module):
    k: int = 512
    bias: bool = True
    layer_norm: bool = False
    
    @nn.compact
    def __call__(self, x):
        # Calculate base feature dimension depending on bias
        feat_dim = self.k - 1 if self.bias else self.k
        
        phi = ConvTorso()(x)
        phi = nn.leaky_relu(phi)
        # Note: No LayerNorm, No L2 normalization
        phi = nn.Dense(feat_dim, kernel_init=orthogonal(jnp.sqrt(2)))(phi)

        if self.layer_norm:
            phi = nn.LayerNorm(use_scale=False, use_bias=False)(phi)
        
        if self.bias:
            bias_shape = phi.shape[:-1] + (1,)
            bias_val = jnp.ones(bias_shape)
            phi = jnp.concatenate([phi, bias_val], axis=-1)
        
        return phi

# =====================================================
# --------------------- RND ---------------------------
# =====================================================

class RND_Net(nn.Module):
    k: int = 384 # same as small dino
    normalize: bool = False
    bias: bool = True
    
    def setup(self):
        # Base feature dimension before optional bias
        self.feat_dim = self.k - 1 if self.bias else self.k
        # If state-action, we need enough outputs for all actions
        self.torso = CNN(self.feat_dim, divide_by_255=False)
    def __call__(self, x):
        phi = self.torso(x)  
        
        if self.normalize:
            # Normalize along the feature dimension
            norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)
            phi = phi / jnp.maximum(norm, 1e-8)
        
        if self.bias:
            # Concatenate 1.0 to the feature dimension
            bias_shape = phi.shape[:-1] + (1,)
            bias = jnp.ones(bias_shape)
            phi = jnp.concatenate([phi, bias], axis=-1)
        
        return phi

# These two are actually used by RND, which has a more extensive predictor network.
class RND_Target(nn.Module):
    k: int = 512 # CleanRL uses 512 for RND features
    layer_norm: bool = False
    l2_norm: bool = False
    
    @nn.compact
    def __call__(self, x):
        phi = ConvTorso()(x)
        phi = nn.leaky_relu(phi)
        phi = nn.Dense(self.k, kernel_init=orthogonal(jnp.sqrt(2)))(phi)
        if self.layer_norm:
            phi = nn.LayerNorm(use_scale=False, use_bias=False)(phi)

        if self.l2_norm:
            phi_norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)
            phi = phi / (phi_norm + 1e-8)
        
        return phi

class RND_Predictor(nn.Module):
    k: int = 512
    
    @nn.compact
    def __call__(self, x):
        phi = ConvTorso()(x)
        phi = nn.leaky_relu(phi)
        
        # CleanRL Predictor: Three Dense layers with ReLUs
        phi = nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)))(phi)
        phi = nn.relu(phi)
        
        phi = nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)))(phi)
        phi = nn.relu(phi)
        
        phi = nn.Dense(self.k, kernel_init=orthogonal(jnp.sqrt(2)))(phi)
        return phi

# class LSTD_Net(nn.Module):
#     k: int = 384 # same as small dino
#     normalize: bool = False
#     bias: bool = True
#     pool: bool = False
    
#     def setup(self):
#         # Base feature dimension before optional bias
#         self.feat_dim = self.k - 1 if self.bias else self.k
#         # If state-action, we need enough outputs for all actions
#         self.torso = LSTD_CNN(self.feat_dim, self.bias)
    
#     def __call__(self, x):
#         phi = self.torso(x)  
        
#         if self.normalize:
#             # Normalize along the feature dimension
#             norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)
#             phi = phi / jnp.maximum(norm, 1e-8)
        
#         if self.bias:
#             # Concatenate 1.0 to the feature dimension
#             bias_shape = phi.shape[:-1] + (1,)
#             bias = jnp.ones(bias_shape)
#             phi = jnp.concatenate([phi, bias], axis=-1)
        
#         return phi

# =====================================================
# ------------ ACTOR-CRITIC (2 HEAD) ------------------
# =====================================================
class PolicyHead(nn.Module):
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(0.01))(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        return distrax.Categorical(logits=logits)

class ActorCritic2Head(nn.Module):
    """
    Returns: (pi, v)
    """
    action_dim: int
    normalize_value_features: bool = False
    out_dim: int = 448
    cnn_torso: str = 'IMPALA_CNN'

    def setup(self):
        if self.cnn_torso == 'IMPALA_CNN':
            self.actor_torso = ImpalaCNN(self.out_dim)
            self.critic_torso = ImpalaCNN(self.out_dim)
        else: 
            self.actor_torso = CNN(self.out_dim)
            self.critic_torso = CNN(self.out_dim)            
        
        self.pi_head = PolicyHead(action_dim=self.action_dim)
        self.v_head = nn.Sequential([nn.relu, nn.Dense(1, kernel_init=orthogonal(1.0))])

    def policy(self, x):
            return self.pi_head(self.actor_torso(x))

    def get_value_features(self, x):
        features = self.critic_torso(x)
        if self.normalize_value_features:
            features = features / (
                jnp.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
            )
        return features

    def value(self, x):
        return self.v_head(self.get_value_features(x)).squeeze(-1)

    def __call__(self, x):
        return self.policy(x), self.value(x)

class ActorCriticSharedTorso(nn.Module):
    """
    Returns: (pi, v_ext)
    Designed for strict ablation against the 3-head RND baseline.
    """
    action_dim: int
    normalize_value_features: bool = False
    out_dim: int = 448
    cnn_torso: str = 'IMPALA_CNN'

    def setup(self):
        if self.cnn_torso == 'IMPALA_CNN':
            self.torso = ImpalaCNN(self.out_dim)
        else:
            self.torso = CNN(self.out_dim)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim, hidden_dim=self.out_dim)
        
        # Retained residual network to match baseline capacity: maps 448 -> 448, std=0.1
        self.v_head_residual = nn.Sequential([
            nn.Dense(self.out_dim, kernel_init=orthogonal(0.1)), 
            nn.relu
        ])
        
        # Pure linear extrinsic value head, std=0.01
        self.v_ext_head = nn.Dense(1, kernel_init=orthogonal(0.01))
        
    def get_value_features(self, phi):
        # Calculate residual using the activated CNN features
        phi_res = self.v_head_residual(phi)
        phi_combined = phi + phi_res
        
        if self.normalize_value_features:
            phi_combined = phi_combined / (
                jnp.linalg.norm(phi_combined, axis=-1, keepdims=True) + 1e-8
            )

        return phi_combined

    def value(self, x):
        phi_raw = self.torso(x)
        phi_act = nn.relu(phi_raw)
        phi_ext = self.get_value_features(phi_act)
        v_ext = self.v_ext_head(phi_ext).squeeze(-1)
        return v_ext

    def __call__(self, x):
        # 1. Run CNN exactly once (no activation here to avoid double ReLU in heads)
        phi_raw = self.torso(x)
        
        # 2. Get Policy (PolicyHead applies its own initial ReLU)
        pi = self.pi_head(phi_raw)
        
        # 3. Get Extrinsic Value
        phi_act = nn.relu(phi_raw)
        phi_ext = self.get_value_features(phi_act)
        v_ext = self.v_ext_head(phi_ext).squeeze(-1)
        
        return pi, v_ext


# =====================================================
# ------------ ACTOR-CRITIC (3 HEAD) ------------------
# =====================================================

class ActorCritic3Head(nn.Module):
    """
    Returns: (pi, v_ext, v_int)
    """
    action_dim: int
    normalize_value_features: bool = False
    out_dim: int= 448
    cnn_torso: str = 'IMPALA_CNN'

    def setup(self):
        if self.cnn_torso == 'IMPALA_CNN':
            self.actor_torso = ImpalaCNN(self.out_dim)
            self.critic_ext = ImpalaCNN(self.out_dim)
            self.critic_int = ImpalaCNN(self.out_dim)
        else:
            self.actor_torso = CNN(self.out_dim)
            self.critic_ext = CNN(self.out_dim)
            self.critic_int = CNN(self.out_dim)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim)
        self.v_ext_head = nn.Sequential([nn.relu, nn.Dense(1, kernel_init=orthogonal(1.0))])
        self.v_int_head = nn.Sequential([nn.relu, nn.Dense(1, kernel_init=orthogonal(1.0))])
        
    # ---------------- Policy ----------------
    def policy(self, x):
        return self.pi_head(self.actor_torso(x))
        
    # ---------------- Value -----------------

    def get_value_features(self, x):
        phi_ext = self.critic_ext(x)
        phi_int = self.critic_int(x)

        if self.normalize_value_features:
            phi_ext = phi_ext / (
                jnp.linalg.norm(phi_ext, axis=-1, keepdims=True) + 1e-8
            )
            phi_int = phi_int / (
                jnp.linalg.norm(phi_int, axis=-1, keepdims=True) + 1e-8
            )

        return phi_ext, phi_int

    def get_i_value_features(self, x):
        """Returns ONLY the intrinsic value features for LSTD evaluation."""
        _, phi_int = self.get_value_features(x)
        return phi_int

    def get_e_value_features(self, x):
        """Returns ONLY the extrinsic value features (for completeness)."""
        phi_ext, _ = self.get_value_features(x)
        return phi_ext

    def value(self, x):
        phi_ext, phi_int = self.get_value_features(x)
        v_ext = self.v_ext_head(phi_ext).squeeze(-1)
        v_int = self.v_int_head(phi_int).squeeze(-1)
        return v_ext, v_int

    # ---------------- Full forward ----------

    def __call__(self, x):
        pi = self.policy(x)
        v_ext, v_int = self.value(x)
        return pi, v_ext, v_int

class ActorCritic3HeadSharedTorso(nn.Module):
    """
    Returns: (pi, v_ext, v_int)
    """
    action_dim: int
    normalize_value_features: bool = False
    out_dim: int = 448
    cnn_torso: str = 'IMPALA_CNN'

    def setup(self):
        if self.cnn_torso == 'IMPALA_CNN':
            self.torso = ImpalaCNN(self.out_dim)
        else:
            self.torso = CNN(self.out_dim)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim, hidden_dim=self.out_dim)
        
        # Residual network: maps 448 -> 448, std=0.1
        self.v_head_residual = nn.Sequential([
            nn.Dense(self.out_dim, kernel_init=orthogonal(0.1)), 
            nn.relu
        ])
        
        # Value heads: pure linear, std=0.01
        self.v_ext_head = nn.Dense(1, kernel_init=orthogonal(0.01))
        self.v_int_head = nn.Dense(1, kernel_init=orthogonal(0.01))
        
    # ---------------- Value -----------------

    def get_value_features(self, phi):
        # Calculate residual using the CNN features (phi)
        phi_res = self.v_head_residual(phi)
        phi_combined = phi + phi_res
        
        if self.normalize_value_features:
            phi_combined = phi_combined / (
                jnp.linalg.norm(phi_combined, axis=-1, keepdims=True) + 1e-8
            )

        return phi_combined, phi_combined

    def get_i_value_features(self, x):
        """Returns ONLY the intrinsic value features for LSTD evaluation."""
        phi = nn.relu(self.torso(x))
        _, phi_int = self.get_value_features(phi)
        return phi_int

    def get_e_value_features(self, x):
        """Returns ONLY the extrinsic value features (for completeness)."""
        phi = nn.relu(self.torso(x))
        phi_ext, _ = self.get_value_features(phi)
        return phi_ext
    
    def value(self, x):
        phi = nn.relu(self.torso(x))
        phi_ext, phi_int = self.get_value_features(phi)
        v_ext = self.v_ext_head(phi_ext).squeeze(-1)
        v_int = self.v_int_head(phi_int).squeeze(-1)
        return v_ext, v_int
    # ---------------- Full forward ----------

    def __call__(self, x):
        # 1. Run CNN exactly once
        phi = nn.relu(self.torso(x))
        
        # 2. Get Policy
        pi = self.pi_head(phi)
        
        # 3. Get Values
        phi_ext, phi_int = self.get_value_features(phi)
        v_ext = self.v_ext_head(phi_ext).squeeze(-1)
        v_int = self.v_int_head(phi_int).squeeze(-1)
        
        return pi, v_ext, v_int


class AC_plus_vi(nn.Module):
    """
    Returns: (pi, v_ext, v_int)
    """
    action_dim: int
    normalize_value_features: bool = False
    out_dim: int = 448
    cnn_torso: str = 'IMPALA_CNN'

    def setup(self):
        if self.cnn_torso == 'IMPALA_CNN':
            self.torso = ImpalaCNN(self.out_dim)
            self.vi_torso = ImpalaCNN(self.out_dim)
        else:
            self.torso = CNN(self.out_dim)
            self.vi_torso = CNN(self.out_dim)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim, hidden_dim=self.out_dim)
        
        # Residual network: maps 448 -> 448, std=0.1
        self.v_head_residual = nn.Sequential([
            nn.Dense(self.out_dim, kernel_init=orthogonal(0.1)), 
            nn.relu
        ])
        
        # Value heads: pure linear, std=0.01
        self.v_ext_head = nn.Dense(1, kernel_init=orthogonal(0.01))
        self.v_int_head = nn.Dense(1, kernel_init=orthogonal(0.01))
        
    # ---------------- Value -----------------
    def get_i_value_features(self, x):
        """Returns ONLY the intrinsic value features for LSTD evaluation."""
        phi = nn.relu(self.vi_torso(x))
        return phi

    def get_e_value_features(self, z):
        """Returns ONLY the extrinsic value features (for completeness)."""
        phi_res = self.v_head_residual(z)
        return z + phi_res
    
    def value(self, x):
        phi_e = self.get_e_value_features(nn.relu(self.torso(x)))
        phi_i = self.get_i_value_features(x)
        v_ext = self.v_ext_head(phi_e).squeeze(-1)
        v_int = self.v_int_head(phi_i).squeeze(-1)
        return v_ext, v_int
    # ---------------- Full forward ----------

    def __call__(self, x):
        # 1. Run CNN exactly once
        phi = nn.relu(self.torso(x))
        
        # 2. Get Policy
        pi = self.pi_head(phi)
        
        # 3. Get Values
        phi_e = self.get_e_value_features(phi)
        phi_i = self.get_i_value_features(x)
        v_ext = self.v_ext_head(phi_e).squeeze(-1)
        v_int = self.v_int_head(phi_i).squeeze(-1)
        
        return pi, v_ext, v_int

# =====================================================
# --------------- INITIALIZATION ----------------------
# =====================================================

class RNDTrainState(TrainState):
    target_params: Any
    
def initialize_rnd_network(rng, obs_shape, normalize_features, bias=True, k=128):
    """
    Initializes the RND network. 
    If state_action_features is True, returns shape (..., n_actions, k).
    Otherwise returns shape (..., k).
    """
    raise NotImplementedError()
    model = RND_Net(
        k=k, 
        normalize=normalize_features, 
        bias=bias, 
    )
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *obs_shape), dtype = jnp.float32)
    params = model.init(init_rng, init_x)
    return model, params


def initialize_lstd_network(rng, 
    obs_shape, 
    normalize_features, 
    bias=True, k=512, pool=False, layer_norm=False
):
    """
    Initializes the random network for LSTD. 
    If state_action_features is True, returns shape (..., n_actions, k).
    Otherwise returns shape (..., k).
    """
    if normalize_features or pool: warnings.warn("normalize features and pooling automatically false in initialize_lstd_network")
    model = LSTD_Feature_Net(
        k=k, 
        bias=bias, 
        layer_norm = layer_norm
    )
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *obs_shape), dtype = jnp.float32)
    params = model.init(init_rng, init_x)
    return model, params

def initialize_actor_critic(rng, obs_shape, action_dim, n_heads: int, cnn_torso = 'IMPALA_CNN', shared_torso=True, seperate_vi = False):
    if n_heads == 1:
        model = Actor1Head(action_dim=action_dim, cnn_torso = cnn_torso)
    elif n_heads == 2:
        model = ActorCritic2Head(action_dim=action_dim, cnn_torso = cnn_torso)
        if shared_torso:
            model = ActorCriticSharedTorso(action_dim = action_dim, cnn_torso = cnn_torso)
    elif n_heads == 3:
        if shared_torso:
            if seperate_vi: 
                model = AC_plus_vi(action_dim=action_dim, cnn_torso = cnn_torso)
            else:
                model = ActorCritic3HeadSharedTorso(action_dim=action_dim, cnn_torso = cnn_torso)
        else:
            model = ActorCritic3Head(action_dim=action_dim, cnn_torso = cnn_torso)
    else:
        raise ValueError("n_heads must be 2 (standard ppo) or 3 (+ rnd intrinsic value head)")

    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *obs_shape), dtype = jnp.float32)
    params = model.init(init_rng, init_x)
    return model, params

def initialize_flax_train_states(config, network, rnd_net, params, rnd_params, target_params=None):
    # --- PPO Agent Scheduler & Optimizer ---
    total_grad_steps = config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=config["LR_END"],
        transition_steps=total_grad_steps
    )
    tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(lr_scheduler, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    rnd_lr = config.get("RND_LR", 1e-4) 
    rnd_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(rnd_lr),
    )
    
    rnd_state = RNDTrainState.create(
        apply_fn=rnd_net.apply,
        params=rnd_params,
        tx=rnd_tx,
        target_params=target_params,
    )
    return train_state, rnd_state


def initialize_single_train_state(config, network, params):
    # --- PPO Agent Scheduler & Optimizer ---
    total_grad_steps = config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=config["LR_END"],
        transition_steps=total_grad_steps
    )
    tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(lr_scheduler, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return train_state
# Learned Features for LSTD:
class FeatureNet(nn.Module):
    """
    Dedicated network for Expressive LSTD and Intrinsic Value computation.
    Returns: (v_int, phi_lstd, current_rnd_pred, next_rnd_pred)
    """
    network_type: str
    k_rnd: int = 128      # Dimension of the fixed RND target
    k_lstd: int = 64      # Dimension of the final features passed to LSTD solver
    lstd_bias: bool = True

    def setup(self):
        # 1. Dedicated Intrinsic Torso
        self.torso = ImpalaCNN(out_dim=128) 

        # 2. LSTD Feature Component
        # Determine the number of learned features so the final concatenated output equals k_lstd
        learned_dim = self.k_lstd - 1 if self.lstd_bias else self.k_lstd
        self.lstd_feature_layer = nn.Dense(learned_dim, kernel_init=orthogonal(jnp.sqrt(2)))
        
        # Disable internal bias to force a pure dot product (mirroring LSTD)
        self.v_int_head = nn.Dense(1, use_bias=False, kernel_init=orthogonal(1.0))

        # 3. Rank-Preserving Regularization Heads
        self.current_rnd_head = nn.Dense(self.k_rnd, kernel_init=orthogonal(1.0))
        self.next_rnd_head = nn.Dense(self.k_rnd, kernel_init=orthogonal(1.0))

    def get_lstd_features(self, x):
        """
        Fast-path for LSTD matrix construction. 
        """
        z_int = nn.relu(self.torso(x))
        phi_lstd = self.lstd_feature_layer(z_int) # Pure linear projection
        
        if self.lstd_bias:
            bias = jnp.ones(phi_lstd.shape[:-1] + (1,))
            phi_lstd = jnp.concatenate([phi_lstd, bias], axis=-1)
            
        return phi_lstd

    def __call__(self, x):
        # --- Base Representation ---
        z_int = nn.relu(self.torso(x))
        
        # --- Value Features ---
        phi_lstd = self.lstd_feature_layer(z_int)
        
        if self.lstd_bias:
            # Explicit bias appendage to match LSTD matrix dimensions exactly
            bias = jnp.ones(phi_lstd.shape[:-1] + (1,))
            phi_lstd = jnp.concatenate([phi_lstd, bias], axis=-1)
        
        # Pure dot product (No ReLU, no internal layer bias)
        v_int = self.v_int_head(phi_lstd).squeeze(-1)

        # --- Regularization Predictions ---
        current_rnd_pred = self.current_rnd_head(z_int)
        next_rnd_pred = self.next_rnd_head(z_int)

        return v_int, phi_lstd, current_rnd_pred, next_rnd_pred

# Extrinsic value from LSTD

class Actor1Head(nn.Module):
    """
    Returns: (pi, v)
    """
    action_dim: int
    out_dim: int = 384
    cnn_torso: str = 'IMPALA_CNN'

    def setup(self):
        if self.cnn_torso == 'IMPALA_CNN':
            self.actor_torso = ImpalaCNN(self.out_dim)
        else:
            self.actor_torso = CNN(self.out_dim)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim)

    def policy(self, x):
            return self.pi_head(self.actor_torso(x))

    def __call__(self, x):
        return self.policy(x)

def basic_flax_train_state(config, network, params):
    # --- PPO Agent Scheduler & Optimizer ---
    total_grad_steps = config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=config["LR_END"],
        transition_steps=total_grad_steps
    )
    tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(lr_scheduler, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return train_state