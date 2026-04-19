from core.imports import *
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import math
from gymnax.environments import spaces

# =====================================================
# --------------- INITIALIZATION ----------------------
# =====================================================

def initialize_rnd_network(rng, obs_shape, network_type, normalize_features, bias=True, k=128, state_action_features=False, n_actions=1):
    """
    Initializes the RND network. 
    If state_action_features is True, returns shape (..., n_actions, k).
    Otherwise returns shape (..., k).
    """
    model = RND_Net(
        network_type=network_type, 
        k=k, 
        normalize=normalize_features, 
        bias=bias, 
        state_action_features=state_action_features,
        n_actions=n_actions
    )
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params


def initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads: int):
    # Detect if continuous
    is_continuous = isinstance(env.action_space(env_params), spaces.Box)
    action_dim = env.action_space(env_params).shape[0] if is_continuous else env.action_space(env_params).n
    
    if n_heads == 2:
        model = ActorCritic2Head(action_dim=action_dim, network_type=config["NETWORK_TYPE"], is_continuous=is_continuous)
    elif n_heads == 3:
        model = ActorCritic3Head(action_dim=action_dim, network_type=config["NETWORK_TYPE"], is_continuous=is_continuous)
    else:
        raise ValueError("n_heads must be 2 (standard ppo) or 3 (+ rnd intrinsic value head)")

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
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

# =====================================================
# ------------------- TORSOS --------------------------
# =====================================================

class MLPTorso(nn.Module):
    hidden_dim: int = 64
    out_dim: int = 128

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )(x)
        x = nn.relu(x)

        x = nn.Dense(
            self.out_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )(x)
        return x

class CNNTorso(nn.Module):
    out_dim: int
    base_channels: int = 16
    max_channels: int = 128

    @nn.compact
    def __call__(self, x):
        """
        x: (..., H, W, C)
        """
        
        H, W = x.shape[-3], x.shape[-2]
        smaller_dim = min(H, W)
        num_downsamples = max(0, math.ceil(math.log2(smaller_dim / 4)))
        channels = self.base_channels

        for i in range(num_downsamples):
            x = nn.Conv(
                features=channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)
            channels = min(channels * 2, self.max_channels)

        x = x.reshape(*x.shape[:-3], -1)
        x = nn.Dense(self.out_dim, name="proj", kernel_init=orthogonal(1.0))(x)
        return x
    
class CNNTorso1D(nn.Module):
    out_dim: int  # e.g., 64 (size of the embedding passed to Policy/Value head)

    @nn.compact
    def __call__(self, x):
        """
        x: (Batch, Length, 1) or (Batch, Length)
        """
        
        if x.ndim <= 2:
            x = x[..., None]
        
        # Layer 1: 200 -> 100 (Stride 2)
        x = nn.Conv(features=4, kernel_size=(3,), strides=(2,), padding="SAME")(x)
        x = nn.relu(x)

        # Layer 2: 100 -> 50 (Stride 2)
        x = nn.Conv(features=8, kernel_size=(3,), strides=(2,), padding="SAME")(x)
        x = nn.relu(x)

        # Layer 3: 50 -> 25 (Stride 2)
        x = nn.Conv(features=8, kernel_size=(3,), strides=(2,), padding="SAME")(x)
        x = nn.relu(x)

        # Flatten: 25 * 8 = 200 features
        x = x.reshape(*x.shape[:-2], -1)
        
        # Final Projection to desired embedding size (e.g. 64)
        x = nn.Dense(self.out_dim)(x)
        return x
    
class Identity(nn.Module):
    out_dim: int  # e.g., 64 (size of the embedding passed to Policy/Value head)
    n_actions: int=1

    @nn.compact
    def __call__(self, x):
        # broadcast the state features to n-actions:
        if self.n_actions > 1:
            k = x.shape[-1]
            repeated = jnp.repeat(x[:,None], self.n_actions, 1)
            # final shape is (..., n_actions, out_dim)
            return repeated.reshape(*x.shape[:-1], self.n_actions, k)
        else:
            return x

def make_torso(network_type: str, **kwargs):
    if network_type == "identity":
        return Identity(**kwargs)
    if network_type == "cnn_1d":
        return CNNTorso1D(**kwargs)
    if network_type == "cnn":
        return CNNTorso(
            **kwargs,
            base_channels=16, 
            max_channels=64
        )
    elif network_type == "mlp":
        return MLPTorso(**kwargs)
    else:
        raise ValueError(f"Unknown NETWORK_TYPE: {network_type}")


# =====================================================
# --------------------- RND ---------------------------
# =====================================================

class RNDTrainState(TrainState):
    target_params: Any

class RND_Net(nn.Module):
    network_type: str
    k: int = 128
    normalize: bool = False
    bias: bool = True
    state_action_features: bool = False
    n_actions: int = 1
    
    def setup(self):
        # Base feature dimension before optional bias
        self.feat_dim = self.k - 1 if self.bias else self.k
        # If state-action, we need enough outputs for all actions
        total_out = self.feat_dim * self.n_actions if self.state_action_features else self.feat_dim
        self.torso = make_torso(self.network_type, out_dim=total_out)

    def __call__(self, x):
        phi = self.torso(x)  

        if self.state_action_features:
            # Reshape to (..., n_actions, feat_dim)
            phi = phi.reshape(*phi.shape[:-1], self.n_actions, self.feat_dim)
        
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

# =====================================================
# ------------ ACTOR-CRITIC (2 HEAD) ------------------
# =====================================================
class PolicyHead(nn.Module):
    action_dim: int
    is_continuous: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.relu(x)
        if not self.is_continuous:
            # Discrete: Output Logits
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            return distrax.Categorical(logits=logits)
        else:
            # Continuous: Output Mean and Log Std
            loc = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            return distrax.MultivariateNormalDiag(loc=loc, scale_diag=jnp.exp(log_std))

class ActorCritic2Head(nn.Module):
    """
    Returns: (pi, v)
    """
    action_dim: int
    network_type: str
    is_continuous: bool = False
    normalize_value_features: bool = False
    
    def setup(self):
        self.actor_torso = make_torso(self.network_type, out_dim=64)
        self.critic_torso = make_torso(self.network_type, out_dim=64)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim, is_continuous=self.is_continuous)
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


# =====================================================
# ------------ ACTOR-CRITIC (3 HEAD) ------------------
# =====================================================

class ActorCritic3Head(nn.Module):
    """
    Returns: (pi, v_ext, v_int)
    """
    action_dim: int
    network_type: str
    is_continuous: bool = False
    normalize_value_features: bool = False

    def setup(self):
        self.actor_torso = make_torso(self.network_type, out_dim=64)
        self.critic_ext = make_torso(self.network_type, out_dim=64)
        self.critic_int = make_torso(self.network_type, out_dim=64)
        
        self.pi_head = PolicyHead(action_dim=self.action_dim, is_continuous=self.is_continuous)
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
        self.torso = make_torso(self.network_type, out_dim=128) 

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
