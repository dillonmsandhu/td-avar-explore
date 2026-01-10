# root/networks.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
from typing import Any
import math
import optax
# =====================================================
# ---------------- ENV → NETWORK ----------------------
# =====================================================

ENV_REGISTRY = {
    "DeepSea-bsuite": {
        "NETWORK_TYPE": "cnn",
        "LR": 2.5e-4,
    },
    "SparseMountainCar-v0": {
        "NETWORK_TYPE": "mlp",
        "LR": 5e-4,
    },
}

def resolve_env_config(config):
    env_cfg = ENV_REGISTRY[config["ENV_NAME"]]
    return {**config, **env_cfg}

# =====================================================
# --------------- INITIALIZATION ----------------------
# =====================================================

def initialize_rnd_network(rng, obs_shape, config):
    model = RND_Net(network_type=config["NETWORK_TYPE"], k=128, normalize = config['NORMALIZE_FEATURES'])
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params

def initialize_actor_critic(rng, obs_shape, action_dim, config, n_heads: int):

    if n_heads == 2:
        model = ActorCritic2Head(action_dim=action_dim, network_type=config["NETWORK_TYPE"])
    elif n_heads == 3:
        model = ActorCritic3Head(action_dim=action_dim, network_type=config["NETWORK_TYPE"])
    else:
        raise ValueError("n_heads must be 2 (standard ppo) or 3 (+ rnd intrinsic value head)")

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params

def initialize_flax_train_states(config, network, rnd_net, params, rnd_params, target_params = None):
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
    rnd_state = RNDTrainState.create(
        apply_fn=rnd_net.apply,
        params=rnd_params,
        tx=tx,
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
        assert H == W, "CNNTorso assumes square inputs"

        num_downsamples = max(0, math.ceil(math.log2(H / 4)))
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

def make_torso(network_type: str, **kwargs):
    if network_type == "cnn":
        return CNNTorso(**kwargs)
    elif network_type == "mlp":
        return MLPTorso(**kwargs)
    else:
        raise ValueError(f"Unknown NETWORK_TYPE: {network_type}")


# =====================================================
# --------------------- RND ---------------------------
# =====================================================

class RNDTrainState(TrainState):
    target_params: Any

# class RND_Net(nn.Module):
#     activation: str = "tanh"

#     @nn.compact
#     def __call__(self, x):
#         if self.activation == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh
#         embedding = nn.Dense(
#             64, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
#         )(x)
#         embedding = activation(embedding)
#         embedding = nn.Dense(
#             128, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
#         )(embedding)
#         embedding = activation(embedding)

#         return embedding

class RND_Net(nn.Module):
    network_type: str
    k: int = 128
    normalize: bool = False
    
    def setup(self):
        # We output k-1 features so that after adding the bias term we have exactly k
        self.torso = make_torso(self.network_type, out_dim=self.k - 1)

    def __call__(self, x):
        phi = self.torso(x)  
        # phi = nn.relu(phi) # we want only 0> features

        if self.normalize:
            norm = jnp.linalg.norm(phi, axis=-1)  # normalize
            phi = phi / jnp.maximum(norm[..., None], 1e-8)
        
        batch_size = phi.shape[:-1]
        # bias_val = 1.0 / jnp.sqrt(self.k)
        bias_val = 1.0
        bias = jnp.ones((*batch_size, 1)) * bias_val
        
        phi = jnp.concatenate([phi, bias], axis=-1)
        
        return phi

# =====================================================
# ------------ ACTOR-CRITIC (2 HEAD) ------------------
# =====================================================

class ActorCritic2Head(nn.Module):
    """
    Returns: (pi, v)
    """
    action_dim: int
    network_type: str
    normalize_value_features: bool = False

    def setup(self):
        self.actor_torso = make_torso(self.network_type, out_dim=64)
        self.critic_torso = make_torso(self.network_type, out_dim=64)
        
        self.pi_head = nn.Sequential([nn.relu, nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))])
        self.v_head = nn.Sequential([nn.relu, nn.Dense(1, kernel_init=orthogonal(1.0))])

    def policy(self, x):
        logits = self.pi_head(self.actor_torso(x))
        return distrax.Categorical(logits=logits)

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
    normalize_value_features: bool = False

    def setup(self):
        self.actor_torso = make_torso(self.network_type, out_dim=64)
        self.critic_ext = make_torso(self.network_type, out_dim=64)
        self.critic_int = make_torso(self.network_type, out_dim=64)
        
        self.pi_head = nn.Sequential([nn.relu, nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))])
        self.v_ext_head = nn.Sequential([nn.relu, nn.Dense(1, kernel_init=orthogonal(1.0))])
        self.v_int_head = nn.Sequential([nn.relu, nn.Dense(1, kernel_init=orthogonal(1.0))])
    # ---------------- Policy ----------------

    def policy(self, x):
        logits = self.pi_head(self.actor_torso(x))
        return distrax.Categorical(logits=logits)

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