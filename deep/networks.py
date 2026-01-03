# root/networks.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
from typing import Any
import math
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
    model = RNDNet(network_type=config["NETWORK_TYPE"])
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params

def initialize_actor_critic(rng, obs_shape, action_dim, config, n_heads: int):

    if n_heads == 2:
        model = ActorCritic2Head(action_dim=action_dim,network_type=config["NETWORK_TYPE"])
    elif n_heads == 3:
        model = ActorCritic3Head(action_dim=action_dim,network_type=config["NETWORK_TYPE"])
    else:
        raise ValueError("n_heads must be 2 (standard ppo) or 3 (+ rnd intrinsic value head)")

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params


# =====================================================
# ------------------- TORSOS --------------------------
# =====================================================

class MLPTorso(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )(x)
        x = nn.relu(x)

        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )(x)
        x = nn.relu(x)
        return x

class CNNTorso(nn.Module):
    out_dim: int
    base_channels: int = 4
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
        x = nn.Dense(self.out_dim, name="proj")(x)
        x = nn.relu(x)
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

class RNDNet(nn.Module):
    network_type: str

    def setup(self):
        self.torso = make_torso(self.network_type, out_dim=64)
        # self.head = nn.Dense(64, kernel_init=orthogonal(1.0))

    def __call__(self, x):
        return self.torso(x)


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

        self.pi_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.v_head = nn.Dense(1, kernel_init=orthogonal(1.0))

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

        self.pi_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.v_ext_head = nn.Dense(1, kernel_init=orthogonal(1.0))
        self.v_int_head = nn.Dense(1, kernel_init=orthogonal(1.0))

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