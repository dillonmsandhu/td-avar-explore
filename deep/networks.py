import jax
from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
import jax.numpy as jnp
import distrax
from typing import Sequence

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(0.001), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    
class CNN(nn.Module):
    n_layers: int = 2
    n_channels: int= 16

    @nn.compact
    def __call__(self, x: jnp.ndarray, debug=False):
        x = x[None, ...] if x.ndim == 3 else x # add batch dim since nn.Conv expects it...
        normalize = lambda x: nn.LayerNorm()(x)
        for l in range(self.n_layers):
            x = nn.Conv(
                self.n_channels,
                kernel_size=(3, 3),
                strides=1,
                padding="VALID",
                kernel_init=nn.initializers.he_normal(),
            )(x)
            x = normalize(x)    
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        if debug:
            print(f"CNN output shape: {x.shape}") 
        return x

class PQN_AC(nn.Module):
    """Actor critic with seperate small CNNs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int = 128 # number of state features
    normalize_features: bool = False
    bias: bool = False
    
    def setup(self):
        self.k = self.final_hidden_dim
        self.a_cnn = CNN()
        self.v_cnn = CNN()
        self.a_hidden = nn.Dense(self.k, kernel_init = orthogonal(1.0), bias_init=constant(0.0))
        self.v_hidden = nn.Dense(self.k, kernel_init = orthogonal(1.0), bias_init=constant(0.0))
        self.a_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.v_weights = self.param('value_weights', nn.initializers.zeros, (self.k + int(self.bias), 1))
        
    def __call__(self, x: jnp.ndarray):
        return self.policy(x), self.value(x)
    
    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        a_features = self.get_actor_features(x)
        policy_logits = self.a_head(a_features)
        return distrax.Categorical(logits=policy_logits)
    
    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        v_features = self.get_v_features(x)
        v = jnp.einsum('...k, k->...', v_features, self.v_weights.squeeze())
        return v
    
    def get_actor_features(self, x: jnp.ndarray):
        a_features = self.a_hidden(self.a_cnn(x))
        return a_features

    def get_v_features(self, x: jnp.ndarray):
        v_features = self.v_hidden(self.v_cnn(x))
        if self.bias:
            ones = jnp.ones((*v_features.shape[:-1], 1))  # (..., 1)
            v_features = jnp.concatenate([ones, v_features], axis=-1)  # (..., k)
        if self.normalize_features:
            norm = jnp.linalg.norm(v_features, axis=-1)  # normalize
            v_features = v_features / jnp.maximum(norm[..., None], 1e-8)
        return v_features
    
    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()
