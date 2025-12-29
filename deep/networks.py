import jax
from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
import jax.numpy as jnp
import distrax
from typing import Sequence, Any
from flax.training.train_state import TrainState
import distrax

class RNDTrainState(TrainState):
    target_params: Any

class RND_V(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        z = RND_Net(activation = 'relu')(x)
        z_sg = jax.lax.stop_gradient(z)
        v = nn.Dense(1, kernel_init=orthogonal(1.0))(z_sg)
        return z, v.squeeze(-1)


class RND_Net(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = nn.Dense(
            64, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(x)
        embedding = activation(embedding)
        embedding = nn.Dense(
            64, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(embedding)
        embedding = activation(embedding)

        return embedding

class Two_Head_ActorCritic(nn.Module):
    """
    3 separate dense networks (Actor, Critic1, Critic2).
    """
    action_dim: int
    activation: str = "tanh"
    normalize_value_features: bool = False

    def setup(self):
        # Resolve activation function
        if self.activation == "relu":
            self.act_fn = nn.relu
        else:
            self.act_fn = nn.tanh

        # --- Actor Layers ---
        self.actor_fc1 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.actor_fc2 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.actor_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

        # --- Critic 1 Layers ---
        self.critic1_fc1 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.critic1_fc2 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.critic1_head = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))

        # --- Critic 2 Layers ---
        self.critic2_fc1 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.critic2_fc2 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.critic2_head = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns pi, v1, v2 to match original signature"""
        pi = self.policy(x)
        v1, v2 = self.value(x)
        return pi, v1, v2

    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.get_actor_features(x)
        logits = self.actor_head(actor_features)
        return distrax.Categorical(logits=logits)

    def value(self, x: jnp.ndarray):
        """Returns V1(s), V2(s)"""
        # Get features for both critics
        c1_features, c2_features = self.get_value_features(x)
        
        # Calculate values
        v1 = self.critic1_head(c1_features)
        v2 = self.critic2_head(c2_features)
        
        return jnp.squeeze(v1, axis=-1), jnp.squeeze(v2, axis=-1)

    def get_actor_features(self, x: jnp.ndarray):
        """Passes input through Actor's hidden layers"""
        x = self.actor_fc1(x)
        x = self.act_fn(x)
        x = self.actor_fc2(x)
        x = self.act_fn(x)
        return x

    def get_value_features(self, x: jnp.ndarray):
            """Passes input through both Critics' hidden layers separately"""
            # Critic 1 trunk
            c1 = self.critic1_fc1(x)
            c1 = self.act_fn(c1)
            c1 = self.critic1_fc2(c1)
            c1 = self.act_fn(c1)

            # Critic 2 trunk
            c2 = self.critic2_fc1(x)
            c2 = self.act_fn(c2)
            c2 = self.critic2_fc2(c2)
            c2 = self.act_fn(c2)
            
            # --- L2 Normalization Logic ---
            if self.normalize_value_features:
                c1 = c1 / (jnp.linalg.norm(c1, axis=-1, keepdims=True))
                c2 = c2 / (jnp.linalg.norm(c2, axis=-1, keepdims=True))
            return c1, c2
    
    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action


class ActorCritic(nn.Module):
    """
    2 separate dense networks (Actor, Critic1).
    """
    action_dim: int
    activation: str = "tanh"
    normalize_value_features: bool = False

    def setup(self):
        # Resolve activation function
        if self.activation == "relu":
            self.act_fn = nn.relu
        else:
            self.act_fn = nn.tanh

        # --- Actor Layers ---
        self.actor_fc1 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.actor_fc2 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.actor_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

        # --- Critic 1 Layers ---
        self.critic1_fc1 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.critic1_fc2 = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.critic1_head = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns pi, v1, v2 to match original signature"""
        pi = self.policy(x)
        v1 = self.value(x)
        return pi, v1

    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.get_actor_features(x)
        logits = self.actor_head(actor_features)
        return distrax.Categorical(logits=logits)

    def value(self, x: jnp.ndarray):
        """Returns V1(s), V2(s)"""
        # Get features for both critics
        c1_features = self.get_value_features(x)
        
        # Calculate values
        v1 = self.critic1_head(c1_features)
        
        return jnp.squeeze(v1, axis=-1)

    def get_actor_features(self, x: jnp.ndarray):
        """Passes input through Actor's hidden layers"""
        x = self.actor_fc1(x)
        x = self.act_fn(x)
        x = self.actor_fc2(x)
        x = self.act_fn(x)
        return x

    def get_value_features(self, x: jnp.ndarray):
            """Passes input through both Critics' hidden layers separately"""
            # Critic 1 trunk
            c1 = self.critic1_fc1(x)
            c1 = self.act_fn(c1)
            c1 = self.critic1_fc2(c1)
            c1 = self.act_fn(c1)
            # --- L2 Normalization Logic ---
            if self.normalize_value_features:
                c1 = c1 / (jnp.linalg.norm(c1, axis=-1, keepdims=True))
            return c1
    
    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action
    
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
