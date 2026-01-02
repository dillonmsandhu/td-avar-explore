import jax.numpy as jnp
import jax
from typing import NamedTuple

def cosine_similarity(a, b):
    dot = jnp.dot(a,b)
    mag = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return dot/mag

def _get_all_traces(traj_batch, features, γ, λ):
    """Get all traces for a batch of trajectories.
    Returns: L x B x k
    """
    def get_lambda_traces(phis_s, dones, γ, λ):
        # We need to manage the carry (prev trace) separate from current trace output
        def _step_trace(trace_prev, inputs):
            phi, done = inputs
            trace_current = trace_prev * γ * λ + phi
            trace_next = trace_current * (1.0 - done)
            # Return: (carry, out)
            return trace_next, trace_current
        init_traces = jnp.zeros_like(phis_s[0]) 
        _, traces = jax.lax.scan(_step_trace, init_traces, (phis_s, dones))
        return traces 
    # Fix: Ensure dones are passed correctly to the inner function
    traces = jax.vmap(get_lambda_traces, in_axes=(1, 1, None, None))(
        features, traj_batch.done, γ, λ
    )
    return traces.transpose(1,0,2)

def _calculate_gae(traj_batch, last_val, γ, λ):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + γ * next_value * (1 - done) - value
        gae = (
            delta
            + γ * λ * (1 - done) * gae
        )
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


class Explore_Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    i_value: jnp.ndarray # extra - intrinsic value from second value head
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray # extra - intrinsic reward (RND loss)
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray # extra - to get next features
    embedding: jnp.ndarray # extra - target embedding from target rnd network
    td_error: jnp.ndarray # for OPG (outer product of gradients), the "meat" of the sandwich covariance
    info: jnp.ndarray

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class FullTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    

def _loss_fn(params, network, traj_batch, gae, targets, config):
    # RERUN NETWORK
    pi, value = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    
    # VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["VF_CLIP"], config["VF_CLIP"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)