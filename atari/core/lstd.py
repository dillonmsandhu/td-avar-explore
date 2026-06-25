# lstd.py
# contains code to solve LSTD with a replay buffer
import jax
import jax.numpy as jnp
from core.buffer import LSTDBufferState,LSTDBufferStateE
from core.helpers import get_scale_free_bonus

def solve_lstd_lambda_from_buffer(buffer: LSTDBufferState, Sigma_inv, config):
    """Solves LSTD over the entire extended buffer using a memory-safe chunked scan."""
    N = buffer.size
    chunk_size = config['CHUNK_SIZE']
    num_chunks = config['NUM_CHUNKS']
    padded_capacity = config['PADDED_CAPACITY']
    gamma_i = config["GAMMA_i"]

    # 1. Reshape all required arrays
    chunked_phi = buffer.features.reshape(num_chunks, chunk_size, -1)
    chunked_next_phi = buffer.next_features.reshape(num_chunks, chunk_size, -1)
    chunked_rho_feats = buffer.rho_features.reshape(num_chunks, chunk_size, -1)
    chunked_next_rho_feats = buffer.next_rho_features.reshape(num_chunks, chunk_size, -1)
    chunked_traces = buffer.traces.reshape(num_chunks, chunk_size, -1)
    chunked_continue_masks = buffer.continue_masks.reshape(num_chunks, chunk_size, 1)
    chunked_absorb_masks = buffer.absorb_masks.reshape(num_chunks, chunk_size, 1)
    
    valid_mask = (jnp.arange(padded_capacity) < buffer.size)[..., None]
    chunked_mask = valid_mask.reshape(num_chunks, chunk_size, 1)

    def process_chunk(carry, chunk_data):
        A_acc, b_acc = carry
        phi, next_phi, traces, rho_feats, next_rho_feats, continue_mask, absorb_mask, mask = chunk_data
        mask = mask.squeeze()
        
        # 1. Target Override: Bootstrap from self if absorbing, else next state
        target_phi = jnp.where(absorb_mask, phi, next_phi)
        target_rho_feats = jnp.where(absorb_mask, rho_feats, next_rho_feats)
        
        # 2. Compute Intrinsic Reward (rho)
        target_rho = get_scale_free_bonus(Sigma_inv, target_rho_feats)
        
        # 3. Standard LSTD Accumulation (continue_mask is already 1.0 or 0.0)
        delta_Phi = phi - gamma_i * target_phi * continue_mask
        
        A_batch = jnp.einsum("ni, nj, n -> ij", traces, delta_Phi, mask)
        b_batch = jnp.einsum("ni, n -> i", traces, target_rho * mask)
        
        return (A_acc + A_batch, b_acc + b_batch), None
    
    k_lstd = chunked_phi.shape[-1]
    init_A = jnp.zeros((k_lstd, k_lstd))
    init_b = jnp.zeros(k_lstd)
    
    # Scan over chunks to accumulate A and b
    (final_A, final_b), _ = jax.lax.scan(
        process_chunk, 
        (init_A, init_b), 
        (
            chunked_phi, chunked_next_phi, chunked_traces, 
            chunked_rho_feats, chunked_next_rho_feats,
            chunked_continue_masks, chunked_absorb_masks, chunked_mask
        )
    )
    
    reg = jnp.eye(k_lstd) * config.get("LSTD_L2_REG", 1e-3) * N
    A_view = final_A  + reg
    w_i = jnp.linalg.solve(A_view, final_b)
    
    return {"w": w_i}


def solve_lstd_lambda_from_buffer_extrinsic(buffer: LSTDBufferStateE, config):
    """Solves LSTD over the entire extended buffer using a memory-safe chunked scan."""
    N = buffer.size
    chunk_size = config['CHUNK_SIZE']
    num_chunks = config['NUM_CHUNKS']
    padded_capacity = config['PADDED_CAPACITY']
    gamma = config["GAMMA"]

    # 1. Reshape all required arrays
    chunked_phi = buffer.features.reshape(num_chunks, chunk_size, -1)
    chunked_next_phi = buffer.next_features.reshape(num_chunks, chunk_size, -1)
    chunked_reward = buffer.reward.reshape(num_chunks, chunk_size, -1)
    chunked_traces = buffer.traces.reshape(num_chunks, chunk_size, -1)
    chunked_continue_masks = buffer.continue_masks.reshape(num_chunks, chunk_size, 1)
    
    valid_mask = (jnp.arange(padded_capacity) < buffer.size)[..., None]
    chunked_mask = valid_mask.reshape(num_chunks, chunk_size, 1)

    def process_chunk(carry, chunk_data):
        A_acc, b_acc = carry
        phi, next_phi, traces, reward, continue_mask, mask = chunk_data
        mask = mask.squeeze()
        reward = reward.squeeze(-1)

        # Standard LSTD Accumulation (continue_mask is already 1.0 or 0.0)
        delta_Phi = phi - gamma * next_phi * continue_mask
        
        A_batch = jnp.einsum("ni, nj, n -> ij", traces, delta_Phi, mask)
        b_batch = jnp.einsum("ni, n -> i", traces, reward * mask)
        
        return (A_acc + A_batch, b_acc + b_batch), None
    
    k_lstd = chunked_phi.shape[-1]
    init_A = jnp.zeros((k_lstd, k_lstd))
    init_b = jnp.zeros(k_lstd)
    
    # Scan over chunks to accumulate A and b
    (final_A, final_b), _ = jax.lax.scan(
        process_chunk, 
        (init_A, init_b), 
        (
            chunked_phi, chunked_next_phi, chunked_traces, chunked_reward,
            chunked_continue_masks, chunked_mask
        )
    )
    
    reg = jnp.eye(k_lstd) * config.get("LSTD_L2_REG", 1e-3) * N
    A_view = final_A  + reg
    w = jnp.linalg.solve(A_view, final_b)
    
    return {"w": w}

