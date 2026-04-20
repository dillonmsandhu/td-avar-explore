# lstd.py
# contains code to solve LSTD with a replay buffer
import jax
import jax.numpy as jnp
from core.buffer import LSTDBufferState, FeatureBufferState
from core.helpers import get_scale_free_bonus, expected_next_sa_features

def solve_lstd_0_from_buffer(buffer_state: FeatureBufferState, Sigma_inv, config):
    """Solves LSTD over the entire extended buffer using a memory-safe chunked scan."""
    N = buffer_state.size
    # Reshape buffer into chunks
    chunk_size = config['CHUNK_SIZE']
    num_chunks = config['NUM_CHUNKS']
    padded_capacity = config['PADDED_CAPACITY']

    chunked_phi = buffer_state.features.reshape(num_chunks, chunk_size, -1)
    chunked_next_phi = buffer_state.next_features.reshape(num_chunks, chunk_size, -1)
    chunked_terminals = buffer_state.terminals.reshape(num_chunks, chunk_size, 1)
    chunked_absorb = buffer_state.absorb_masks.reshape(num_chunks, chunk_size, 1)
    
    valid_mask = (jnp.arange(padded_capacity) < buffer_state.size)[..., None]
    chunked_mask = valid_mask.reshape(num_chunks, chunk_size, 1)
    
    gamma_i = config["GAMMA_i"]

    def process_chunk(carry, chunk_data):
        A_acc, b_acc = carry
        phi, next_phi, term, absorb, mask = chunk_data
        mask = mask
        
        next_rho = get_scale_free_bonus(Sigma_inv, next_phi)
        # if next_phi is a terminal state, then zero it out, UNLESS absorbing=True.
        cut = term * (1.0 - absorb)
        cut_factor = 1.0 - cut
        delta_Phi = phi - gamma_i * next_phi * cut_factor
    
        
        A_batch = jnp.einsum("ni, nj, n -> ij", phi, delta_Phi, mask)
        b_batch = jnp.einsum("ni, n -> i", phi, next_rho * mask)
        
        # if absorbing=True, then add a ghost self-loop
        phi_absorb = next_phi * absorb
        A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", phi_absorb, phi_absorb)
        b_abs = jnp.einsum("ni, n, n -> i", phi_absorb, next_rho)
        return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs), None
    
    k_lstd = chunked_phi.shape[-1]
    init_A = jnp.zeros((k_lstd, k_lstd))
    init_b = jnp.zeros(k_lstd)
    
    (final_A, final_b), _ = jax.lax.scan(
        process_chunk, 
        (init_A, init_b), 
        (chunked_phi, chunked_next_phi, chunked_terminals, chunked_absorb, chunked_mask)
    )
    
    # 5. Regularize and Solve ONLY ONCE at the end
    reg = jnp.eye(k_lstd) * config.get("LSTD_L2_REG", 1e-3) * N
    A_view = final_A  + reg
    w_i = jnp.linalg.solve(A_view, final_b)
    
    return {"w": w_i,}

def solve_lstd_lambda_from_buffer(buffer_state: LSTDBufferState, Sigma_inv, lstd_state, config):
    """Solves LSTD over the entire extended buffer using a memory-safe chunked scan."""
    N = buffer_state.size
    # Reshape buffer into chunks
    chunk_size = config['CHUNK_SIZE']
    num_chunks = config['NUM_CHUNKS']
    padded_capacity = config['PADDED_CAPACITY']

    chunked_phi = buffer_state.features.reshape(num_chunks, chunk_size, -1)
    chunked_next_phi = buffer_state.next_features.reshape(num_chunks, chunk_size, -1)
    chunked_traces = buffer_state.traces.reshape(num_chunks, chunk_size, -1)
    chunked_terminals = buffer_state.terminals.reshape(num_chunks, chunk_size, 1)
    chunked_absorb = buffer_state.absorb_masks.reshape(num_chunks, chunk_size, 1)
    
    valid_mask = (jnp.arange(padded_capacity) < buffer_state.size)[..., None]
    chunked_mask = valid_mask.reshape(num_chunks, chunk_size, 1)
    
    gamma_i = config["GAMMA_i"]

    def process_chunk(carry, chunk_data):
        A_acc, b_acc = carry
        phi, next_phi, traces, term, absorb, mask = chunk_data
        mask = mask.squeeze()
        
        next_rho = get_scale_free_bonus(Sigma_inv, next_phi)
        
        cut = term * (1.0 - absorb)
        cut_factor = 1.0 - cut
        delta_Phi = phi - gamma_i * next_phi * cut_factor

        A_batch = jnp.einsum("ni, nj, n -> ij", traces, delta_Phi, mask)
        b_batch = jnp.einsum("ni, n -> i", traces, next_rho * mask)
        
        phi_absorb = next_phi * absorb
        A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", phi_absorb, phi_absorb)
        b_abs = jnp.einsum("ni, n -> i", phi_absorb, next_rho * absorb.squeeze() * mask)
        
        return (A_acc + A_batch + A_abs, b_acc + b_batch + b_abs), None
    
    k_lstd = chunked_phi.shape[-1]
    init_A = jnp.zeros((k_lstd, k_lstd))
    init_b = jnp.zeros(k_lstd)
    
    (final_A, final_b), _ = jax.lax.scan(
        process_chunk, 
        (init_A, init_b), 
        (chunked_phi, chunked_next_phi, chunked_traces, chunked_terminals, chunked_absorb, chunked_mask)
    )
    
    # 5. Regularize and Solve ONLY ONCE at the end
    reg = jnp.eye(k_lstd) * config.get("LSTD_L2_REG", 1e-3) * N
    A_view = final_A  + reg
    w_i = jnp.linalg.solve(A_view, final_b)
    
    return {"w": w_i,}

def solve_lspi_buffer(buffer_state: FeatureBufferState, Sigma_inv, lstd_state, config):
    """Solves LSPI over the entire extended buffer using a memory-safe chunked scan."""
    N = buffer_state.size
    # Reshape buffer into chunks
    chunk_size = config['CHUNK_SIZE']
    num_chunks = config['NUM_CHUNKS']
    padded_capacity = config['PADDED_CAPACITY']
    n_actions = config['N_ACTIONS']

    chunked_phi_sa = buffer_state.features.reshape(num_chunks, chunk_size, -1)
    chunked_next_phi_s = buffer_state.next_features.reshape(num_chunks, chunk_size, -1)
    chunked_terminals = buffer_state.terminals.reshape(num_chunks, chunk_size, 1)
    chunked_absorb = buffer_state.absorb_masks.reshape(num_chunks, chunk_size, 1)

    dim_kA = chunked_phi_sa.shape[-1]
    k_lstd = chunked_next_phi_s.shape[-1]
    
    valid_mask = (jnp.arange(padded_capacity) < buffer_state.size)[..., None]
    chunked_mask = valid_mask.reshape(num_chunks, chunk_size, 1)
    
    gamma_i = config["GAMMA_i"]

    def lspi_step(w_current, _):
        def process_chunk(carry, chunk_data):
            A_acc, b_acc = carry
            c_phi_sa, c_next_phi_s, c_term, c_absorb, c_mask = chunk_data
            
            next_rho = get_scale_free_bonus(Sigma_inv, c_next_phi_s)
            
            # 1. Greedy Policy Evaluation
            w_reshaped = w_current.reshape(n_actions, k_lstd)
            Q_next = jnp.einsum("...k, ak -> ...a", c_next_phi_s, w_reshaped)
            greedy_actions = jnp.argmax(Q_next, axis=-1)
            Pi_greedy = jax.nn.one_hot(greedy_actions, n_actions)
            
            PΠφ = expected_next_sa_features(c_next_phi_s, Pi_greedy, dim_kA )
            
            # 2. Construction of A
            c_traces_masked = c_phi_sa * c_mask
            
            S = jnp.einsum("ni, nj -> ij", c_traces_masked, c_phi_sa)
            
            cut = term * (1.0 - absorb)
            cut_factor = 1.0 - cut

            γPΠφ = gamma_i * cut_factor * PΠφ
            Z_γPΠΦ = jnp.einsum("ni, nj -> ij", c_traces_masked, γPΠφ)
            A_std = S - Z_γPΠΦ
            
            phi_absorb = PΠφ * c_absorb
            phi_absorb = c_phi_sa * c_absorb
            A_abs = (1 - gamma_i) * jnp.einsum("ni, nj -> ij", phi_absorb * c_mask, phi_absorb)
            
            A_batch = A_std + A_abs
            
            # 3. Construction of b
            b_std = jnp.einsum("ni, n -> i", c_traces_masked, next_rho)
            b_abs = jnp.einsum("ni, n -> i", phi_absorb * c_mask, next_rho)
            b_batch = b_std + b_abs
            
            return (A_acc + A_batch, b_acc + b_batch), None

        init_A = jnp.zeros((dim_kA, dim_kA))
        init_b = jnp.zeros(dim_kA)
        
        (final_A, final_b), _ = jax.lax.scan(
            process_chunk, 
            (init_A, init_b), 
            (chunked_phi_sa, chunked_next_phi_s, chunked_terminals, chunked_absorb, chunked_mask)
        )
        
        
        reg = jnp.eye(dim_kA) * config.get("LSTD_L2_REG", 1e-3) * N
        A_view = final_A + reg
        w_new = jnp.linalg.solve(A_view, final_b)
        
        return w_new, None

    w_init = lstd_state["w"]
    w_final, _ = jax.lax.scan(lspi_step, w_init, None, length=config.get("LSPI_NUM_ITERS", 3))
    
    return {
        "w": w_final,
    }
    