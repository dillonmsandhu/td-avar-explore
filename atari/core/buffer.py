# buffer.py
from typing import NamedTuple, Generic, TypeVar, Any
import jax
import jax.numpy as jnp
import core.helpers as helpers
# --- Base Buffer Class ---
# Generic Type Definition
BufferStateT = TypeVar("BufferStateT", bound=tuple)

class BaseBufferManager(Generic[BufferStateT]):
    """Generic stateless buffer manager handling JAX PyTree updates."""
    
    def __init__(self, config, k_lstd, buffer_capacity, extended_capacity, chunk_size):
        self.config = config
        self.k_lstd = k_lstd
        self.buffer_capacity = buffer_capacity
        
        self.num_chunks = (extended_capacity + chunk_size - 1) // chunk_size        
        self.padded_capacity = self.num_chunks * chunk_size
        
        self.batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        self.static_fifo_drops = int(self.batch_size * config.get("PERCENT_FIFO", 0.2))
        self.static_prb_drops = self.batch_size - self.static_fifo_drops
        self.num_cuts = config.get("EVICTION_CUTS", 4)
        self.static_drops_per_cut = self.static_prb_drops // self.num_cuts

    def init_state(self) -> BufferStateT:
        raise NotImplementedError

    def evict_buffer(self, buffer_state: BufferStateT, rng) -> BufferStateT:
        raise NotImplementedError

    def update_buffer(self, buffer_state: BufferStateT, new_batch: BufferStateT) -> BufferStateT:
            """Generically appends a new batch of data using jax.tree.map."""
            start_idx = buffer_state.size
            
            def _update_single_array(buffer_arr, new_arr):
                if buffer_arr.ndim == 0: 
                    return buffer_arr
                # Now JAX knows exactly how large this reshape will be at compile time
                new_arr = new_arr.reshape(self.batch_size, -1).astype(jnp.float32)
                return jax.lax.dynamic_update_slice(buffer_arr, new_arr, (start_idx, 0))

            updated_state = jax.tree.map(_update_single_array, buffer_state, new_batch)
            # Adding a static int to a dynamic tensor (start_idx) is perfectly safe
            return updated_state._replace(size=start_idx + self.batch_size)


# --------------------------------------------------------------------------------------------
# LSTD(Lambda) Buffer
# --------------------------------------------------------------------------------------------

class LSTDBufferState(NamedTuple):
    traces: jnp.ndarray
    features: jnp.ndarray
    next_features: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray 
    size: jnp.ndarray  

class FeatureTraceBufferManager(BaseBufferManager[LSTDBufferState]):
    
    def init_state(self) -> LSTDBufferState:
        return LSTDBufferState(
            traces=jnp.zeros((self.padded_capacity, self.k_lstd), dtype=jnp.float32),
            features=jnp.zeros((self.padded_capacity, self.k_lstd), dtype=jnp.float32),
            next_features=jnp.zeros((self.padded_capacity, self.k_lstd), dtype=jnp.float32),
            terminals=jnp.zeros((self.padded_capacity, 1), dtype=jnp.float32),
            absorb_masks=jnp.zeros((self.padded_capacity, 1), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32)
        )

    def evict_buffer(self, buffer_state: LSTDBufferState, rng) -> LSTDBufferState:
        """Computes scores and generically evicts items using jax.tree.map."""
        size = buffer_state.size
        
        phi = buffer_state.features
        next_phi = buffer_state.next_features
        traces = buffer_state.traces
        terminals = buffer_state.terminals
        
        buffer_is_full = size > self.buffer_capacity
        indices = jnp.arange(self.padded_capacity)
        valid_mask = indices < size
        
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < self.static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))

        Z_all = traces
        X_all = phi - self.config["GAMMA_i"] * (1 - terminals) * next_phi

        def cut_step(carry, step_idx):
            mask_curr = carry
            valid_Z = Z_all * mask_curr[:, None]
            valid_X = X_all * mask_curr[:, None]
            
            A_curr = jnp.einsum("ni, nj -> ij", valid_Z, valid_X) 
            reg = jnp.eye(self.k_lstd) * self.config.get("LSTD_L2_REG", 1e-3) * size
            A_inv_curr = jnp.linalg.solve(A_curr + reg, jnp.eye(self.k_lstd)) 
            
            U = Z_all @ A_inv_curr.T
            V = X_all @ A_inv_curr
            W = V @ A_inv_curr.T
            
            c = 1.0 - jnp.sum(X_all * U, axis=-1)
            c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
            
            scores = (2.0 * jnp.sum(U * W, axis=-1) / c) + (jnp.sum(U * U, axis=-1) * jnp.sum(V * V, axis=-1)) / (c * c)
            
            drop_logits = -scores / self.config.get("STOCHASTIC_TEMP", 1.0)
            drop_logits = jnp.where(mask_curr, drop_logits, -jnp.inf)
            
            rng_key = jax.random.fold_in(rng, step_idx) 
            noisy_logits = drop_logits + jax.random.gumbel(rng_key, drop_logits.shape)
            
            _, drop_indices = jax.lax.top_k(noisy_logits, self.static_drops_per_cut)
            mask_next_candidate = mask_curr.at[drop_indices].set(False)
            mask_next = jnp.where(buffer_is_full, mask_next_candidate, mask_curr)
            
            return mask_next, None

        final_mask, _ = jax.lax.scan(cut_step, initial_mask, jnp.arange(self.num_cuts))
        
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, self.buffer_capacity)

        def _compact_array(arr):
            if arr.ndim == 0: 
                return arr
            return jnp.zeros_like(arr).at[:self.buffer_capacity].set(arr[keep_indices])

        compacted_state = jax.tree.map(_compact_array, buffer_state)
        new_size = jnp.minimum(size, self.buffer_capacity)

        return compacted_state._replace(size=new_size)

# --------------------------------------------------------------------------------------------
# LSTD(0) Buffer
# --------------------------------------------------------------------------------------------

class FeatureBufferState(NamedTuple):
    features: jnp.ndarray
    next_features: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    size: jnp.ndarray

class FeatureBufferManager(BaseBufferManager[FeatureBufferState]):
    """LSTD(0) Buffer Manager that does not compute or store eligibility traces."""

    def init_state(self) -> FeatureBufferState:
        return FeatureBufferState(
            features=jnp.zeros((self.padded_capacity, self.k_lstd), dtype=jnp.float32),
            next_features=jnp.zeros((self.padded_capacity, self.k_lstd), dtype=jnp.float32),
            terminals=jnp.zeros((self.padded_capacity, 1), dtype=jnp.float32),
            absorb_masks=jnp.zeros((self.padded_capacity, 1), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32)
        )

    def evict_buffer(self, buffer_state: FeatureBufferState, rng) -> FeatureBufferState:
        size = buffer_state.size
        phi = buffer_state.features
        next_phi = buffer_state.next_features
        terminals = buffer_state.terminals
        
        buffer_is_full = size > self.buffer_capacity
        indices = jnp.arange(self.padded_capacity)
        valid_mask = indices < size
        
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < self.static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))

        Z_all = phi
        X_all = phi - self.config["GAMMA_i"] * (1 - terminals) * next_phi

        def cut_step(carry, step_idx):
            mask_curr = carry
            valid_Z = Z_all * mask_curr[:, None]
            valid_X = X_all * mask_curr[:, None]
            
            A_curr = jnp.einsum("ni, nj -> ij", valid_Z, valid_X) 
            reg = jnp.eye(self.k_lstd) * self.config.get("LSTD_L2_REG", 1e-3) * size
            A_inv_curr = jnp.linalg.solve(A_curr + reg, jnp.eye(self.k_lstd)) 
            
            U = Z_all @ A_inv_curr.T
            V = X_all @ A_inv_curr
            W = V @ A_inv_curr.T
            
            c = 1.0 - jnp.sum(X_all * U, axis=-1)
            c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
            
            scores = (2.0 * jnp.sum(U * W, axis=-1) / c) + (jnp.sum(U * U, axis=-1) * jnp.sum(V * V, axis=-1)) / (c * c)
            
            drop_logits = -scores / self.config.get("STOCHASTIC_TEMP", 1.0)
            drop_logits = jnp.where(mask_curr, drop_logits, -jnp.inf)
            
            rng_key = jax.random.fold_in(rng, step_idx) 
            noisy_logits = drop_logits + jax.random.gumbel(rng_key, drop_logits.shape)
            
            _, drop_indices = jax.lax.top_k(noisy_logits, self.static_drops_per_cut)
            mask_next_candidate = mask_curr.at[drop_indices].set(False)
            mask_next = jnp.where(buffer_is_full, mask_next_candidate, mask_curr)
            
            return mask_next, None

        final_mask, _ = jax.lax.scan(cut_step, initial_mask, jnp.arange(self.num_cuts))
        
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, self.buffer_capacity)

        def _compact_array(arr):
            if arr.ndim == 0: 
                return arr
            return jnp.zeros_like(arr).at[:self.buffer_capacity].set(arr[keep_indices])

        compacted_state = jax.tree.map(_compact_array, buffer_state)
        new_size = jnp.minimum(size, self.buffer_capacity)

        return compacted_state._replace(size=new_size)

# --------------------------------------------------------------------------------------------
# LSPI(0) Buffer
# --------------------------------------------------------------------------------------------

from typing import NamedTuple
import jax
import jax.numpy as jnp

# Define the explicit LSPI State Tuple
class LSPIBufferState(NamedTuple):
    features: jnp.ndarray
    next_features: jnp.ndarray
    terminals: jnp.ndarray
    absorb_masks: jnp.ndarray
    size: jnp.ndarray


# Inherit explicitly with [LSPIBufferState]
class LSPIFeatureBufferManager(BaseBufferManager[LSPIBufferState]):
    """LSPI(0) Buffer Manager that does not compute or store eligibility traces."""

    def __init__(self, config, k_lstd, buffer_capacity, extended_capacity, chunk_size, dim_kA, n_actions):
        super().__init__(config, k_lstd, buffer_capacity, extended_capacity, chunk_size)
        
        # LSPI specific properties
        self.dim_kA = dim_kA
        self.n_actions = n_actions


    def init_state(self) -> LSPIBufferState:
        """Initializes an empty LSPI buffer with pre-allocated zero arrays."""
        return LSPIBufferState(
            features=jnp.zeros((self.padded_capacity, self.dim_kA), dtype=jnp.float32), 
            next_features=jnp.zeros((self.padded_capacity, self.k_lstd), dtype=jnp.float32), 
            terminals=jnp.zeros((self.padded_capacity, 1), dtype=jnp.float32),
            absorb_masks=jnp.zeros((self.padded_capacity, 1), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32)
        )

    # 3. Add lstd_state to the signature so we can compute the greedy policy
    def evict_buffer(self, buffer_state: LSPIBufferState, rng, lstd_state) -> LSPIBufferState:
        size = buffer_state.size
        
        # 4. Fix variable names to match your math below
        phi_sa = buffer_state.features
        next_phi_s = buffer_state.next_features
        terminals = buffer_state.terminals
        
        buffer_is_full = size > self.buffer_capacity
        indices = jnp.arange(self.padded_capacity)
        valid_mask = indices < size
        
        fifo_invalid_mask = jnp.logical_and(buffer_is_full, indices < self.static_fifo_drops)
        initial_mask = jnp.logical_and(valid_mask, jnp.logical_not(fifo_invalid_mask))

        # --- THE LSPI MATH ---
        # 5. Use self.n_actions and self.k_lstd
        w_reshaped = lstd_state["w"].reshape(self.n_actions, self.k_lstd)
        Q_next = jnp.einsum("...k, ak -> ...a", next_phi_s, w_reshaped)
        greedy_actions = jnp.argmax(Q_next, axis=-1)
        Pi_greedy = jax.nn.one_hot(greedy_actions, self.n_actions)
        
        PΠφ = helpers.expected_next_sa_features(next_phi_s, Pi_greedy, self.dim_kA)
        
        Z_all = phi_sa
        # 6. Use self.config
        X_all = phi_sa - self.config["GAMMA_i"] * (1 - terminals) * PΠφ

        def cut_step(carry, step_idx):
            mask_curr = carry
            valid_Z = Z_all * mask_curr[:, None]
            valid_X = X_all * mask_curr[:, None]
            
            A_curr = jnp.einsum("ni, nj -> ij", valid_Z, valid_X) 
            # 7. Reg uses dim_kA, not k_lstd!
            reg = jnp.eye(self.dim_kA) * self.config.get("LSTD_L2_REG", 1e-3) * size
            A_inv_curr = jnp.linalg.solve(A_curr + reg, jnp.eye(self.dim_kA)) 
            
            U = Z_all @ A_inv_curr.T
            V = X_all @ A_inv_curr
            W = V @ A_inv_curr.T
            
            c = 1.0 - jnp.sum(X_all * U, axis=-1)
            c = jnp.where(jnp.abs(c) < 1e-5, 1e-5, c)
            
            scores = (2.0 * jnp.sum(U * W, axis=-1) / c) + (jnp.sum(U * U, axis=-1) * jnp.sum(V * V, axis=-1)) / (c * c)
            
            drop_logits = -scores / self.config.get("STOCHASTIC_TEMP", 1.0)
            drop_logits = jnp.where(mask_curr, drop_logits, -jnp.inf)
            
            rng_key = jax.random.fold_in(rng, step_idx) 
            noisy_logits = drop_logits + jax.random.gumbel(rng_key, drop_logits.shape)
            
            _, drop_indices = jax.lax.top_k(noisy_logits, self.static_drops_per_cut)
            mask_next_candidate = mask_curr.at[drop_indices].set(False)
            mask_next = jnp.where(buffer_is_full, mask_next_candidate, mask_curr)
            
            return mask_next, None

        final_mask, _ = jax.lax.scan(cut_step, initial_mask, jnp.arange(self.num_cuts))
        
        selection_scores = jnp.where(final_mask, 1.0, 0.0) + (indices.astype(jnp.float32) * 1e-7)
        _, keep_indices = jax.lax.top_k(selection_scores, self.buffer_capacity)

        def _compact_array(arr):
            if arr.ndim == 0: 
                return arr
            return jnp.zeros_like(arr).at[:self.buffer_capacity].set(arr[keep_indices])

        compacted_state = jax.tree.map(_compact_array, buffer_state)
        new_size = jnp.minimum(size, self.buffer_capacity)

        return compacted_state._replace(size=new_size)
