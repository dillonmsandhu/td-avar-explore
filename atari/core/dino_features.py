"""Online DINOv2 feature extractor for the Atari LSTD value branch.

Wraps a frozen torch DINOv2 ViT backbone (default: dinov2_vits14, embed_dim=384)
and exposes a single `__call__(obs) -> jnp.ndarray` bridge that uses DLPack for
zero-copy GPU transfer with JAX. Designed to be called once per rollout from
within a Python loop that sandwiches it between two jit'd halves.

Inputs are Atari-style frame stacks of shape (N, 4, 84, 84) uint8 on GPU; the
most recent frame is broadcast to 3 channels, bilinearly upsampled to 224x224,
and ImageNet-normalized before the network. Output is the CLS token, shape
(N, embed_dim) fp32 on GPU.
"""
import time
from typing import Any
import jax
import jax.numpy as jnp
import jax.image
from transformers import FlaxDinov2Model

DINO_PATH = "/usr/xtmp/ds541/hf_models/dino_v2_flax_reg"
dino_model = FlaxDinov2Model.from_pretrained(DINO_PATH)

def get_dino_features_on_atari_obs_grid(obs):
    # 1. Parse the shape: obs is usually (B, 4, 84, 84) from EnvPool
    B, C, H, W = obs.shape

    # 2. Extract the 4 individual grayscale frames
    f1 = obs[:, 0, :, :] # Oldest frame (B, 84, 84)
    f2 = obs[:, 1, :, :]
    f3 = obs[:, 2, :, :]
    f4 = obs[:, 3, :, :] # Newest frame (B, 84, 84)

    # 3. Stitch into a 2x2 grid -> (B, 168, 168)
    # Put frame 1 and 2 side-by-side for the top row
    top_row = jnp.concatenate([f1, f2], axis=-1)       # Shape: (B, 84, 168)
    # Put frame 3 and 4 side-by-side for the bottom row
    bottom_row = jnp.concatenate([f3, f4], axis=-1)    # Shape: (B, 84, 168)
    # Stack the rows vertically
    grid = jnp.concatenate([top_row, bottom_row], axis=1) # Shape: (B, 168, 168)

    # 4. Add channel dimension and repeat 3 times to fake RGB
    grid = grid[:, None, :, :]          # Shape: (B, 1, 168, 168)
    grid = jnp.repeat(grid, 3, axis=1)  # Shape: (B, 3, 168, 168)

    # 5. Cast to float and scale to [0, 1]
    grid = grid.astype(jnp.float32) / 255.0

    # 6. Resize to DINO's expected 224x224
    grid = jax.image.resize(grid, shape=(B, 3, 224, 224), method='bilinear')

    # 7. Apply standard ImageNet normalization
    mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32).reshape(1, 3, 1, 1)
    std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32).reshape(1, 3, 1, 1)
    grid = (grid - mean) / std

    # 8. Forward pass through DINO
    outputs = dino_model(pixel_values=grid, params=dino_model.params)

    # 9. Extract CLS token and project
    # Note: We only have 1 image per batch item now, so we just take the CLS token
    cls_tokens = outputs.last_hidden_state[:, 0, :] # Shape: (B, 384)
    return cls_tokens