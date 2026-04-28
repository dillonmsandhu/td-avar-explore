import jax.numpy as jnp
from transformers import FlaxDinov2Model

# Load your local "Registers" version
DINO_PATH = "/usr/xtmp/ds541/hf_models/dino_v2_flax_reg"
model = FlaxDinov2Model.from_pretrained(DINO_PATH)

# Create a dummy image (B=1, C=3, H=224, W=224)
dummy_input = jnp.ones((1, 3, 224, 224))

# Forward pass
outputs = model(pixel_values=dummy_input)

# Check the shape
# last_hidden_state should be (1, 257 + 4, 384) 
# (1 CLS + 256 Patches + 4 Registers)
print(f"Output shape: {outputs.last_hidden_state.shape}")

# Extract CLS token
cls_token = outputs.last_hidden_state[:, 0, :]
print(f"CLS token shape: {cls_token.shape}") # Should be (1, 384)

# Check for NaNs (ensure weights loaded correctly)
print(f"Contains NaNs: {jnp.isnan(cls_token).any()}")