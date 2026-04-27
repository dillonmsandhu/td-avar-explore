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
from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp


class DinoFeatureExtractor:
    def __init__(self, config: dict[str, Any]):
        import torch
        import torch.nn.functional as F  # noqa: F401  (used in __call__)

        self._torch = torch
        self._F = F

        if not torch.cuda.is_available():
            raise RuntimeError("DinoFeatureExtractor requires a CUDA device.")

        model_name = config.get("DINO_MODEL", "dinov2_vits14")
        dtype_name = config.get("DINO_DTYPE", "fp16")
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unknown DINO_DTYPE={dtype_name!r}")
        self.dtype = dtype_map[dtype_name]
        self.device = torch.device("cuda")
        self.sub_batch = int(config.get("DINO_BATCH_SIZE", 4096))

        frame_strategy = config.get("DINO_FRAME_STRATEGY", "last")
        if frame_strategy != "last":
            raise NotImplementedError(
                f"DINO_FRAME_STRATEGY={frame_strategy!r} not implemented; only 'last' supported."
            )
        self.frame_strategy = frame_strategy

        repo = "facebookresearch/dinov2"
        net = torch.hub.load(repo, model_name)
        net = net.to(self.device, dtype=self.dtype).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.k = int(net.embed_dim)

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=self.dtype
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=self.dtype
        ).view(1, 3, 1, 1)

        self._last_call_seconds = 0.0

    def __call__(self, obs):
        torch = self._torch
        F = self._F

        if not isinstance(obs, jax.Array):
            obs = jnp.asarray(obs)

        with torch.no_grad():
            t0 = time.perf_counter()
            x = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(obs))
            # (N, 4, 84, 84) uint8 -> last frame (N, 1, 84, 84) in self.dtype
            x = x[:, -1:, :, :].to(self.dtype) / 255.0
            x = x.expand(-1, 3, -1, -1).contiguous()
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std

            feats = []
            for i in range(0, x.shape[0], self.sub_batch):
                feats.append(self.net(x[i : i + self.sub_batch]).float())
            out = torch.cat(feats, dim=0).contiguous()

            jax_out = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(out))
            self._last_call_seconds = time.perf_counter() - t0
        return jax_out
