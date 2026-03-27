"""
TurboQuant KV-Cache Compressor.

Compresses FP16/FP32 KV-cache vectors to b-bit representation with
near-optimal distortion and ZERO memory overhead for normalization constants.

Algorithm:
  1. Record norm: ||x||
  2. Normalize: x_hat = x / ||x||
  3. Random rotate: y = R @ x_hat  (R is orthogonal, fixed per instance)
  4. Scalar quantize each coordinate: idx_j = LloydMax(y_j)
  5. Store: (indices: uint8, norm: fp16)
  6. Dequantize: y' = centroids[indices], x' = ||x|| * R^T @ y'

The key insight: after rotation, coordinates follow a known Beta
distribution, so optimal quantizer centroids are DATA-INDEPENDENT
and can be precomputed. No per-token or per-channel scaling needed.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
           Google Research, ICLR 2026 (arXiv:2504.19874)
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from .codebook import LloydMaxCodebook


class CompressedKVCache:
    """Container for compressed KV-cache data."""

    __slots__ = ("indices", "norms", "shape", "dtype")

    def __init__(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
    ):
        self.indices = indices
        self.norms = norms
        self.shape = shape
        self.dtype = dtype

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def to(self, device: torch.device) -> "CompressedKVCache":
        return CompressedKVCache(
            indices=self.indices.to(device),
            norms=self.norms.to(device),
            shape=self.shape,
            dtype=self.dtype,
        )


class TurboQuant:
    """
    TurboQuant KV-Cache Compressor.

    Compresses KV-cache vectors from FP16/FP32 to b-bit representation
    with near-optimal distortion and ZERO memory overhead for normalization.

    Works on CPU, CUDA, and MPS (Apple Silicon).

    Args:
        bits: Quantization bits (2, 3, or 4). Default: 4.
        head_dim: Attention head dimension. Default: 128.
        seed: Random seed for rotation matrix reproducibility.
        codebook_iterations: Lloyd-Max iterations for codebook optimization.

    Example::

        tq = TurboQuant(bits=4, head_dim=128)
        # x shape: (batch, num_heads, seq_len, head_dim)
        compressed = tq.compress(x)
        reconstructed = tq.decompress(compressed)
    """

    def __init__(
        self,
        bits: int = 4,
        head_dim: int = 128,
        seed: int = 42,
        codebook_iterations: int = 300,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        if head_dim < 2:
            raise ValueError(f"head_dim must be >= 2, got {head_dim}")

        self.bits = bits
        self.head_dim = head_dim
        self.seed = seed

        # Precompute fixed random orthogonal rotation matrix
        rng = np.random.RandomState(seed)
        q, _ = np.linalg.qr(rng.randn(head_dim, head_dim))
        self.rotation = torch.tensor(q, dtype=torch.float32)
        self.rotation_t = self.rotation.T.contiguous()

        # Precompute Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(
            bits=bits, dim=head_dim, iterations=codebook_iterations
        )

        # Stats tracking
        self._compress_time = 0.0
        self._decompress_time = 0.0
        self._original_bytes = 0
        self._compressed_bytes = 0

    def compress(self, x: torch.Tensor) -> CompressedKVCache:
        """
        Compress KV-cache tensor to b-bit indices + FP16 norms.

        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim) in FP16/FP32.

        Returns:
            CompressedKVCache with uint8 indices, fp16 norms, and metadata.

        Raises:
            ValueError: If last dimension doesn't match head_dim.
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Last dim must be {self.head_dim}, got {x.shape[-1]}"
            )

        t0 = time.time()
        shape, dtype, device = x.shape, x.dtype, x.device
        x = x.float()

        # Step 1: Compute and store norms
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Step 2: Normalize to unit sphere
        x_hat = x / norms

        # Step 3: Random orthogonal rotation
        y = torch.matmul(x_hat, self.rotation.to(device).T)

        # Step 4: Scalar quantize each coordinate
        indices = self.codebook.quantize(y)

        # Track stats
        num_tokens = int(np.prod(shape[:-1]))
        self._original_bytes += x.numel() * 2  # FP16
        self._compressed_bytes += indices.numel() * self.bits / 8 + num_tokens * 2
        self._compress_time += time.time() - t0

        return CompressedKVCache(
            indices=indices,
            norms=norms.to(torch.float16).squeeze(-1),
            shape=shape,
            dtype=dtype,
        )

    def decompress(self, compressed: CompressedKVCache) -> torch.Tensor:
        """
        Decompress back to original shape and dtype.

        Args:
            compressed: CompressedKVCache from compress().

        Returns:
            Reconstructed tensor of original shape and dtype.
        """
        t0 = time.time()
        device = compressed.device

        # Step 1: Dequantize indices to centroid values
        y_hat = self.codebook.dequantize(compressed.indices)

        # Step 2: Inverse rotation
        x_hat = torch.matmul(y_hat, self.rotation_t.to(device).T)

        # Step 3: Rescale by norms
        x = x_hat * compressed.norms.float().unsqueeze(-1)

        self._decompress_time += time.time() - t0
        return x.to(compressed.dtype)

    def memory_bytes(self, compressed: CompressedKVCache) -> Dict[str, float]:
        """
        Calculate memory usage for a compressed KV cache.

        Returns:
            Dict with 'original', 'compressed', 'ratio', and 'savings_pct'.
        """
        n_elements = compressed.indices.numel()
        n_tokens = compressed.norms.numel()
        orig = n_elements * 2  # FP16 original
        comp = n_elements * self.bits / 8 + n_tokens * 2  # packed bits + FP16 norms
        ratio = orig / comp if comp > 0 else float("inf")
        return {
            "original": orig,
            "compressed": int(comp),
            "ratio": ratio,
            "savings_pct": (1 - 1 / ratio) * 100 if ratio > 0 else 0,
        }

    def compression_ratio(self) -> float:
        """Get cumulative compression ratio across all compress() calls."""
        if self._compressed_bytes == 0:
            return 0.0
        return self._original_bytes / self._compressed_bytes

    def stats(self) -> Dict[str, object]:
        """Get cumulative statistics."""
        return {
            "bits": self.bits,
            "head_dim": self.head_dim,
            "compression_ratio": f"{self.compression_ratio():.1f}x",
            "original_bytes": self._original_bytes,
            "compressed_bytes": int(self._compressed_bytes),
            "compress_time_ms": f"{self._compress_time * 1000:.1f}",
            "decompress_time_ms": f"{self._decompress_time * 1000:.1f}",
        }

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self._compress_time = 0.0
        self._decompress_time = 0.0
        self._original_bytes = 0
        self._compressed_bytes = 0

    def to(self, device: torch.device) -> "TurboQuant":
        """Move rotation matrices and codebook to a device."""
        self.rotation = self.rotation.to(device)
        self.rotation_t = self.rotation_t.to(device)
        self.codebook.to(device)
        return self


class StandardQ4Quantizer:
    """
    Standard per-group INT4 quantization baseline (what llama.cpp/GGUF uses).

    Useful for fair comparison against TurboQuant at the same bit budget.

    Args:
        group_size: Number of elements per quantization group. Default: 32.
    """

    def __init__(self, group_size: int = 32):
        self.group_size = group_size
        self.bits = 4
        self.qmin = 0
        self.qmax = 2**4 - 1  # 15

    def compress(self, x: torch.Tensor) -> Dict:
        """Quantize to INT4 with per-group scale and zero-point."""
        shape = x.shape
        dtype = x.dtype
        x = x.float()

        *batch_dims, dim = x.shape
        if dim % self.group_size != 0:
            raise ValueError(
                f"dim {dim} not divisible by group_size {self.group_size}"
            )

        x_grouped = x.reshape(*batch_dims, dim // self.group_size, self.group_size)

        g_min = x_grouped.amin(dim=-1, keepdim=True)
        g_max = x_grouped.amax(dim=-1, keepdim=True)

        scale = ((g_max - g_min) / self.qmax).clamp(min=1e-8)
        zero_point = (-g_min / scale).round().clamp(self.qmin, self.qmax)

        indices = ((x_grouped - g_min) / scale).round().clamp(self.qmin, self.qmax).to(torch.uint8)

        return {
            "indices": indices,
            "scales": scale.to(torch.float16),
            "zeros": zero_point.to(torch.float16),
            "g_min": g_min.to(torch.float16),
            "shape": shape,
            "dtype": dtype,
        }

    def decompress(self, comp: Dict) -> torch.Tensor:
        """Dequantize: x = indices * scale + min."""
        x_grouped = comp["indices"].float() * comp["scales"].float() + comp["g_min"].float()
        return x_grouped.reshape(comp["shape"]).to(comp["dtype"])

    def memory_bytes(self, comp: Dict) -> Dict[str, float]:
        """Calculate actual memory including per-group overhead."""
        n_elements = comp["indices"].numel()
        n_groups = comp["scales"].numel()

        original = n_elements * 2  # FP16
        packed_indices = n_elements * self.bits / 8
        overhead = n_groups * 4  # scale(fp16) + zero(fp16) per group
        compressed = packed_indices + overhead
        actual_bits = compressed * 8 / n_elements if n_elements > 0 else 0

        return {
            "original": int(original),
            "compressed": int(compressed),
            "ratio": original / compressed if compressed > 0 else float("inf"),
            "actual_bits_per_element": round(actual_bits, 2),
            "overhead_bytes": int(overhead),
            "overhead_pct": round(overhead / compressed * 100, 1) if compressed > 0 else 0,
        }
