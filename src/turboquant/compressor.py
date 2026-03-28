"""
Stage 1: PolarQuant MSE compression — pure MLX.
Rotate → quantize → return indices + residual for QJL.

Codebook is cached: built once per (bits, head_dim), reused across all layers.
Only rotation matrix differs per layer (different seed).
"""

import mlx.core as mx
import numpy as np
from .codebook import build_codebook, build_rotation

# Global caches
_codebook_cache = {}  # (bits, head_dim) → mx.array
_rotation_cache = {}  # (head_dim, seed) → (rotation, rotation_t)


def _get_codebook(head_dim: int, bits: int) -> mx.array:
    """Get or build codebook. Cached — same codebook for all layers."""
    key = (bits, head_dim)
    if key not in _codebook_cache:
        centroids_np = build_codebook(bits, head_dim)
        _codebook_cache[key] = mx.array(centroids_np)
        mx.eval(_codebook_cache[key])
    return _codebook_cache[key]


def _get_rotation(head_dim: int, seed: int):
    """Get or build rotation. Cached per (head_dim, seed)."""
    key = (head_dim, seed)
    if key not in _rotation_cache:
        r = mx.array(build_rotation(head_dim, seed))
        rt = mx.transpose(r)
        mx.eval(r, rt)
        _rotation_cache[key] = (r, rt)
    return _rotation_cache[key]


class PolarQuantMLX:
    """
    Stage 1 of TurboQuant: MSE-optimal quantization.
    Rotates vectors, quantizes per-coordinate with Lloyd-Max codebook.

    Codebook is shared across all instances with same (bits, head_dim).
    Only rotation matrix is unique per layer (via seed).
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42):
        self.head_dim = head_dim
        self.bits = bits

        # Shared codebook (cached)
        self.centroids = _get_codebook(head_dim, bits)

        # Per-layer rotation (cached per seed)
        self.rotation, self.rotation_t = _get_rotation(head_dim, seed)

    def quantize(self, x: mx.array):
        """Quantize to indices. Input: (..., head_dim). Returns uint8."""
        x = x.astype(mx.float32)
        y = x @ self.rotation_t
        dists = mx.abs(mx.expand_dims(y, axis=-1) - self.centroids)
        return mx.argmin(dists, axis=-1).astype(mx.uint8)

    def dequantize(self, indices: mx.array) -> mx.array:
        """Dequantize indices back to vectors. Returns float32."""
        y_hat = self.centroids[indices.astype(mx.uint32)]
        return y_hat @ self.rotation

    def quantize_with_residual(self, x: mx.array):
        """Quantize and return (indices, residual)."""
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        residual = x.astype(mx.float32) - x_hat
        return indices, residual
