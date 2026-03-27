"""
Stage 1: PolarQuant MSE compression — pure MLX.
Rotate → quantize → return indices + residual for QJL.
"""

import math
import mlx.core as mx
import numpy as np
from .codebook import build_codebook, build_rotation


class PolarQuantMLX:
    """
    Stage 1 of TurboQuant: MSE-optimal quantization.
    Rotates vectors, quantizes per-coordinate with Lloyd-Max codebook.
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42):
        self.head_dim = head_dim
        self.bits = bits

        # Build on CPU, move to MLX
        centroids_np = build_codebook(bits, head_dim)
        rotation_np = build_rotation(head_dim, seed)

        self.centroids = mx.array(centroids_np)
        self.rotation = mx.array(rotation_np)
        self.rotation_t = mx.transpose(self.rotation)
        mx.eval(self.centroids, self.rotation, self.rotation_t)

    def quantize(self, x: mx.array):
        """
        Quantize to indices. Input: (..., head_dim).
        Returns: uint8 indices of same shape.
        """
        x = x.astype(mx.float32)
        y = x @ self.rotation_t  # rotate
        dists = mx.abs(mx.expand_dims(y, axis=-1) - self.centroids)
        return mx.argmin(dists, axis=-1).astype(mx.uint8)

    def dequantize(self, indices: mx.array) -> mx.array:
        """
        Dequantize indices back to vectors. Returns float32.
        """
        y_hat = self.centroids[indices.astype(mx.uint32)]
        return y_hat @ self.rotation  # unrotate

    def quantize_with_residual(self, x: mx.array):
        """
        Quantize and return both indices and residual vector.
        Input x should already be normalized (unit sphere).
        Returns: (indices, residual) where residual = x - dequantize(indices).
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        residual = x.astype(mx.float32) - x_hat
        return indices, residual
