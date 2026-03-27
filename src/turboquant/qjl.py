"""
Stage 2: QJL (Quantized Johnson-Lindenstrauss) residual correction — pure MLX.

Projects quantization residual through random Gaussian matrix, stores only sign bits.
Used DURING attention computation (not during decompression) to produce unbiased inner products.
"""

import math
import mlx.core as mx
import numpy as np
from .codebook import build_qjl_matrix


class QJLMLX:
    """
    QJL residual projector.

    Given a residual vector r = x - x_mse:
      1. Project: p = r @ S.T
      2. Store: sign(p) as packed bits + ||r|| as float16

    During attention, corrects inner product:
      <q, x> ≈ <q, x_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs>
    """

    def __init__(self, head_dim: int, m: int = None, seed: int = 43):
        self.head_dim = head_dim
        self.m = m or head_dim
        self.correction_scale = math.sqrt(math.pi / 2) / self.m

        S_np = build_qjl_matrix(head_dim, m=self.m, seed=seed)
        self.S = mx.array(S_np)
        self.S_t = mx.transpose(self.S)
        mx.eval(self.S, self.S_t)

    def compute_signs(self, residual: mx.array):
        """
        Project residual and extract sign bits.

        Args:
            residual: (..., head_dim) float32

        Returns:
            signs: (..., m) float32 with values ±1
            residual_norm: (...,) float16
        """
        projected = residual.astype(mx.float32) @ self.S_t  # (..., m)
        signs = mx.sign(projected)
        # Map zeros to +1
        signs = mx.where(signs == 0, mx.array(1.0), signs)
        residual_norm = mx.sqrt(mx.sum(residual * residual, axis=-1))
        return signs, residual_norm.astype(mx.float16)

    def pack_signs(self, signs: mx.array) -> mx.array:
        """
        Pack ±1 signs into uint8 (8 signs per byte).
        signs: (..., m) with values ±1
        Returns: (..., m//8) uint8
        """
        # Convert ±1 to 0/1
        bits = ((signs + 1) / 2).astype(mx.uint8)  # 0 or 1
        *batch_dims, m = bits.shape
        padded_m = ((m + 7) // 8) * 8
        if padded_m > m:
            pad_shape = list(batch_dims) + [padded_m - m]
            bits = mx.concatenate([bits, mx.zeros(pad_shape, dtype=mx.uint8)], axis=-1)
        bits = bits.reshape(*batch_dims, -1, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        packed = mx.sum(bits * powers, axis=-1).astype(mx.uint8)
        return packed

    def unpack_signs(self, packed: mx.array, m: int) -> mx.array:
        """
        Unpack uint8 to ±1 signs.
        packed: (..., m//8) uint8
        Returns: (..., m) float32 with values ±1
        """
        *batch_dims, n_bytes = packed.shape
        packed_expanded = mx.expand_dims(packed, axis=-1)  # (..., n_bytes, 1)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        bits = ((packed_expanded.astype(mx.uint32) // powers.astype(mx.uint32)) % 2).astype(mx.float32)
        bits = bits.reshape(*batch_dims, -1)  # (..., n_bytes*8)
        bits = bits[..., :m]  # trim padding
        return bits * 2 - 1  # 0/1 → -1/+1

    def correct_inner_product(self, query: mx.array, signs: mx.array,
                               residual_norm: mx.array) -> mx.array:
        """
        Compute QJL correction term for attention scores.

        Args:
            query: (batch, n_heads, seq_q, head_dim) — the query vectors
            signs: (batch, n_kv_heads, seq_kv, m) — packed then unpacked sign bits
            residual_norm: (batch, n_kv_heads, seq_kv) — ||residual|| per token

        Returns:
            correction: (batch, n_heads, seq_q, seq_kv) — additive correction to attention scores
        """
        # Project query through S: q_sketch = q @ S^T → (batch, n_heads, seq_q, m)
        q_sketched = query.astype(mx.float32) @ self.S_t

        # Dot q_sketch with signs: (batch, n_heads, seq_q, m) @ (batch, n_kv_heads, m, seq_kv)
        # Handle GQA: expand kv_heads to match n_heads
        signs_t = mx.transpose(signs, (0, 1, 3, 2))  # (batch, n_kv_heads, m, seq_kv)
        correction = q_sketched @ signs_t  # broadcasts over GQA

        # Scale: sqrt(π/2)/m * ||r||
        correction = correction * self.correction_scale
        correction = correction * mx.expand_dims(residual_norm.astype(mx.float32), axis=-2)

        return correction
