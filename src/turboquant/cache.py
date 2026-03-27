"""
TurboQuant KV Cache — stores compressed keys (MSE + QJL) and values (MSE only).

Drop-in compatible with MLX's KVCache interface.
Keys use TurboQuantProd (MSE + QJL) because attention needs unbiased inner products.
Values use PolarQuant MSE only because they're used for weighted sum, not inner products.
"""

import mlx.core as mx
from .compressor import PolarQuantMLX
from .qjl import QJLMLX
from .codebook import build_codebook, build_rotation


class TurboQuantCache:
    """
    Compressed KV cache for one transformer layer.

    Stores:
      Keys:   MSE indices (uint8) + QJL signs (packed uint8) + residual norms (fp16) + vector norms (fp16)
      Values: MSE indices (uint8) + vector norms (fp16)

    On update_and_fetch:
      - Compresses new K,V tokens
      - Returns (keys_for_attention, values_for_attention) where:
        - keys include metadata needed for QJL-corrected attention
        - values are MSE-dequantized

    The attention function must use turboquant_attention() instead of standard Q@K^T.
    """

    # MLX checks for this
    step = 256

    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 4,
                 layer_idx: int = 0):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.layer_idx = layer_idx

        # Key compressor: (bits-1) for MSE, 1 for QJL
        mse_bits = max(key_bits - 1, 1)
        self.key_mse = PolarQuantMLX(head_dim, mse_bits, seed=42 + layer_idx)
        self.key_qjl = QJLMLX(head_dim, seed=43 + layer_idx)

        # Value compressor: full bits for MSE (no QJL needed for values)
        self.val_mse = PolarQuantMLX(head_dim, value_bits, seed=1000 + layer_idx)

        # Stored compressed data
        self.key_indices = None      # uint8, (..., head_dim)
        self.key_signs = None        # float32, (..., m) — unpacked for fast attention
        self.key_residual_norms = None  # fp16, (...,)
        self.key_norms = None        # fp16, (...,)
        self.val_indices = None      # uint8, (..., head_dim)
        self.val_norms = None        # fp16, (...,)
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """
        Compress new K,V and append to cache.

        Args:
            keys: (batch, n_kv_heads, new_seq, head_dim)
            values: (batch, n_kv_heads, new_seq, head_dim)

        Returns:
            Tuple of:
              - self (cache reference, attention function reads compressed data directly)
              - dequantized values for attention output aggregation
        """
        k_float = keys.astype(mx.float32)
        v_float = values.astype(mx.float32)

        # Key norms
        k_norms = mx.sqrt(mx.sum(k_float * k_float, axis=-1))
        k_norms_safe = mx.maximum(k_norms, 1e-8)
        k_normalized = k_float / mx.expand_dims(k_norms_safe, axis=-1)

        # Key Stage 1: MSE quantize on unit sphere
        k_indices, k_residual = self.key_mse.quantize_with_residual(k_normalized)

        # Key Stage 2: QJL on residual
        k_signs, k_res_norms = self.key_qjl.compute_signs(k_residual)

        # Value: MSE quantize (full precision, no QJL)
        v_norms = mx.sqrt(mx.sum(v_float * v_float, axis=-1))
        v_norms_safe = mx.maximum(v_norms, 1e-8)
        v_normalized = v_float / mx.expand_dims(v_norms_safe, axis=-1)
        v_indices = self.val_mse.quantize(v_normalized)

        # Append to stored cache
        if self.key_indices is None:
            self.key_indices = k_indices
            self.key_signs = k_signs
            self.key_residual_norms = k_res_norms
            self.key_norms = k_norms.astype(mx.float16)
            self.val_indices = v_indices
            self.val_norms = v_norms.astype(mx.float16)
        else:
            self.key_indices = mx.concatenate([self.key_indices, k_indices], axis=2)
            self.key_signs = mx.concatenate([self.key_signs, k_signs], axis=2)
            self.key_residual_norms = mx.concatenate([self.key_residual_norms, k_res_norms], axis=2)
            self.key_norms = mx.concatenate([self.key_norms, k_norms.astype(mx.float16)], axis=2)
            self.val_indices = mx.concatenate([self.val_indices, v_indices], axis=2)
            self.val_norms = mx.concatenate([self.val_norms, v_norms.astype(mx.float16)], axis=2)

        self.offset += keys.shape[2]

        # Return dequantized keys and values for the attention function
        # Keys: dequantize MSE part + rescale (attention function adds QJL correction)
        k_mse = self.key_mse.dequantize(self.key_indices)
        k_mse = k_mse * mx.expand_dims(self.key_norms.astype(mx.float32), axis=-1)

        v_mse = self.val_mse.dequantize(self.val_indices)
        v_mse = v_mse * mx.expand_dims(self.val_norms.astype(mx.float32), axis=-1)

        return k_mse.astype(keys.dtype), v_mse.astype(values.dtype)

    def get_qjl_data(self):
        """Return QJL data needed for attention correction."""
        return {
            "signs": self.key_signs,
            "residual_norms": self.key_residual_norms,
            "key_norms": self.key_norms,
        }

    def size(self):
        return self.offset

    def make_mask(self, N, return_array, window_size=None):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array=return_array, window_size=window_size)

    @property
    def state(self):
        return self.key_indices, self.val_indices

    def is_trimmable(self):
        return False

    def empty(self):
        return self.key_indices is None
