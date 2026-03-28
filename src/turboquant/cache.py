"""
TurboQuant KV Cache — stores compressed keys with QJL correction data.

Keys: MSE indices + QJL signs + residual norms + vector norms
Values: MSE dequantized (values tolerate MSE-only, no QJL needed)

The cache returns MSE-dequantized tensors for standard attention,
plus QJL data that the patched attention uses for bias correction.
"""

import mlx.core as mx
from .compressor import PolarQuantMLX
from .qjl import QJLMLX


class TurboQuantCache:
    """
    Drop-in KVCache replacement that compresses on insert.

    Shapes match MLX's KVCache exactly:
      keys/values: (batch, n_kv_heads, seq_len, head_dim)
    """

    step = 256  # MLX checks this

    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 4,
                 layer_idx: int = 0):
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        # Full bits for MSE — quality over compression
        self.key_mse = PolarQuantMLX(head_dim, key_bits, seed=42 + layer_idx)
        self.key_qjl = QJLMLX(head_dim, seed=43 + layer_idx)
        self.val_mse = PolarQuantMLX(head_dim, value_bits, seed=1000 + layer_idx)

        # Dequantized keys/values (what the model sees)
        self.keys = None
        self.values = None
        self.offset = 0

        # QJL correction data (used by patched attention)
        self.qjl_signs = None       # (B, n_kv, seq, m) float32 ±1
        self.qjl_res_norms = None   # (B, n_kv, seq) float16
        self.qjl_key_norms = None   # (B, n_kv, seq) float16

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """
        Compress new K,V and return dequantized versions.
        Also stores QJL correction data for attention.

        Args:
            keys: (B, n_kv_heads, new_seq, head_dim) — after RoPE
            values: (B, n_kv_heads, new_seq, head_dim)

        Returns:
            (dequantized_keys, dequantized_values) — same shapes as input cache
        """
        k_f = keys.astype(mx.float32)
        v_f = values.astype(mx.float32)

        # Key norms + normalize
        k_norms = mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True))
        k_norms = mx.maximum(k_norms, 1e-8)
        k_unit = k_f / k_norms

        # Key MSE quantize on unit sphere + get residual
        k_indices, k_residual = self.key_mse.quantize_with_residual(k_unit)
        k_hat_unit = self.key_mse.dequantize(k_indices)

        # Key QJL on residual
        k_signs, k_res_norms = self.key_qjl.compute_signs(k_residual)

        # Key dequantize: rescale by norms
        k_deq = (k_hat_unit * k_norms).astype(keys.dtype)

        # Value: MSE quantize + dequantize
        v_norms = mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True))
        v_norms = mx.maximum(v_norms, 1e-8)
        v_unit = v_f / v_norms
        v_hat_unit = self.val_mse.dequantize(self.val_mse.quantize(v_unit))
        v_deq = (v_hat_unit * v_norms).astype(values.dtype)

        # Append to storage
        k_norms_sq = k_norms.squeeze(-1).astype(mx.float16)
        if self.keys is None:
            self.keys = k_deq
            self.values = v_deq
            self.qjl_signs = k_signs
            self.qjl_res_norms = k_res_norms
            self.qjl_key_norms = k_norms_sq
        else:
            self.keys = mx.concatenate([self.keys, k_deq], axis=2)
            self.values = mx.concatenate([self.values, v_deq], axis=2)
            self.qjl_signs = mx.concatenate([self.qjl_signs, k_signs], axis=2)
            self.qjl_res_norms = mx.concatenate([self.qjl_res_norms, k_res_norms], axis=2)
            self.qjl_key_norms = mx.concatenate([self.qjl_key_norms, k_norms_sq], axis=2)

        self.offset += keys.shape[2]
        return self.keys, self.values

    def size(self):
        return self.offset

    def make_mask(self, N, return_array, window_size=None):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array=return_array, window_size=window_size)

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    def is_trimmable(self):
        return False

    def empty(self):
        return self.keys is None
