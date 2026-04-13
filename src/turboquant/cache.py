"""
TurboQuant KV Cache — compress on insert, dequantize incrementally.

Stores only compressed indices + norms permanently.
Maintains a dequantized FP16 decode buffer that grows incrementally —
only new tokens are dequantized each step, not the full sequence.

Two modes:
  use_wht=False: Dense rotation (backward compatible)
  use_wht=True: Walsh-Hadamard Transform (from paper, O(d log d))
"""

import mlx.core as mx
from .compressor import PolarQuantMLX


class TurboQuantCache:
    """
    Drop-in KVCache replacement. Compresses on insert, dequantizes incrementally.

    On each update_and_fetch:
      1. Quantize new K/V tokens → append to compressed indices
      2. Dequantize ONLY the new tokens → append to FP16 decode buffer
      3. Return the full decode buffer for SDPA

    The decode buffer grows by 1 token per step (O(1) dequantize).
    Compressed indices are the authoritative storage.
    """

    step = 256  # MLX checks this

    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 4,
                 layer_idx: int = 0, use_wht: bool = False):
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        self.key_mse = PolarQuantMLX(head_dim, key_bits, seed=42 + layer_idx, use_wht=use_wht)
        self.val_mse = PolarQuantMLX(head_dim, value_bits, seed=1000 + layer_idx, use_wht=use_wht)

        # Compressed storage (authoritative)
        self.k_indices = None    # uint8: (B, n_kv, seq, head_dim)
        self.k_norms = None      # float16: (B, n_kv, seq, 1)
        self.v_indices = None
        self.v_norms = None

        # Dequantized decode buffer (incremental, for SDPA)
        self._k_buf = None       # float16: (B, n_kv, seq, head_dim)
        self._v_buf = None
        self.offset = 0

    def _dequant_and_scale(self, indices, norms, compressor):
        """Dequantize indices and scale by norms. Returns float16."""
        hat = compressor.dequantize(indices)
        return (hat * norms.astype(mx.float32)).astype(mx.float16)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress new tokens, dequantize only them, append to buffer.

        Args:
            keys: (B, n_kv_heads, new_seq, head_dim) — after RoPE
            values: (B, n_kv_heads, new_seq, head_dim)

        Returns:
            (full_keys, full_values) — dequantized FP16 for SDPA
        """
        k_f = keys.astype(mx.float32)
        v_f = values.astype(mx.float32)

        # Normalize and quantize
        k_norms = mx.maximum(mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True)), 1e-8)
        k_indices = self.key_mse.quantize(k_f / k_norms)

        v_norms = mx.maximum(mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True)), 1e-8)
        v_indices = self.val_mse.quantize(v_f / v_norms)

        k_norms_stored = k_norms.astype(mx.float16)
        v_norms_stored = v_norms.astype(mx.float16)

        # Append to compressed storage
        if self.k_indices is None:
            self.k_indices = k_indices
            self.k_norms = k_norms_stored
            self.v_indices = v_indices
            self.v_norms = v_norms_stored
        else:
            self.k_indices = mx.concatenate([self.k_indices, k_indices], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, k_norms_stored], axis=2)
            self.v_indices = mx.concatenate([self.v_indices, v_indices], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, v_norms_stored], axis=2)

        # Dequantize ONLY the new tokens (not the full sequence)
        k_new = self._dequant_and_scale(k_indices, k_norms_stored, self.key_mse)
        v_new = self._dequant_and_scale(v_indices, v_norms_stored, self.val_mse)

        # Append to decode buffer
        if self._k_buf is None:
            self._k_buf = k_new
            self._v_buf = v_new
        else:
            self._k_buf = mx.concatenate([self._k_buf, k_new], axis=2)
            self._v_buf = mx.concatenate([self._v_buf, v_new], axis=2)

        self.offset += keys.shape[2]

        return self._k_buf, self._v_buf

    @property
    def keys(self):
        return self._k_buf

    @keys.setter
    def keys(self, value):
        if value is None:
            self._k_buf = None
            self.k_indices = None
            self.k_norms = None

    @property
    def values(self):
        return self._v_buf

    @values.setter
    def values(self, value):
        if value is None:
            self._v_buf = None
            self.v_indices = None
            self.v_norms = None

    @property
    def nbytes(self):
        """Compressed storage size (not including decode buffer)."""
        total = 0
        if self.k_indices is not None:
            total += self.k_indices.nbytes + self.k_norms.nbytes
        if self.v_indices is not None:
            total += self.v_indices.nbytes + self.v_norms.nbytes
        return total

    def size(self):
        return self.offset

    def make_mask(self, N, return_array, window_size=None):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array=return_array, window_size=window_size)

    @property
    def state(self):
        if self._k_buf is None:
            return None, None
        return self._k_buf, self._v_buf

    def is_trimmable(self):
        return False

    def empty(self):
        return self.k_indices is None
