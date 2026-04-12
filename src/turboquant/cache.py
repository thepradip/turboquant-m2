"""
TurboQuant KV Cache — stores compressed keys/values.

v0.5.0: Stores uint8 indices + float16 norms (real memory savings).
Dequantizes on-the-fly when returning keys/values to the model.

Keys: MSE indices + norms
Values: MSE indices + norms
"""

import mlx.core as mx
from .compressor import PolarQuantMLX, pack_indices, unpack_indices


class TurboQuantCache:
    """
    Drop-in KVCache replacement that compresses on insert.

    v0.5.0: Stores uint8 indices + float16 norms internally.
    Returns dequantized FP16 tensors to the model (same API as KVCache).

    Shapes match MLX's KVCache exactly:
      keys/values returned: (batch, n_kv_heads, seq_len, head_dim)
    """

    step = 256  # MLX checks this

    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 4,
                 layer_idx: int = 0, fused: bool = False):
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.fused = fused  # If True, update_and_fetch returns empty placeholders

        self.key_mse = PolarQuantMLX(head_dim, key_bits, seed=42 + layer_idx)
        self.val_mse = PolarQuantMLX(head_dim, value_bits, seed=1000 + layer_idx)

        # Compressed storage (real memory savings)
        self.k_indices = None    # uint8: (B, n_kv, seq, head_dim)
        self.k_norms = None      # float16: (B, n_kv, seq, 1)
        self.v_indices = None    # uint8: (B, n_kv, seq, head_dim)
        self.v_norms = None      # float16: (B, n_kv, seq, 1)
        self.offset = 0

        # For fused mode: store the most recent uncompressed token for SDPA
        self._new_keys = None    # (B, n_kv, n_new, head_dim) float16
        self._new_values = None  # (B, n_kv, n_new, head_dim) float16

    def _dequantize_keys(self) -> mx.array:
        """Reconstruct FP16 keys from packed indices + norms."""
        k_idx = unpack_indices(self.k_indices, self.key_mse.bits, self.head_dim)
        k_hat = self.key_mse.dequantize(k_idx)
        return (k_hat * self.k_norms.astype(mx.float32)).astype(mx.float16)

    def _dequantize_values(self) -> mx.array:
        """Reconstruct FP16 values from packed indices + norms."""
        v_idx = unpack_indices(self.v_indices, self.val_mse.bits, self.head_dim)
        v_hat = self.val_mse.dequantize(v_idx)
        return (v_hat * self.v_norms.astype(mx.float32)).astype(mx.float16)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """
        Compress new K,V and return values for attention.

        In fused mode: compresses to indices, returns the new FP16 tokens only
        (fused SDPA reads compressed indices directly from self).
        In standard mode: returns full dequantized sequence (backward compat).

        Args:
            keys: (B, n_kv_heads, new_seq, head_dim) — after RoPE
            values: (B, n_kv_heads, new_seq, head_dim)

        Returns:
            (keys_for_sdpa, values_for_sdpa)
        """
        k_f = keys.astype(mx.float32)
        v_f = values.astype(mx.float32)

        # Key norms + normalize
        k_norms = mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True))
        k_norms = mx.maximum(k_norms, 1e-8)
        k_unit = k_f / k_norms
        k_indices = self.key_mse.quantize(k_unit)

        # Value: normalize + quantize
        v_norms = mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True))
        v_norms = mx.maximum(v_norms, 1e-8)
        v_unit = v_f / v_norms
        v_indices = self.val_mse.quantize(v_unit)

        # Append to compressed storage
        k_norms_stored = k_norms.astype(mx.float16)
        v_norms_stored = v_norms.astype(mx.float16)

        # Pack indices for memory savings (4-bit: 2 per byte = 2x smaller)
        k_packed = pack_indices(k_indices, self.key_mse.bits)
        v_packed = pack_indices(v_indices, self.val_mse.bits)

        if self.k_indices is None:
            self.k_indices = k_packed
            self.k_norms = k_norms_stored
            self.v_indices = v_packed
            self.v_norms = v_norms_stored
        else:
            self.k_indices = mx.concatenate([self.k_indices, k_packed], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, k_norms_stored], axis=2)
            self.v_indices = mx.concatenate([self.v_indices, v_packed], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, v_norms_stored], axis=2)

        self.offset += keys.shape[2]

        if self.fused:
            # Fused mode: return the new FP16 tokens only.
            # Fused SDPA will read compressed indices directly from self,
            # and use these new tokens as the uncompressed window.
            self._new_keys = keys
            self._new_values = values
            # Return placeholders that match expected shapes but fused SDPA ignores
            return keys, values

        return self._dequantize_keys(), self._dequantize_values()

    @property
    def keys(self):
        if self.k_indices is None:
            return None
        return self._dequantize_keys()

    @keys.setter
    def keys(self, value):
        if value is None:
            self.k_indices = None
            self.k_norms = None

    @property
    def values(self):
        if self.v_indices is None:
            return None
        return self._dequantize_values()

    @values.setter
    def values(self, value):
        if value is None:
            self.v_indices = None
            self.v_norms = None

    @property
    def nbytes(self):
        """Real memory footprint (compressed)."""
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
        if self.k_indices is None:
            return None, None
        return self._dequantize_keys(), self._dequantize_values()

    def is_trimmable(self):
        return False

    def empty(self):
        return self.k_indices is None
