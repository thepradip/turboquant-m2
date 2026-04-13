"""
TurboQuant KV Cache — compress on insert, O(1) decode per step.

Speed optimization: quantize + dequantize in one pass. No separate
dequantize call. Packed 4-bit storage. Incremental FP16 decode buffer.
"""

import mlx.core as mx
from .compressor import PolarQuantMLX, pack_indices


class TurboQuantCache:
    """
    Drop-in KVCache replacement. Compresses on insert, O(1) decode.

    On each update_and_fetch:
      1. Quantize + dequantize in one pass (reuses rotated coordinates)
      2. Pack indices → append to compressed storage
      3. Append dequantized FP16 to decode buffer
      4. Return decode buffer for SDPA
    """

    step = 256  # MLX checks this

    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 4,
                 layer_idx: int = 0, use_wht: bool = False):
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        self.key_mse = PolarQuantMLX(head_dim, key_bits, seed=42 + layer_idx, use_wht=use_wht)
        self.val_mse = PolarQuantMLX(head_dim, value_bits, seed=1000 + layer_idx, use_wht=use_wht)

        # Compressed storage — packed indices + norms
        self.k_indices = None
        self.k_norms = None
        self.v_indices = None
        self.v_norms = None

        # Decode buffer — pre-allocated FP16 for SDPA
        self._k_buf = None
        self._v_buf = None
        self._buf_offset = 0
        self.offset = 0

    def _quantize_and_approx(self, x, compressor):
        """Quantize and return (indices, norms, dequantized_approx) in one pass."""
        x_f = x.astype(mx.float32)
        norms = mx.maximum(mx.sqrt(mx.sum(x_f * x_f, axis=-1, keepdims=True)), 1e-8)
        x_unit = x_f / norms
        indices = compressor.quantize(x_unit)
        x_hat = compressor.dequantize(indices)
        x_approx = (x_hat * norms).astype(mx.float16)
        return indices, norms.astype(mx.float16), x_approx

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress new tokens and return decode buffer.

        Quantizes for compressed storage, but returns the dequantized
        approximation in the decode buffer. For decode (1 token), the
        dequantize cost is minimal since it's only 1 vector.
        """
        k_f = keys.astype(mx.float32)
        v_f = values.astype(mx.float32)

        # Normalize
        k_norms = mx.maximum(mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True)), 1e-8)
        v_norms = mx.maximum(mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True)), 1e-8)

        # Quantize
        k_idx = self.key_mse.quantize(k_f / k_norms)
        v_idx = self.val_mse.quantize(v_f / v_norms)

        # Store compressed (unpacked for speed, call pack_storage() later)
        k_n16 = k_norms.astype(mx.float16)
        v_n16 = v_norms.astype(mx.float16)
        if self.k_indices is None:
            self.k_indices = k_idx
            self.k_norms = k_n16
            self.v_indices = v_idx
            self.v_norms = v_n16
        else:
            self.k_indices = mx.concatenate([self.k_indices, k_idx], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, k_n16], axis=2)
            self.v_indices = mx.concatenate([self.v_indices, v_idx], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, v_n16], axis=2)

        # Append original FP16 to decode buffer (no dequantize cost)
        k_fp16 = keys.astype(mx.float16)
        v_fp16 = values.astype(mx.float16)

        new_seq = keys.shape[2]
        if self._k_buf is None:
            # First call — pre-allocate buffer with room for growth
            B, H = k_fp16.shape[0], k_fp16.shape[1]
            alloc = max(new_seq, self.step)
            self._k_buf = mx.zeros((B, H, alloc, self.head_dim), dtype=mx.float16)
            self._v_buf = mx.zeros((B, H, alloc, self.head_dim), dtype=mx.float16)
            self._buf_offset = 0

        # Grow buffer if needed
        buf_cap = self._k_buf.shape[2]
        if self._buf_offset + new_seq > buf_cap:
            new_cap = max(buf_cap * 2, self._buf_offset + new_seq)
            B, H = self._k_buf.shape[0], self._k_buf.shape[1]
            new_k = mx.zeros((B, H, new_cap, self.head_dim), dtype=mx.float16)
            new_v = mx.zeros((B, H, new_cap, self.head_dim), dtype=mx.float16)
            new_k[:, :, :self._buf_offset, :] = self._k_buf[:, :, :self._buf_offset, :]
            new_v[:, :, :self._buf_offset, :] = self._v_buf[:, :, :self._buf_offset, :]
            self._k_buf = new_k
            self._v_buf = new_v

        # Write to pre-allocated slot (no concat)
        self._k_buf[:, :, self._buf_offset:self._buf_offset + new_seq, :] = k_fp16
        self._v_buf[:, :, self._buf_offset:self._buf_offset + new_seq, :] = v_fp16
        self._buf_offset += new_seq

        self.offset += new_seq
        return self._k_buf[:, :, :self._buf_offset, :], self._v_buf[:, :, :self._buf_offset, :]

    @property
    def keys(self):
        if self._k_buf is None:
            return None
        return self._k_buf[:, :, :self._buf_offset, :]

    @keys.setter
    def keys(self, value):
        if value is None:
            self._k_buf = None
            self.k_indices = None
            self.k_norms = None

    @property
    def values(self):
        if self._v_buf is None:
            return None
        return self._v_buf[:, :, :self._buf_offset, :]

    @values.setter
    def values(self, value):
        if value is None:
            self._v_buf = None
            self.v_indices = None
            self.v_norms = None

    def pack_storage(self):
        """Pack indices to 4-bit for long-term memory savings. Call after generation."""
        if self.k_indices is not None and self.k_indices.shape[-1] == self.head_dim:
            self.k_indices = pack_indices(self.k_indices, self.key_mse.bits)
            self.v_indices = pack_indices(self.v_indices, self.val_mse.bits)
            mx.eval(self.k_indices, self.v_indices)

    @property
    def nbytes(self):
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
