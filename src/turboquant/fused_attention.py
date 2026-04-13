"""
Fused TurboQuant Attention — uses MLX's built-in fast SDPA.

Strategy: dequantize K/V per-layer on-demand, feed to mx.fast.scaled_dot_product_attention,
let MLX handle the fused computation. The dequantized FP16 is transient — not stored.

This avoids:
1. Storing full FP16 K/V permanently (only compressed indices stored)
2. Building custom Metal kernels (MLX's SDPA is already optimal)
"""

import mlx.core as mx


def tq_sdpa(queries, cache, scale, mask=None):
    """Attention using MLX's fast SDPA with on-demand dequantization.

    Dequantizes K/V from cache's compressed indices, calls mx.fast.scaled_dot_product_attention,
    then the dequantized arrays are transient (freed by MLX after use).

    Args:
        queries: (B, n_heads, seq_q, head_dim)
        cache: TurboQuantCache with compressed indices
        scale: 1/sqrt(head_dim)
        mask: attention mask

    Returns:
        (B, n_heads, seq_q, head_dim)
    """
    # Dequantize on-demand, compute attention, eval immediately to free FP16
    keys = cache._dequantize_keys()
    values = cache._dequantize_values()

    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask)

    # Force eval to free the transient dequantized K/V immediately
    mx.eval(output)
    return output
