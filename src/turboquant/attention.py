"""
TurboQuant Attention — standard attention + QJL correction.

This replaces scaled_dot_product_attention when TurboQuant cache is used.
The QJL correction removes inner product bias introduced by MSE quantization.
"""

import math
import mlx.core as mx


def turboquant_attention(
    queries: mx.array,
    keys_mse: mx.array,
    values_mse: mx.array,
    qjl_data: dict,
    qjl: "QJLMLX",
    scale: float,
    mask=None,
) -> mx.array:
    """
    Compute attention with QJL correction on compressed keys.

    scores = (Q @ K_mse^T + QJL_correction) * scale
    output = softmax(scores) @ V_mse

    Args:
        queries: (B, n_heads, seq_q, head_dim)
        keys_mse: (B, n_kv_heads, seq_kv, head_dim) — MSE-dequantized keys
        values_mse: (B, n_kv_heads, seq_kv, head_dim) — MSE-dequantized values
        qjl_data: dict with 'signs', 'residual_norms', 'key_norms'
        qjl: QJLMLX instance for this layer
        scale: 1/sqrt(head_dim)
        mask: attention mask

    Returns:
        output: (B, n_heads, seq_q, head_dim)
    """
    B, n_heads, seq_q, d = queries.shape
    _, n_kv_heads, seq_kv, _ = keys_mse.shape

    # GQA expansion: repeat KV heads to match query heads
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        keys_mse = mx.repeat(keys_mse, n_rep, axis=1)
        values_mse = mx.repeat(values_mse, n_rep, axis=1)

    # Term 1: Standard MSE attention scores
    scores = (queries.astype(mx.float32) @ mx.transpose(keys_mse.astype(mx.float32), (0, 1, 3, 2))) * scale

    # Term 2: QJL correction (removes inner product bias)
    if qjl_data is not None and qjl_data.get("signs") is not None:
        signs = qjl_data["signs"]          # (B, n_kv_heads, seq_kv, m)
        res_norms = qjl_data["residual_norms"]  # (B, n_kv_heads, seq_kv)
        key_norms = qjl_data["key_norms"]  # (B, n_kv_heads, seq_kv)

        # GQA expansion for QJL data
        if n_rep > 1:
            signs = mx.repeat(signs, n_rep, axis=1)
            res_norms = mx.repeat(res_norms, n_rep, axis=1)
            key_norms = mx.repeat(key_norms, n_rep, axis=1)

        # Project query: q_sketch = q @ S^T
        q_sketched = queries.astype(mx.float32) @ qjl.S_t  # (B, n_heads, seq_q, m)

        # Dot with signs: (B, n_heads, seq_q, m) @ (B, n_heads, m, seq_kv)
        signs_t = mx.transpose(signs.astype(mx.float32), (0, 1, 3, 2))
        qjl_scores = q_sketched @ signs_t  # (B, n_heads, seq_q, seq_kv)

        # Scale by correction factor and residual norms
        # residual_norms need to account for the original key norm
        # (residual was computed on normalized vectors, so scale by key_norm)
        effective_res_norm = res_norms.astype(mx.float32) * key_norms.astype(mx.float32)
        qjl_scores = qjl_scores * qjl.correction_scale
        qjl_scores = qjl_scores * mx.expand_dims(effective_res_norm, axis=-2)
        qjl_scores = qjl_scores * scale  # same attention scale as MSE term

        scores = scores + qjl_scores

    # Apply mask
    if mask is not None:
        scores = scores + mask.astype(scores.dtype)

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Aggregate values
    output = weights @ values_mse.astype(mx.float32)

    return output.astype(queries.dtype)
