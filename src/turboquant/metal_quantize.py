"""
Fused Metal kernel for TurboQuant quantization.

Fuses: norm + normalize + WHT (signs + butterfly) + boundary comparisons
into a single GPU dispatch per (batch, head) pair.

Input: raw float16 KV vector (head_dim elements)
Output: uint8 indices (head_dim) + float16 norm (1 scalar)
"""

import math
import mlx.core as mx


_QUANTIZE_KERNEL_SOURCE = """
    // Each thread processes one (batch, kv_head) vector of head_dim elements.
    uint bh = thread_position_in_grid.x;  // batch * n_kv + kv_head
    if (bh >= total_vectors) return;

    uint base = bh * head_dim;

    // --- Step 1: Load vector and compute L2 norm ---
    float sum_sq = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        float v = (float)input_vec[base + d];
        sum_sq += v * v;
    }
    float norm = sqrt(max(sum_sq, 1e-16f));
    norms_out[bh] = (float16_t)norm;
    float inv_norm = 1.0f / norm;

    // --- Step 2: Normalize + apply WHT random signs ---
    // Work buffer in registers (max head_dim=256)
    float buf[256];
    for (uint d = 0; d < head_dim; d++) {
        float v = (float)input_vec[base + d] * inv_norm;
        buf[d] = v * signs[d];  // random ±1 signs
    }

    // --- Step 3: Walsh-Hadamard butterfly ---
    float scale = 1.0f / sqrt((float)head_dim);
    uint h = 1;
    while (h < head_dim) {
        for (uint i = 0; i < head_dim; i += 2 * h) {
            for (uint j = i; j < i + h; j++) {
                float a = buf[j];
                float b = buf[j + h];
                buf[j] = a + b;
                buf[j + h] = a - b;
            }
        }
        h *= 2;
    }
    for (uint d = 0; d < head_dim; d++) {
        buf[d] *= scale;
    }

    // --- Step 4: Boundary comparisons → indices ---
    for (uint d = 0; d < head_dim; d++) {
        uint8_t idx = 0;
        for (uint b = 0; b < n_boundaries; b++) {
            if (buf[d] >= boundaries[b]) idx++;
        }
        indices_out[base + d] = idx;
    }
"""


_kernel = None


def metal_quantize(x, signs, boundaries):
    """Fused Metal quantize: norm + WHT + boundaries in one dispatch.

    Args:
        x: (B, n_kv, new_seq, head_dim) float16 — raw KV vectors
        signs: (head_dim,) float32 — WHT random ±1 signs
        boundaries: (n_boundaries,) float32 — decision boundaries

    Returns:
        indices: (B, n_kv, new_seq, head_dim) uint8
        norms: (B, n_kv, new_seq, 1) float16
    """
    global _kernel

    orig_shape = x.shape
    head_dim = orig_shape[-1]
    total_vectors = 1
    for s in orig_shape[:-1]:
        total_vectors *= s

    x_flat = x.reshape(total_vectors, head_dim).astype(mx.float16)

    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="tq_fused_quantize",
            input_names=["input_vec", "signs", "boundaries"],
            output_names=["indices_out", "norms_out"],
            source=_QUANTIZE_KERNEL_SOURCE,
        )

    n_boundaries = boundaries.shape[0]

    outputs = _kernel(
        inputs=[x_flat, signs.astype(mx.float32), boundaries.astype(mx.float32)],
        output_shapes=[(total_vectors * head_dim,), (total_vectors,)],
        output_dtypes=[mx.uint8, mx.float16],
        grid=(total_vectors, 1, 1),
        threadgroup=(min(total_vectors, 256), 1, 1),
        template=[
            ("head_dim", head_dim),
            ("total_vectors", total_vectors),
            ("n_boundaries", n_boundaries),
        ],
    )

    indices = outputs[0].reshape(*orig_shape[:-1], head_dim)
    norms = outputs[1].reshape(*orig_shape[:-1], 1)

    return indices, norms
