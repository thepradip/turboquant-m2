#!/usr/bin/env python3
"""
Example 2: Compare 2-bit, 3-bit, 4-bit Compression
====================================================
Shows the quality vs compression tradeoff.
"""

import torch
from turboquant import TurboQuant, measure_distortion

head_dim = 128
kv = torch.randn(1, 8, 256, head_dim, dtype=torch.float16)

print(f"Input: {kv.shape} ({kv.numel() * 2 / 1024:.0f} KB)\n")
print(f"{'Bits':>5} | {'Ratio':>6} | {'Cosine':>8} | {'IP Corr':>8} | {'MSE':>10} | {'Compressed':>11}")
print(f"{'─' * 65}")

for bits in [2, 3, 4]:
    tq = TurboQuant(bits=bits, head_dim=head_dim)
    comp = tq.compress(kv)
    recon = tq.decompress(comp)

    metrics = measure_distortion(kv, recon)
    mem = tq.memory_bytes(comp)

    print(
        f"  {bits}-bit | {mem['ratio']:>5.1f}x | {metrics['cosine_similarity_mean']:>8.4f} | "
        f"{metrics['inner_product_correlation']:>8.4f} | {metrics['mse']:>10.6f} | "
        f"{mem['compressed'] / 1024:>8.1f} KB"
    )

print(f"""
Recommendations:
  4-bit  →  Near-lossless. Use for production.
  3-bit  →  Great balance. Use when memory is tight.
  2-bit  →  Aggressive. Use for very long contexts.
""")
