#!/usr/bin/env python3
"""
Example 1: Basic TurboQuant Usage
==================================
Compress and decompress a KV-cache tensor in 5 lines.
"""

import torch
from turboquant import TurboQuant, cosine_similarity

# ─── Simulate a KV-cache tensor ───
# Shape: (batch, num_kv_heads, seq_len, head_dim)
# This is what a real model produces during inference
batch, num_heads, seq_len, head_dim = 1, 8, 512, 128
kv_tensor = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16)

print(f"Original KV tensor: {kv_tensor.shape}")
print(f"Original memory:    {kv_tensor.numel() * 2 / 1024:.1f} KB")

# ─── Compress ───
tq = TurboQuant(bits=4, head_dim=128)
compressed = tq.compress(kv_tensor)

# ─── Decompress ───
reconstructed = tq.decompress(compressed)

# ─── Measure quality ───
cos = cosine_similarity(kv_tensor, reconstructed)
mem = tq.memory_bytes(compressed)

print(f"\nCompressed memory:  {mem['compressed'] / 1024:.1f} KB")
print(f"Compression ratio:  {mem['ratio']:.1f}x")
print(f"Memory saved:       {mem['savings_pct']:.1f}%")
print(f"Cosine similarity:  {cos:.4f}  (1.0 = perfect)")
print(f"\nStats: {tq.stats()}")
