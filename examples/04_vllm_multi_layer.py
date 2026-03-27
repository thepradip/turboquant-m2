#!/usr/bin/env python3
"""
Example 4: Multi-Layer KV Manager (vLLM-style)
================================================
Use TurboQuantKVManager to compress/decompress entire model KV caches.
This is the pattern you'd use in a serving framework like vLLM.
"""

import torch
from turboquant.integrations.vllm_adapter import TurboQuantKVManager

# ─── Model config (Qwen 3.5 4B as example) ───
NUM_LAYERS = 36
NUM_KV_HEADS = 4
HEAD_DIM = 128
BITS = 4

print(f"Model config: {NUM_LAYERS}L × {NUM_KV_HEADS}KV × {HEAD_DIM}dim")

# ─── Create manager ───
manager = TurboQuantKVManager(
    bits=BITS,
    head_dim=HEAD_DIM,
    num_layers=NUM_LAYERS,
)

# ─── Simulate a full KV cache ───
seq_len = 256
kv_cache = [
    (
        torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.float16),  # key
        torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.float16),  # value
    )
    for _ in range(NUM_LAYERS)
]

print(f"KV cache: {NUM_LAYERS} layers × ({1}, {NUM_KV_HEADS}, {seq_len}, {HEAD_DIM})")

# ─── Compress all layers at once ───
compressed = manager.compress_all(kv_cache)
print(f"Compressed: {len(compressed)} layers")

# ─── Decompress all layers ───
decompressed = manager.decompress_all(compressed)
print(f"Decompressed: {len(decompressed)} layers")

# ─── Verify quality ───
from turboquant import cosine_similarity

for i in [0, NUM_LAYERS // 2, NUM_LAYERS - 1]:
    orig_key = kv_cache[i][0]
    recon_key = decompressed[i][0]
    cos = cosine_similarity(orig_key, recon_key)
    print(f"  Layer {i:>2}: key cosine = {cos:.4f}")

# ─── Per-layer compress/decompress (for streaming) ───
print(f"\nPer-layer streaming example:")
key = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)  # 1 new token
value = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)

comp_k, comp_v = manager.compress_layer(layer_idx=0, key=key, value=value)
recon_k, recon_v = manager.decompress_layer(layer_idx=0, compressed_key=comp_k, compressed_value=comp_v)
print(f"  Compressed 1 token at layer 0: key shape={recon_k.shape}")

# ─── Memory savings report ───
print(f"\nMemory savings projections:")
for ctx in [4096, 32768, 131072]:
    savings = manager.memory_savings(num_kv_heads=NUM_KV_HEADS, context_length=ctx)
    print(f"  {ctx:>7} tokens: {savings['original_mb']:>7.1f}MB → {savings['compressed_mb']:>7.1f}MB "
          f"({savings['ratio']}x, saves {savings['savings_pct']}%)")
