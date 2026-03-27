#!/usr/bin/env python3
"""
Example 3: Compress Real KV Cache from a HuggingFace Model
============================================================
Extracts the actual KV cache from a model and compresses it.

Requires: pip install turboquant[transformers]
"""

import torch
from turboquant import TurboQuant, cosine_similarity, measure_distortion

# ─── Step 1: Load model ───
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for demo
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
)
model.eval()

# ─── Step 2: Extract KV cache ───
prompt = "Explain how transformers work in deep learning."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

print(f"Prompt tokens: {inputs.input_ids.shape[1]}")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

kv_cache = outputs.past_key_values
# kv_cache is a tuple of (key, value) per layer
# Each key/value shape: (batch, num_kv_heads, seq_len, head_dim)

num_layers = len(kv_cache)
key_shape = kv_cache[0][0].shape
print(f"KV cache: {num_layers} layers × {key_shape}")

head_dim = key_shape[-1]
num_kv_heads = key_shape[1]
seq_len = key_shape[2]

# ─── Step 3: Compress every layer ───
print(f"\nCompressing KV cache at 4-bit...")

total_original = 0
total_compressed = 0
cos_scores = []

for layer_idx, (key, value) in enumerate(kv_cache):
    # One TurboQuant per layer (different rotation per layer)
    tq_k = TurboQuant(bits=4, head_dim=head_dim, seed=42 + layer_idx)
    tq_v = TurboQuant(bits=4, head_dim=head_dim, seed=1000 + layer_idx)

    # Compress
    comp_k = tq_k.compress(key)
    comp_v = tq_v.compress(value)

    # Decompress
    recon_k = tq_k.decompress(comp_k)
    recon_v = tq_v.decompress(comp_v)

    # Measure quality
    cos_k = cosine_similarity(key, recon_k)
    cos_scores.append(cos_k)

    # Memory
    mem_k = tq_k.memory_bytes(comp_k)
    mem_v = tq_v.memory_bytes(comp_v)
    total_original += mem_k["original"] + mem_v["original"]
    total_compressed += mem_k["compressed"] + mem_v["compressed"]

    if layer_idx < 3 or layer_idx == num_layers - 1:
        print(f"  Layer {layer_idx:>2}: cosine={cos_k:.4f}")

# ─── Step 4: Report ───
ratio = total_original / total_compressed
avg_cos = sum(cos_scores) / len(cos_scores)

print(f"\n{'=' * 50}")
print(f"  Model:          {model_id}")
print(f"  Layers:         {num_layers}")
print(f"  KV heads:       {num_kv_heads}")
print(f"  Head dim:       {head_dim}")
print(f"  Seq len:        {seq_len}")
print(f"  Original KV:    {total_original / 1024:.1f} KB")
print(f"  Compressed KV:  {total_compressed / 1024:.1f} KB")
print(f"  Ratio:          {ratio:.1f}x")
print(f"  Avg cosine sim: {avg_cos:.4f}")
print(f"{'=' * 50}")

# ─── Step 5: Project memory at scale ───
fp16_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # bytes

print(f"\nMemory projections (this model at various context lengths):")
print(f"  {'Context':>8} | {'FP16 KV':>10} | {'TQ 4-bit':>10} | {'Saved':>7}")
print(f"  {'─' * 45}")
for ctx in [4096, 16384, 32768, 65536, 131072]:
    fp16_mb = fp16_per_token * ctx / 1024 / 1024
    tq_mb = fp16_mb / ratio
    pct = (1 - 1 / ratio) * 100
    print(f"  {ctx:>8} | {fp16_mb:>8.1f}MB | {tq_mb:>8.1f}MB | {pct:>5.0f}%")
