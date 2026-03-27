#!/usr/bin/env python3
"""
Example 5: Ollama + TurboQuant Memory Projection
==================================================
Compare standard FP16 KV cache vs TurboQuant compressed KV cache
for models served via Ollama.

Note: Ollama uses GGUF for WEIGHT quantization.
      TurboQuant compresses the KV-CACHE (runtime memory).
      They are COMPLEMENTARY — use both together.

Requires: pip install turboquant[ollama]
          ollama must be running: ollama serve
"""

from turboquant.integrations.ollama_adapter import (
    list_models,
    get_model_info,
    generate,
    project_kv_memory,
)

# ─── Step 1: Check available models ───
print("Available Ollama models:")
try:
    models = list_models()
    for m in models[:10]:
        print(f"  - {m}")
except Exception as e:
    print(f"  Ollama not running: {e}")
    print("  Start with: ollama serve")
    print("\n  Showing projection with Qwen 3.5 2B defaults instead...\n")

    # Can still use projections without Ollama running
    projections = project_kv_memory(
        num_layers=28, num_kv_heads=4, head_dim=128
    )

    print(f"KV-Cache Memory: FP16 vs TurboQuant 4-bit")
    print(f"Model: Qwen 3.5 2B (28L × 4KV × 128dim)\n")
    print(f"  {'Context':>8} | {'FP16 KV':>10} | {'TQ 4-bit':>10} | {'Saved':>7} | {'Saved MB':>9}")
    print(f"  {'─' * 55}")
    for p in projections:
        print(f"  {p['context_length']:>8} | {p['fp16_kv_mb']:>8.1f}MB | {p['tq_kv_mb']:>8.1f}MB | "
              f"{p['saved_pct']:>5.1f}% | {p['saved_mb']:>7.1f}MB")
    exit()

# ─── Step 2: Get model info ───
model = models[0]  # Use first available model
print(f"\nModel: {model}")

info = get_model_info(model)
print(f"  Family:       {info.get('family')}")
print(f"  Size:         {info.get('parameter_size')}")
print(f"  Quantization: {info.get('quantization')}")

# ─── Step 3: Run a generation benchmark ───
print(f"\nGenerating text...")
result = generate(model, "What is the transformer architecture?", max_tokens=100)
print(f"  Tokens:    {result['tokens']}")
print(f"  Speed:     {result['tok_per_sec']} tok/s")
print(f"  TTFT:      {result['ttft_ms']} ms")
print(f"  Response:  {result['response'][:200]}...")

# ─── Step 4: Project KV savings ───
num_layers = info.get('num_hidden_layers', 28)
num_kv_heads = info.get('num_key_value_heads', 4)
head_dim = info.get('head_dim', 128)

projections = project_kv_memory(
    num_layers=num_layers,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
)

print(f"\nKV-Cache Memory Savings with TurboQuant 4-bit:")
print(f"  {'Context':>8} | {'FP16':>10} | {'TQ 4-bit':>10} | {'Saved':>7}")
print(f"  {'─' * 45}")
for p in projections:
    print(f"  {p['context_length']:>8} | {p['fp16_kv_mb']:>8.1f}MB | {p['tq_kv_mb']:>8.1f}MB | {p['saved_pct']:>5.1f}%")
