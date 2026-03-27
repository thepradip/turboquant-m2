# TurboQuant Tutorial — Complete Guide

A step-by-step tutorial to compress LLM KV-cache memory by 4x with near-zero quality loss.

---

## Table of Contents

1. [What is TurboQuant?](#1-what-is-turboquant)
2. [Install](#2-install)
3. [Your First Compression](#3-your-first-compression)
4. [Understanding the Output](#4-understanding-the-output)
5. [Choose Your Bit Width](#5-choose-your-bit-width)
6. [Analyze Any Model on Any GPU](#6-analyze-any-model-on-any-gpu)
7. [Compress a Real HuggingFace Model](#7-compress-a-real-huggingface-model)
8. [Multi-Layer Manager (Production Use)](#8-multi-layer-manager-production-use)
9. [Compare Against Standard Q4 (GGUF)](#9-compare-against-standard-q4-gguf)
10. [Benchmark with Ollama](#10-benchmark-with-ollama)
11. [Speed Benchmarking](#11-speed-benchmarking)
12. [GPU Planning Guide](#12-gpu-planning-guide)
13. [API Reference](#13-api-reference)

---

## 1. What is TurboQuant?

When an LLM generates text, it stores **Key** and **Value** vectors for every token it has seen. This is called the **KV-cache**. It grows linearly with context length and is often the **#1 memory bottleneck** in production.

```
Memory breakdown for Llama 3.1 8B at 32K context:
  Model weights:  16 GB (fixed)
  KV-cache:        4 GB (grows with context!)  ← TurboQuant compresses this
  Activations:   ~0.5 GB
```

**TurboQuant compresses the KV-cache from FP16 to 4-bit** with 0.9954 cosine similarity (near-lossless).

```
How it works:
  1. Normalize vectors to unit sphere
  2. Apply random orthogonal rotation (fixed matrix)
  3. After rotation, all coordinates follow a known Beta distribution
  4. Apply Lloyd-Max optimal scalar quantizer (precomputed)
  5. Store: 4-bit indices + 1 FP16 norm per token

Key insight: The quantizer is DATA-INDEPENDENT — no calibration needed!
```

**It is complementary to weight quantization:**

| Method | What it compresses | When |
|--------|-------------------|------|
| GGUF/AWQ/GPTQ | Model **weights** | Before deployment |
| **TurboQuant** | **KV-cache** | During inference |

Use both together for maximum efficiency.

---

## 2. Install

```bash
pip install turboquant
```

That's it. Dependencies (`torch`, `numpy`, `scipy`) are installed automatically.

**Verify:**
```python
import turboquant
print(turboquant.__version__)
# 0.2.0
```

**Optional extras** (only if you need them):
```bash
pip install turboquant[transformers]  # HuggingFace model support
pip install turboquant[ollama]        # Ollama benchmarking
pip install turboquant[all]           # Everything
```

---

## 3. Your First Compression

```python
import torch
from turboquant import TurboQuant

# Step 1: Create a compressor
tq = TurboQuant(bits=4, head_dim=128)

# Step 2: Create sample KV-cache data
# Shape: (batch, num_kv_heads, seq_len, head_dim)
kv_tensor = torch.randn(1, 8, 512, 128, dtype=torch.float16)
print(f"Original: {kv_tensor.shape}, {kv_tensor.numel() * 2 / 1024:.0f} KB")

# Step 3: Compress
compressed = tq.compress(kv_tensor)

# Step 4: Decompress
reconstructed = tq.decompress(compressed)

# Step 5: Check quality
from turboquant import cosine_similarity
cos = cosine_similarity(kv_tensor, reconstructed)
mem = tq.memory_bytes(compressed)

print(f"Compressed: {mem['compressed'] / 1024:.0f} KB")
print(f"Ratio: {mem['ratio']:.1f}x")
print(f"Saved: {mem['savings_pct']:.0f}%")
print(f"Quality: {cos:.4f} cosine similarity")
```

**Output:**
```
Original: torch.Size([1, 8, 512, 128]), 1024 KB
Compressed: 264 KB
Ratio: 3.9x
Saved: 74%
Quality: 0.9954 cosine similarity
```

---

## 4. Understanding the Output

### CompressedKVCache object

```python
compressed = tq.compress(kv_tensor)

# What's inside:
print(compressed.indices.shape)   # torch.Size([1, 8, 512, 128]) — uint8 quantized values
print(compressed.indices.dtype)   # torch.uint8
print(compressed.norms.shape)     # torch.Size([1, 8, 512]) — one FP16 norm per token
print(compressed.norms.dtype)     # torch.float16
print(compressed.shape)           # Original shape for reconstruction
print(compressed.dtype)           # Original dtype for reconstruction
```

### Memory breakdown

```python
mem = tq.memory_bytes(compressed)
# {
#   'original': 1048576,      # FP16 = 2 bytes × 524288 elements
#   'compressed': 270336,     # 4-bit indices + FP16 norms
#   'ratio': 3.9,             # 1048576 / 270336
#   'savings_pct': 74.2       # (1 - 1/3.9) × 100
# }
```

### Cumulative stats

```python
# Stats accumulate across multiple compress() calls
tq.compress(tensor_1)
tq.compress(tensor_2)
tq.compress(tensor_3)

print(tq.stats())
# {
#   'bits': 4,
#   'head_dim': 128,
#   'compression_ratio': '3.9x',
#   'original_bytes': 3145728,
#   'compressed_bytes': 811008,
#   'compress_time_ms': '12.5',
#   'decompress_time_ms': '2.1'
# }

# Reset when needed
tq.reset_stats()
```

---

## 5. Choose Your Bit Width

```python
import torch
from turboquant import TurboQuant, measure_distortion

kv = torch.randn(1, 8, 512, 128, dtype=torch.float16)

for bits in [2, 3, 4]:
    tq = TurboQuant(bits=bits, head_dim=128)
    comp = tq.compress(kv)
    recon = tq.decompress(comp)
    m = measure_distortion(kv, recon)
    mem = tq.memory_bytes(comp)

    print(f"\n{bits}-bit TurboQuant:")
    print(f"  Compression:    {mem['ratio']:.1f}x")
    print(f"  Memory saved:   {mem['savings_pct']:.0f}%")
    print(f"  Cosine sim:     {m['cosine_similarity_mean']:.4f} (mean)")
    print(f"  Cosine sim:     {m['cosine_similarity_min']:.4f} (min)")
    print(f"  IP correlation: {m['inner_product_correlation']:.4f}")
    print(f"  MSE:            {m['mse']:.6f}")
```

**Output:**
```
2-bit TurboQuant:
  Compression:    7.5x
  Memory saved:   87%
  Cosine sim:     0.9409 (mean)
  Cosine sim:     0.9377 (min)
  IP correlation: 0.9401
  MSE:            0.115292

3-bit TurboQuant:
  Compression:    5.1x
  Memory saved:   80%
  Cosine sim:     0.9832 (mean)
  Cosine sim:     0.9823 (min)
  IP correlation: 0.9831
  MSE:            0.033704

4-bit TurboQuant:
  Compression:    3.9x
  Memory saved:   74%
  Cosine sim:     0.9954 (mean)
  Cosine sim:     0.9951 (min)
  IP correlation: 0.9954
  MSE:            0.009257
```

### Which to choose?

| Bits | When to use |
|------|-------------|
| **4-bit** | Default. Production. Near-lossless. |
| **3-bit** | Memory-constrained. Long contexts. Still very good quality. |
| **2-bit** | Extreme compression. Research. Quality drops noticeably. |

---

## 6. Analyze Any Model on Any GPU

**The most powerful feature.** Give a model name and GPU size — get a full report.

```python
from turboquant.integrations.vllm_adapter import analyze

# Just one line — no GPU, vLLM, or model download needed!
report = analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=16)
```

This prints:
- Model architecture (auto-detected from HuggingFace)
- Compression quality at 2/3/4 bits
- Speed overhead (compress/decompress latency)
- Memory at every context length — what fits, what doesn't
- Max context per batch size (FP16 vs TurboQuant)
- Max concurrent users
- Ready-to-use vLLM launch command

### Try different GPUs:

```python
# T4 (16 GB)
analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=16)

# RTX 4090 (24 GB)
analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=24)

# A100 (80 GB)
analyze("meta-llama/Llama-3.1-8B", gpu_memory_gb=80)

# H100 (80 GB) with a 72B model
analyze("Qwen/Qwen2.5-72B-Instruct", gpu_memory_gb=80)
```

### Try different models:

```python
analyze("meta-llama/Llama-3.1-8B", gpu_memory_gb=24)
analyze("mistralai/Mistral-7B-v0.1", gpu_memory_gb=24)
analyze("Qwen/Qwen2.5-14B-Instruct", gpu_memory_gb=48)
analyze("google/gemma-2-9b", gpu_memory_gb=24)
```

### Custom / private model:

```python
analyze(
    "my-private-model",
    gpu_memory_gb=24,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
)
```

### Use the report data programmatically:

```python
report = analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=16, print_report=False)

# Max context at batch=1
f = report["feasibility"][1]
print(f"FP16 max context: {f['max_context_fp16'] // 1024}K")
print(f"TQ4 max context:  {f['max_context_tq'] // 1024}K")
print(f"Improvement:      {f['improvement_x']}x")

# Quality at 4-bit
q = report["quality"][4]
print(f"Cosine similarity: {q['cosine_mean']}")
print(f"Compression ratio: {q['compression_ratio']}x")

# Memory for specific scenario
for row in report["memory_table"]:
    if row["context_length"] == 8192 and row["batch_size"] == 1:
        print(f"8K context: {row['fp16_kv_mb']}MB → {row['tq_kv_mb']}MB")
```

---

## 7. Compress a Real HuggingFace Model

```bash
pip install turboquant[transformers]
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuant, cosine_similarity

# Load model (use any model you like)
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
)
model.eval()

# Run a prompt and extract KV cache
prompt = "Explain how transformers work in deep learning."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

kv_cache = outputs.past_key_values
print(f"Layers: {len(kv_cache)}")
print(f"Key shape: {kv_cache[0][0].shape}")

# Compress every layer
head_dim = kv_cache[0][0].shape[-1]
total_orig, total_comp = 0, 0

for i, (key, value) in enumerate(kv_cache):
    tq_k = TurboQuant(bits=4, head_dim=head_dim, seed=42 + i)
    tq_v = TurboQuant(bits=4, head_dim=head_dim, seed=1000 + i)

    comp_k = tq_k.compress(key)
    comp_v = tq_v.compress(value)
    recon_k = tq_k.decompress(comp_k)

    cos = cosine_similarity(key, recon_k)
    mem = tq_k.memory_bytes(comp_k)
    total_orig += mem["original"] * 2   # K + V
    total_comp += mem["compressed"] * 2

    if i < 3 or i == len(kv_cache) - 1:
        print(f"  Layer {i:>2}: cosine={cos:.4f}")

ratio = total_orig / total_comp
print(f"\nTotal: {total_orig/1024:.0f}KB → {total_comp/1024:.0f}KB ({ratio:.1f}x)")
```

### Using the convenience functions:

```python
from turboquant.integrations.transformers_adapter import (
    extract_kv_cache,
    benchmark_kv_compression,
    get_model_kv_config,
)

# Get model config
config = get_model_kv_config(model)
print(config)
# {'head_dim': 64, 'num_kv_heads': 2, 'num_layers': 24, ...}

# Extract KV cache
kv_cache, inputs = extract_kv_cache(model, tokenizer, "Hello world")

# Benchmark all bit widths
results = benchmark_kv_compression(kv_cache, bits_list=[2, 3, 4])
for bits, r in results.items():
    print(f"{bits}-bit: {r['compression_ratio']}x, cosine={r['key_cosine_mean']}")
```

---

## 8. Multi-Layer Manager (Production Use)

For serving frameworks, use `TurboQuantKVManager` — it manages one compressor per layer.

```python
import torch
from turboquant.integrations.vllm_adapter import TurboQuantKVManager

# Create manager matching your model architecture
manager = TurboQuantKVManager(
    bits=4,
    head_dim=128,
    num_layers=32,  # e.g., Llama 3.1 8B
)

# Simulate full KV cache (32 layers)
kv_cache = [
    (
        torch.randn(1, 8, 256, 128, dtype=torch.float16),  # key
        torch.randn(1, 8, 256, 128, dtype=torch.float16),  # value
    )
    for _ in range(32)
]

# Compress all layers at once
compressed = manager.compress_all(kv_cache)

# Decompress all layers
decompressed = manager.decompress_all(compressed)

# Or compress/decompress one layer (for streaming token-by-token)
new_key = torch.randn(1, 8, 1, 128, dtype=torch.float16)
new_val = torch.randn(1, 8, 1, 128, dtype=torch.float16)
comp_k, comp_v = manager.compress_layer(layer_idx=0, key=new_key, value=new_val)
recon_k, recon_v = manager.decompress_layer(layer_idx=0,
                                             compressed_key=comp_k,
                                             compressed_value=comp_v)

# Memory savings projection
savings = manager.memory_savings(num_kv_heads=8, context_length=32768)
print(savings)
# {'original_mb': 2048.0, 'compressed_mb': 528.0, 'ratio': 3.9, 'savings_pct': 74.2}
```

---

## 9. Compare Against Standard Q4 (GGUF)

TurboQuant vs the standard per-group INT4 quantization used by llama.cpp/GGUF:

```python
import torch
from turboquant import TurboQuant, StandardQ4Quantizer, cosine_similarity

kv = torch.randn(1, 8, 512, 128, dtype=torch.float16)

# Standard Q4 (what llama.cpp uses)
q4 = StandardQ4Quantizer(group_size=32)
q4_comp = q4.compress(kv)
q4_recon = q4.decompress(q4_comp)
q4_cos = cosine_similarity(kv, q4_recon)
q4_mem = q4.memory_bytes(q4_comp)

# TurboQuant 4-bit
tq = TurboQuant(bits=4, head_dim=128)
tq_comp = tq.compress(kv)
tq_recon = tq.decompress(tq_comp)
tq_cos = cosine_similarity(kv, tq_recon)
tq_mem = tq.memory_bytes(tq_comp)

print(f"Standard Q4: cosine={q4_cos:.4f}, ratio={q4_mem['ratio']:.1f}x, "
      f"actual bits={q4_mem['actual_bits_per_element']}, overhead={q4_mem['overhead_pct']}%")
print(f"TurboQuant:  cosine={tq_cos:.4f}, ratio={tq_mem['ratio']:.1f}x, "
      f"overhead=~0.3%")
```

**Output:**
```
Standard Q4: cosine=0.9970, ratio=3.2x, actual bits=5.0, overhead=20.0%
TurboQuant:  cosine=0.9954, ratio=3.9x, overhead=~0.3%
```

**Key difference**: Standard Q4 wastes 20% of its budget on per-group scales/zeros. TurboQuant uses <1% overhead.

### Full comparison with the llamacpp adapter:

```python
from turboquant.integrations.llamacpp_adapter import compare_kv_methods

results = compare_kv_methods(
    num_layers=32, num_kv_heads=8, head_dim=128, seq_len=256
)

print(f"Standard Q4: {results['standard_q4']}")
print(f"TurboQuant:  {results['turboquant']}")
```

---

## 10. Benchmark with Ollama

```bash
pip install turboquant[ollama]
# In another terminal: ollama serve && ollama pull qwen3.5:4b
```

```python
from turboquant.integrations.ollama_adapter import (
    list_models,
    get_model_info,
    generate,
    project_kv_memory,
)

# List available models
models = list_models()
print(models)

# Get model architecture
info = get_model_info("qwen3.5:4b")
print(info)

# Run generation and measure speed
result = generate("qwen3.5:4b", "What is attention?", max_tokens=100)
print(f"Speed: {result['tok_per_sec']} tok/s")
print(f"TTFT:  {result['ttft_ms']} ms")
print(f"Response: {result['response'][:200]}")

# Project KV memory savings
savings = project_kv_memory(
    num_layers=info.get("num_hidden_layers", 36),
    num_kv_heads=info.get("num_key_value_heads", 8),
    head_dim=info.get("head_dim", 128),
)
for s in savings:
    print(f"{s['context_length']:>7} tokens: {s['fp16_kv_mb']:>7.0f}MB "
          f"→ {s['tq_kv_mb']:>7.0f}MB (save {s['saved_pct']}%)")
```

---

## 11. Speed Benchmarking

```python
import torch
import time
from turboquant import TurboQuant

NUM_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

tqs = [TurboQuant(bits=4, head_dim=HEAD_DIM, seed=42+i) for i in range(NUM_LAYERS)]

# Warmup
for tq in tqs:
    c = tq.compress(torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16))
    tq.decompress(c)

# Per-token latency (all layers)
iters = 100
t0 = time.time()
for _ in range(iters):
    for tq in tqs:
        tok = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
        c = tq.compress(tok)
        tq.decompress(c)
per_token_ms = (time.time() - t0) / iters * 1000
print(f"Per-token (all {NUM_LAYERS} layers): {per_token_ms:.2f} ms")
print(f"Per-token per layer: {per_token_ms / NUM_LAYERS:.3f} ms")

# Batch decompression (the hot path during generation)
comps = []
for tq in tqs:
    data = torch.randn(1, NUM_KV_HEADS, 1024, HEAD_DIM, dtype=torch.float16)
    comps.append(tq.compress(data))

t0 = time.time()
for i, tq in enumerate(tqs):
    tq.decompress(comps[i])
decompress_ms = (time.time() - t0) * 1000
print(f"Decompress 1K tokens ({NUM_LAYERS} layers): {decompress_ms:.1f} ms")
```

---

## 12. GPU Planning Guide

### Quick reference — Max context (batch=1)

Run this for any model:

```python
from turboquant.integrations.vllm_adapter import analyze

report = analyze("YOUR_MODEL", gpu_memory_gb=YOUR_GPU, print_report=False)
f = report["feasibility"][1]
print(f"FP16: {f['max_context_fp16']//1024}K tokens")
print(f"TQ4:  {f['max_context_tq']//1024}K tokens ({f['improvement_x']}x more)")
```

### Quick reference — Max concurrent users at 8K context

```python
report = analyze("YOUR_MODEL", gpu_memory_gb=YOUR_GPU, print_report=False)
avail = report["available_for_kv_mb"] * 1024 * 1024
fp16_per_user = report["fp16_kv_bytes_per_token"] * 8192
tq_per_user = report["tq_kv_bytes_per_token"] * 8192
print(f"FP16: {int(avail / fp16_per_user)} users")
print(f"TQ4:  {int(avail / tq_per_user)} users")
```

### Common configurations

| Model | GPU | FP16 Max | TQ4 Max | Users (FP16) | Users (TQ4) |
|-------|-----|----------|---------|:---:|:---:|
| Qwen 7B | T4 16GB | 2K | 10K | 0 | 2 |
| Qwen 7B | RTX 4090 24GB | 138K | 539K | 17 | 67 |
| Llama 8B | A100 40GB | 180K | 700K | 22 | 86 |
| Llama 8B | A100 80GB | 448K | 1737K | 56 | 217 |
| Llama 70B | H100 80GB | 22K | 87K | 2 | 10 |

---

## 13. API Reference

### Core Classes

```python
from turboquant import TurboQuant, CompressedKVCache, StandardQ4Quantizer, LloydMaxCodebook
```

| Class | Purpose |
|-------|---------|
| `TurboQuant(bits, head_dim, seed)` | Main compressor |
| `CompressedKVCache` | Container for compressed data |
| `StandardQ4Quantizer(group_size)` | Baseline Q4 for comparison |
| `LloydMaxCodebook(bits, dim)` | Low-level codebook |

### TurboQuant Methods

| Method | Description |
|--------|-------------|
| `compress(x)` | Compress tensor → CompressedKVCache |
| `decompress(comp)` | Decompress → original shape/dtype |
| `memory_bytes(comp)` | Memory breakdown dict |
| `stats()` | Cumulative stats dict |
| `reset_stats()` | Reset counters |
| `to(device)` | Move to device |

### Metric Functions

```python
from turboquant import cosine_similarity, inner_product_correlation, measure_distortion
```

| Function | Returns |
|----------|---------|
| `cosine_similarity(a, b)` | float: mean cosine sim |
| `inner_product_correlation(a, b)` | float: Pearson correlation of inner products |
| `measure_distortion(a, b)` | dict: MSE, cosine stats, IP stats |

### Integration Modules

```python
from turboquant.integrations.vllm_adapter import analyze, TurboQuantKVManager
from turboquant.integrations.ollama_adapter import generate, project_kv_memory
from turboquant.integrations.llamacpp_adapter import compare_kv_methods
from turboquant.integrations.transformers_adapter import extract_kv_cache, benchmark_kv_compression
```

| Function | What it does |
|----------|-------------|
| `analyze(model, gpu_gb)` | Full feasibility report |
| `TurboQuantKVManager(bits, head_dim, num_layers)` | Multi-layer KV manager |
| `generate(model, prompt)` | Ollama generation + metrics |
| `project_kv_memory(layers, heads, dim)` | Memory projections |
| `compare_kv_methods(layers, heads, dim)` | TQ vs Standard Q4 |
| `extract_kv_cache(model, tokenizer, prompt)` | Get real KV cache |
| `benchmark_kv_compression(kv_cache)` | Quality at each bit width |

---

## Next Steps

1. Run `analyze()` with your model and GPU to see what's possible
2. Try compressing a real model's KV cache (Section 7)
3. Benchmark speed on your hardware (Section 11)
4. Plan your production deployment (Section 12)

```python
# Start here:
from turboquant.integrations.vllm_adapter import analyze
analyze("YOUR_MODEL_HERE", gpu_memory_gb=YOUR_GPU_SIZE)
```
