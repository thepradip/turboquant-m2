# TurboQuant — Quick Start Guide

## Step 1: Install

```bash
# Basic install (works standalone — no other LLM library needed)
pip install turboquant

# From TestPyPI (current)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ turboquant
```

**That's it.** TurboQuant only needs `torch`, `numpy`, `scipy` — all installed automatically.

### Optional extras (only if you use these frameworks)

```bash
pip install turboquant[transformers]  # If you use HuggingFace models
pip install turboquant[ollama]        # If you use Ollama
pip install turboquant[all]           # Everything
```

---

## Step 2: Check Performance (Zero Setup — No Other Library Needed)

### Option A: One-liner — Analyze any model

```python
from turboquant.integrations.vllm_adapter import analyze

# Just give a model name + your GPU size — get full report
analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=16)   # T4
analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=24)   # RTX 3090/4090
analyze("meta-llama/Llama-3.1-8B", gpu_memory_gb=80)     # A100
analyze("mistralai/Mistral-7B-v0.1", gpu_memory_gb=48)   # A6000
```

This prints a full report: quality, speed, memory savings, max context, max users.

**No vLLM/Ollama/GPU needed** — it simulates the KV cache and gives projections.

### Option B: Hands-on — Compress and measure yourself

```python
import torch
from turboquant import TurboQuant, cosine_similarity

# Create compressor
tq = TurboQuant(bits=4, head_dim=128)

# Simulate KV cache: (batch, kv_heads, seq_len, head_dim)
kv = torch.randn(1, 8, 1024, 128, dtype=torch.float16)

# Compress
compressed = tq.compress(kv)

# Decompress
reconstructed = tq.decompress(compressed)

# Check quality
print(f"Cosine similarity: {cosine_similarity(kv, reconstructed):.4f}")
print(f"Memory: {tq.memory_bytes(compressed)}")
print(f"Stats: {tq.stats()}")
```

### Option C: Compare all bit widths

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
    print(f"{bits}-bit: {mem['ratio']:.1f}x compression, "
          f"cosine={m['cosine_similarity_mean']:.4f}, "
          f"saved {mem['savings_pct']:.0f}%")
```

Output:
```
2-bit: 7.5x compression, cosine=0.9409, saved 87%
3-bit: 5.1x compression, cosine=0.9832, saved 80%
4-bit: 3.9x compression, cosine=0.9954, saved 74%
```

### Option D: Full report for your specific GPU

```python
from turboquant.integrations.vllm_adapter import analyze

# T4 16GB
report = analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=16)

# Access the data programmatically
print(report["feasibility"])        # Max context per batch size
print(report["quality"])            # Compression quality at each bit width
print(report["memory_table"])       # Full memory breakdown
print(report["speed"])              # Compress/decompress latency
```

---

## Do I Need Any Other Library?

| What you want to do | Extra library needed? |
|---------------------|----------------------|
| **Compress/decompress KV cache** | NO — just `pip install turboquant` |
| **Check performance & quality** | NO — works standalone |
| **Analyze any model (full report)** | NO — auto-detects config from model name |
| **Run on actual HuggingFace model** | YES — `pip install transformers` |
| **Benchmark against Ollama** | YES — `pip install requests` + Ollama running |
| **Use inside vLLM serving** | YES — `pip install vllm` |

**Bottom line: `pip install turboquant` is all you need to test and evaluate.**

---

## Step 3: Run on a Real Model (Optional)

### With HuggingFace Transformers

```bash
pip install turboquant[transformers]
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuant, cosine_similarity

# Load any model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
)
model.eval()

# Extract real KV cache
prompt = "Explain transformers in deep learning."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

kv_cache = outputs.past_key_values  # Real KV cache from model

# Compress each layer
head_dim = kv_cache[0][0].shape[-1]
for i, (key, value) in enumerate(kv_cache):
    tq = TurboQuant(bits=4, head_dim=head_dim, seed=42 + i)
    comp = tq.compress(key)
    recon = tq.decompress(comp)
    cos = cosine_similarity(key, recon)
    if i < 3:
        print(f"Layer {i}: cosine = {cos:.4f}")

print(f"... {len(kv_cache)} layers compressed successfully")
```

### With Ollama

```bash
pip install turboquant[ollama]
ollama serve  # in another terminal
ollama pull qwen3.5:4b
```

```python
from turboquant.integrations.ollama_adapter import generate, project_kv_memory

# Run inference
result = generate("qwen3.5:4b", "What is attention?", max_tokens=100)
print(f"Speed: {result['tok_per_sec']} tok/s")

# Project savings
savings = project_kv_memory(num_layers=36, num_kv_heads=8, head_dim=128)
for s in savings:
    print(f"  {s['context_length']:>6} tokens: "
          f"{s['fp16_kv_mb']:.0f}MB → {s['tq_kv_mb']:.0f}MB "
          f"(save {s['saved_pct']}%)")
```

---

## Quick Reference

```python
# Install
pip install turboquant

# 3 lines to compress
from turboquant import TurboQuant
tq = TurboQuant(bits=4, head_dim=128)
compressed = tq.compress(kv_tensor)       # compress
reconstructed = tq.decompress(compressed)  # decompress

# Full GPU analysis — one line
from turboquant.integrations.vllm_adapter import analyze
analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=16)
```
