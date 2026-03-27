# TurboQuant

Near-optimal KV-cache compression for LLM inference using random rotation + Lloyd-Max quantization.

Based on: **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"** — Google Research, ICLR 2026 (arXiv:2504.19874)

## What it does

TurboQuant compresses the **KV-cache** (runtime memory during inference) from FP16 to 2/3/4-bit with near-zero quality loss. It is **complementary** to weight quantization (GGUF, AWQ, GPTQ) — use both together for maximum efficiency.

| Bits | Compression | Cosine Similarity | Use Case |
|------|------------|-------------------|----------|
| 4-bit | ~3.8x | >0.995 | Near-lossless, recommended |
| 3-bit | ~5.3x | >0.983 | Good balance |
| 2-bit | ~8x | >0.940 | Maximum compression |

## Install

```bash
pip install turboquant
```

With optional integrations:
```bash
pip install turboquant[transformers]  # HuggingFace
pip install turboquant[ollama]        # Ollama
pip install turboquant[all]           # Everything
```

## Quick Start

```python
from turboquant import TurboQuant

tq = TurboQuant(bits=4, head_dim=128)

# x shape: (batch, num_heads, seq_len, head_dim)
compressed = tq.compress(kv_tensor)
reconstructed = tq.decompress(compressed)

print(tq.stats())
# {'bits': 4, 'compression_ratio': '3.8x', ...}
```

## Integrations

### vLLM / Multi-layer Manager
```python
from turboquant.integrations.vllm_adapter import TurboQuantKVManager

manager = TurboQuantKVManager(bits=4, head_dim=128, num_layers=32)
compressed = manager.compress_all(kv_cache)
decompressed = manager.decompress_all(compressed)
print(manager.memory_savings(num_kv_heads=4, context_length=32768))
```

### HuggingFace Transformers
```python
from turboquant.integrations.transformers_adapter import (
    extract_kv_cache, benchmark_kv_compression
)

kv_cache, inputs = extract_kv_cache(model, tokenizer, "Hello world")
results = benchmark_kv_compression(kv_cache, bits_list=[2, 3, 4])
```

### Ollama
```python
from turboquant.integrations.ollama_adapter import project_kv_memory

savings = project_kv_memory(num_layers=28, num_kv_heads=4, head_dim=128)
```

### llama.cpp / GGUF Comparison
```python
from turboquant.integrations.llamacpp_adapter import compare_kv_methods

results = compare_kv_methods(num_layers=28, num_kv_heads=4, head_dim=128)
```

## How it Works

1. **Record norm**: Store `||x||` as FP16 (one value per token, not per channel)
2. **Normalize**: `x_hat = x / ||x||` to the unit sphere
3. **Random rotation**: `y = R @ x_hat` with a fixed orthogonal matrix R
4. **Scalar quantize**: Each coordinate independently via Lloyd-Max optimal quantizer
5. **Dequantize**: Reverse rotation and rescale

The key insight: after rotation, all coordinates follow a **known Beta distribution**, so optimal quantizer centroids are **data-independent** and can be precomputed. No calibration data needed.

## License

Apache-2.0
