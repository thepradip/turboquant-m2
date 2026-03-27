# TurboQuant

KV-cache compression for LLM inference on Apple Silicon using MLX.

Based on: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Google Research, ICLR 2026

## Install

```bash
pip install mlx mlx-lm scipy
pip install git+https://github.com/thepradip/turboquant-m2.git
```

## Usage

```python
import mlx_lm
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from turboquant import compress_kv_cache_mlx

# Load any HuggingFace model via MLX
model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-2B-4bit")

# Tokenize and prefill
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False, add_generation_prompt=True
)
ids = mx.array(tokenizer.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Compress KV cache — auto-detects model config
result = compress_kv_cache_mlx(cache, model=model)
print(result)
# {'cosine': 0.9953, 'compress_ms': 1620, 'layers_compressed': 6,
#  'original_mb': 24.3, 'compressed_mb': 6.2, 'saved_mb': 18.1, 'ratio': 3.9}

# Continue generation with compressed cache
y = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
for _ in range(200):
    logits = model(y.reshape(1, -1), cache=cache)
    mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    if y.item() == tokenizer.eos_token_id:
        break
    tokens.append(y.item())

print(tokenizer.decode(tokens))
```

## Low-level API

```python
from turboquant import TurboQuantMLX

tq = TurboQuantMLX(bits=4, head_dim=128)
compressed = tq.compress(kv_tensor)   # MLX array in
reconstructed = tq.decompress(compressed)  # MLX array out
print(tq.memory_bytes(compressed))
```

## Auto-detect model config

```python
from turboquant import get_model_config

config = get_model_config(model)
# {'head_dim': 256, 'num_layers': 24, 'num_kv_heads': 2, ...}
```

## Tested models

| Model | head_dim | Layers compressed | Cosine | Status |
|-------|:---:|:---:|:---:|:---:|
| mlx-community/Qwen3.5-2B-4bit | 256 | 6/24 (full attn only) | 0.9953 | Works |
| Qwen/Qwen2.5-1.5B-Instruct | 128 | 28/28 | 0.9959 | Compression works, generation degrades |
| Qwen/Qwen2.5-0.5B-Instruct | 64 | 24/24 | 0.9955 | Not recommended (head_dim too small) |

## Known limitations

- **Generation quality**: Compressing all layers degrades output on standard models (Qwen2.5, Llama). Works best on hybrid models (Qwen3.5) where only full-attention layers are compressed.
- **Apple Silicon only**: Uses MLX. Does not work on CUDA/CPU-only machines.
- **Metal buffer limit**: Contexts above ~24K tokens may hit Metal's 8GB single-allocation limit.
- **One-time overhead**: Codebook initialization takes 1-5s per request depending on context length.

## How it works

1. Record vector norm
2. Normalize to unit sphere
3. Apply random orthogonal rotation (fixed per layer)
4. After rotation, coordinates follow known Beta distribution
5. Apply Lloyd-Max optimal scalar quantizer (precomputed centroids)
6. Store: 4-bit indices + FP16 norm per token
7. Dequantize: lookup centroids, inverse rotation, rescale by norm
