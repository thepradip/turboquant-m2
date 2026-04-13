# TurboQuant v0.7.0

KV-cache compression for LLM inference on Apple Silicon (MLX).

Compresses KV cache using **Walsh-Hadamard Transform + Lloyd-Max quantization** (4-bit/3-bit). Drop-in `TurboQuantCache` replaces MLX's `KVCache` — compresses on insert, dequantizes incrementally.

Based on: [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)

## Install

```bash
pip install git+https://github.com/thepradip/turboquant-mlx.git
```

From source:
```bash
git clone https://github.com/thepradip/turboquant-mlx.git
cd turboquant-mlx
pip install -e ".[eval]"
```

## Quick Start

```python
from mlx_lm import load
from turboquant import make_turboquant_cache, chunked_prefill

model, tokenizer = load("mlx-community/Qwen3.5-4B-MLX-4bit")

# Create compressed cache (drop-in replacement for make_prompt_cache)
cache = make_turboquant_cache(model, bits=4)

# Prefill
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain gravity in one sentence."}],
    tokenize=False, add_generation_prompt=True,
)
ids = mx.array(tokenizer.encode(text))
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Generate — standard loop, no changes needed
import mlx.core as mx
y = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
for _ in range(200):
    logits = model(y.reshape(1, 1), cache=cache)
    mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    if y.item() == tokenizer.eos_token_id:
        break
    tokens.append(y.item())
print(tokenizer.decode(tokens))
```

That's it. KV cache is compressed automatically. Standard generation loop works unchanged.

## How It Works

```
Input KV vector x (head_dim=128):
  1. Extract norm: γ = ||x||₂
  2. Normalize: x̂ = x / γ
  3. Walsh-Hadamard Transform: y = WHT(x̂)    ← O(d log d) butterfly
     Coordinates now follow Beta distribution
  4. Lloyd-Max quantize: idx = nearest_centroid(y)
     4-bit = 16 centroids, 3-bit = 8 centroids
  5. Store: (uint8 indices, float16 norm) per vector

Dequantize (incremental — only new tokens each step):
  centroids[idx] → inverse WHT → × norm → FP16
```

### Incremental Decode Buffer

The cache dequantizes only **new tokens** each step, not the full sequence:

| Step | Action | Cost |
|------|--------|------|
| Prefill (1000 tokens) | Quantize all → dequantize all once | O(n) |
| Token 1001 | Quantize 1 token → dequantize 1 → append to buffer | O(1) |
| Token 1002 | Quantize 1 token → dequantize 1 → append to buffer | O(1) |

MLX's built-in `scaled_dot_product_attention` handles the attention computation.

## API

### Core

```python
from turboquant import make_turboquant_cache, compress_cache, chunked_prefill

# Create TurboQuant cache (recommended)
cache = make_turboquant_cache(model, bits=4)         # WHT mode (default)
cache = make_turboquant_cache(model, bits=4, use_wht=False)  # Dense rotation

# Or compress an existing cache
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
result = compress_cache(cache, model=model, bits=4, compact=False)
# result: {'cosine': 0.991, 'ratio': 3.9, 'layers_compressed': 8, ...}

# Long prompts (>2K tokens)
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
```

### Eval Suite

```bash
# Quality benchmark — 65 questions, LLM-as-judge
pip install -e ".[eval]"
python3 benchmarks/tq_eval.py --model mlx-community/Qwen3.5-4B-MLX-4bit --judge

# With 3-bit
python3 benchmarks/tq_eval.py --model ... --configs fp16 tq_4bit tq_3bit
```

## Results

Tested on M2 Pro 16GB. Qwen3.5-4B-MLX-4bit, needle-in-haystack:

| Context | FP16 TPS | TQ4 TPS | KV Compression | Quality |
|---------|----------|---------|----------------|---------|
| 2K | 50 | 35 | 1.5x | PASS |
| 4K | 48 | 31 | 1.6x | PASS |
| 8K | 48 | 26 | 1.7x | PASS |
| 16K | 41 | 10 | 1.8x | PASS |

Quality evaluation (54 reliable questions, LLM-as-judge):

| Model | FP16 | TQ 4-bit | TQ 3-bit |
|-------|------|----------|----------|
| Qwen3.5-4B | 87% (9.1/10) | 82% (8.2/10) | 74% (7.9/10) |
| Gemma-4 E4B | 91% (9.1/10) | 80% (8.6/10) | 88% (9.1/10) |
| Qwen3.5-9B | 80% (8.4/10) | 85% (8.9/10) | 77% (8.4/10) |

## Architecture

```
src/turboquant/
├── __init__.py         # Public API
├── hadamard.py         # Walsh-Hadamard Transform (butterfly, O(d log d))
├── compressor.py       # PolarQuant: WHT/rotation + Lloyd-Max codebook
├── codebook.py         # Lloyd-Max centroid computation
├── cache.py            # TurboQuantCache: compress on insert, incremental decode
├── patch.py            # compress_cache, chunked_prefill, generate_step
├── attention.py        # SDPA wrapper
├── results.py          # Experiment save/load
├── qjl.py             # QJL residual (disabled, kept for research)
├── metal_kernel.py     # Metal kernel (used by bonsai_loader)
└── bonsai_loader.py    # 1-bit Bonsai model loader

benchmarks/
├── tq_eval.py                # Unified eval suite
├── tq_eval_report.py         # HTML report generator
└── tq_eval_65_questions.json # Test dataset with reference answers

tests/
└── test_core.py              # 55 tests
```

## Known Limitations

- **Hybrid attention**: Qwen3.5 compresses 8/32 layers (hybrid architecture). Models with full attention on all layers benefit more.
- **Speed vs FP16**: TurboQuantCache runs at 60-70% of FP16 speed due to per-step dequantization overhead.
- **Peak memory during prefill**: Large prefills create intermediate tensors. Use `chunked_prefill` with `chunk_size=2048`.
- **WHT requires power-of-2 head_dim**: Standard for LLMs (128, 256). Falls back to dense rotation otherwise.

## Tests

```bash
pytest tests/test_core.py -v    # 55 tests
```

## License

Apache 2.0

## Author

Pradip Tivhale
