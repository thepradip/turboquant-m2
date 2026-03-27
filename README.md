# TurboQuant

KV-cache compression for LLM inference on Apple Silicon (MLX).

Based on: [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)

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
from turboquant import compress_cache

# Load any HuggingFace model
model, tok = mlx_lm.load("Qwen/Qwen2.5-1.5B-Instruct")

# Prefill
text = tok.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False, add_generation_prompt=True
)
ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Compress KV cache — one line, auto-detects everything
result = compress_cache(cache, model=model, bits=4)
print(result)

# Generate with compressed cache
y = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
for _ in range(200):
    logits = model(y.reshape(1, -1), cache=cache)
    mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    if y.item() == tok.eos_token_id:
        break
    tokens.append(y.item())
print(tok.decode(tokens))
```

## Status

Tested on M2 Pro 16GB with Qwen2.5-1.5B-Instruct (28 layers, head_dim=128):

| Test | FP16 baseline | TurboQuant 4-bit | Result |
|------|:---:|:---:|:---:|
| "What is 2+2?" | "4" | "4" | PASS |
| Attention formula | Correct | Correct (minor grammar) | PASS |
| TCP vs UDP | 3 correct points | 3 correct points | PASS |
| Code generation | Correct function | Starts OK, then degrades | PARTIAL |

**What works**: Factual QA, reasoning, short answers, topic identification.

**What doesn't work reliably**: Long code generation, complex multi-step outputs. The MSE quantization introduces bias that accumulates over long generation.

**Next step**: QJL residual correction (Stage 2 of TurboQuant paper) to remove inner product bias. Code is written (`qjl.py`, `attention.py`, `cache.py`) but attention integration needs debugging.

## Architecture

```
src/turboquant/
├── __init__.py       # compress_cache, get_model_config
├── patch.py          # compress_cache() — compress KV in-place
├── compressor.py     # Stage 1: PolarQuant (rotation + Lloyd-Max)
├── codebook.py       # Lloyd-Max codebook builder
├── qjl.py           # Stage 2: QJL residual correction (written, not integrated)
├── attention.py      # Custom attention with QJL (written, not integrated)
└── cache.py          # TurboQuant cache structure (written, not integrated)
```
