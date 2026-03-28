# TurboQuant

KV-cache compression for LLM inference on Apple Silicon (MLX).

Directly accesses and compresses the KV cache at code level — reads real key/value tensors, compresses with 4-bit Lloyd-Max quantization, writes back. Model continues generating from compressed cache.

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

model, tok = mlx_lm.load("Qwen/Qwen3.5-2B")

text = tok.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False, add_generation_prompt=True
)
ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Compress KV cache — one line
result = compress_cache(cache, model=model, bits=4)

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

### Saving experiment results

```python
from turboquant import save_experiment, list_experiments, load_experiment

result = compress_cache(cache, model=model, bits=4)
save_experiment(
    model_name="Qwen3.5-2B",
    compress_result=result,
    context_tokens=8000,
    gen_tps=33.7,
    ttft_ms=2795,
    passed=True,
)

# Review past experiments
for e in list_experiments():
    print(e["filename"], e["model_name"], e["cosine"])

# Load specific result
data = load_experiment("qwen3.5-2b_8000tok_20260328_120000.json")
```

Results are saved to `results/` as JSON with timestamp, hardware info, and all metrics.

## Test Results

All results below come from local JSON files produced by running TurboQuant on real models via MLX. Each claim references its source file. Results are not committed to git — run experiments locally with `save_experiment()` to reproduce.

### KV Memory and Generation Speed

**Source: `turboquant_mlx_report.json`**
Model: `mlx-community/Qwen3.5-2B-4bit` (1010 MB), Apple M2 Pro 16GB.
Config: head_dim=256, num_layers=24, num_kv_heads=2.
Cosine computed per layer using real key vectors (v0.4.3+ code, commit `0327d3d`).

| Tokens | FP16 KV | TQ KV | Saved | Compress ms | Cosine | Baseline tps | TQ tps |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1,038 | 48.7 MB | 12.5 MB | 36.2 MB | 1,297 | 0.9953 | 44.5 | 97.9 |
| 2,101 | 98.5 MB | 25.3 MB | 73.2 MB | 1,517 | 0.9953 | 27.8 | 92.1 |
| 4,117 | 193.0 MB | 49.5 MB | 143.5 MB | 1,971 | 0.9953 | 15.4 | 57.1 |
| 7,896 | 370.1 MB | 94.9 MB | 275.2 MB | 2,795 | 0.9954 | 8.7 | 33.7 |
| 15,750 | 738.3 MB | 189.3 MB | 549.0 MB | 4,787 | 0.9953 | 4.4 | 11.2 |

Baseline and TQ responses were captured side-by-side in the same file. Both produce correct, coherent answers to the same questions (attention formula, transformer authors, etc.).

### Multi-Model Benchmark

**Source: `experiment_report.json`**
4 models, 4 context lengths each, 16 runs total. All on Apple M2 Pro 16GB via MLX.
Cosine computed per layer (commit `0327d3d`, real `mx.mean(dot / (nk * nr))` computation).

**Qwen3.5-4B-MLX-4bit** (8/32 layers compressible):

| Context | FP16 KV | Saved | Compress ms | Baseline tps | TQ tps | Pass |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2,033 | 254.1 MB | 47.4 MB | 2,678 | 12.1 | 51.3 | yes |
| 4,268 | 533.5 MB | 99.5 MB | 3,911 | 6.5 | 30.7 | yes |
| 8,912 | 1,114.0 MB | 207.8 MB | 7,136 | 3.2 | 10.7 | yes |
| 13,233 | 1,654.1 MB | 308.5 MB | 11,455 | 2.2 | 3.7 | yes |

**Gemma3-4B-4bit** (34/34 layers compressible):

| Context | FP16 KV | Saved | Compress ms | Baseline tps | TQ tps | Pass |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2,083 | 345.8 MB | 206.4 MB | 11,269 | 12.9 | 52.0 | yes |
| 4,347 | 721.7 MB | 430.7 MB | 16,403 | 7.1 | 45.7 | yes |
| 8,997 | 1,493.6 MB | 891.5 MB | 31,367 | 3.6 | 7.4 | yes |
| 13,365 | 2,218.8 MB | 1,324.3 MB | 44,998 | 2.5 | 5.6 | yes |

**Qwen3.5-2B-OptiQ-4bit** (6/24 layers compressible):

| Context | FP16 KV | Saved | Compress ms | Baseline tps | TQ tps | Pass |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2,035 | 95.4 MB | 17.8 MB | 1,601 | 26.2 | 76.5 | yes |
| 4,270 | 200.2 MB | 37.3 MB | 2,030 | 15.5 | 74.8 | yes |
| 8,914 | 417.8 MB | 77.9 MB | 3,125 | 8.1 | 32.8 | yes |
| 13,235 | 620.4 MB | 115.7 MB | 4,786 | 5.5 | 12.2 | yes |

**Qwen/Qwen3.5-2B** (6/24 layers compressible):

| Context | FP16 KV | Saved | Compress ms | Baseline tps | TQ tps | Pass |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2,035 | 95.4 MB | 17.8 MB | 1,617 | 24.4 | 32.8 | yes |
| 4,270 | 200.2 MB | 37.3 MB | 1,998 | 17.1 | 35.2 | yes |
| 8,914 | 417.8 MB | 77.9 MB | 3,043 | 9.9 | 10.3 | yes |
| 13,235 | 620.4 MB | 115.7 MB | 5,513 | 7.3 | 3.5 | yes |

Note: `experiment_report.json` contains fields (`bl_tps`, `ttft_ms`, `gen_ms`, `tq_gen_tps`, `tq_total_s`, `pass`) that are not returned by `compress_cache()`. These were produced by a separate experiment script that is not in this repository.

### Long Context (chunked prefill)

`chunked_prefill()` processes prompts in 2048-token chunks, keeping the attention matrix small per chunk instead of allocating a single (seq x seq) buffer.

```python
from turboquant import compress_cache, chunked_prefill
from mlx_lm.models.cache import make_prompt_cache

cache = make_prompt_cache(model)
ids = mx.array(tok.encode(text))
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
mx.eval(logits)

compress_cache(cache, model=model, bits=4)
```

**No max-context experiment data is saved.** The README previously claimed 86K tokens for Qwen3.5-2B-OptiQ-4bit, but no JSON result file exists to back this. This needs to be re-tested with `save_experiment()`.

## How It Works

```
1. Model does full prefill -> KV cache filled with FP16 values
2. TurboQuant reads cache[layer].keys and cache[layer].values
3. For each layer:
   a. Normalize vectors to unit sphere (store norms)
   b. Apply random orthogonal rotation
   c. Quantize each coordinate with Lloyd-Max codebook (4-bit)
   d. Dequantize -> inverse rotate -> rescale by norms
   e. Write compressed values back into cache
4. Model generates next tokens from compressed cache
```

## Architecture

```
src/turboquant/
├── __init__.py       # Public API exports
├── patch.py          # compress_cache(), chunked_prefill() — main API
├── compressor.py     # PolarQuant: rotation + Lloyd-Max quantization
├── codebook.py       # Lloyd-Max codebook builder
├── results.py        # save_experiment(), list_experiments(), load_experiment()
├── metal_kernel.py   # Fused Metal kernel (validated)
├── qjl.py           # QJL residual correction (code present, disabled)
├── attention.py      # Custom attention function
├── cache.py          # TurboQuant cache structure
├── torch_backend.py  # PyTorch backend (CPU/CUDA/MPS)
└── mlx_native.py     # Pure MLX implementation
```

## Known Issues

- **Uncommitted code has hardcoded cosine**: Working copy `patch.py:203` returns `0.9953` constant instead of computing real cosine similarity. The committed v0.5.0 code computes it correctly. This needs to be fixed before next release.
- **Uncommitted code has dead memory tracking**: Lines 206-208 in working copy are unreachable after `break` on line 204. `saved_mb`, `original_mb`, `compressed_mb`, `ratio` all return 0.
- **Hybrid attention models**: Qwen3.5 models use hybrid attention — only 6/24 or 8/32 layers have standard KV cache. TurboQuant only compresses those layers, resulting in ~18% KV savings vs ~60% for fully-compressible models like Gemma3.
- **QJL disabled**: Stage 2 of TurboQuant paper (QJL residual correction) is written but disabled. Added variance instead of reducing bias in testing.
- **Apple Silicon only**: Uses MLX. Does not run on CUDA.
- **Experiment script missing**: The script that produced `experiment_report.json` (with baseline tps, ttft, generation metrics) is not in the repository. Results cannot be reproduced without it.
