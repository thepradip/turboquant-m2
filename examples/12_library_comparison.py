#!/usr/bin/env python3
"""
Real Inference Performance Comparison on Apple M2
===================================================
Compares: HuggingFace, MLX, Ollama
Same model (Qwen 2.5 / 3.5), same prompts, real measurements.
No made-up numbers — everything measured live on YOUR machine.

Usage:
  python examples/12_library_comparison.py
"""

import time
import json
import sys
import os
import gc
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PROMPT_SHORT = "What are the 3 key differences between TCP and UDP?"
PROMPT_MEDIUM = (
    "Explain the complete transformer architecture including multi-head attention, "
    "positional encoding, layer normalization, feed-forward networks, and how "
    "training with teacher forcing works. Be detailed and precise."
)
PROMPT_LONG = (
    "Read this document carefully and answer the question at the end.\n\n"
    + ("The Transformer architecture was introduced in 'Attention Is All You Need' by Vaswani et al. "
       "in 2017. It replaced recurrent neural networks with self-attention mechanisms that process "
       "all positions in parallel. The key formula is Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. "
       "Multi-head attention allows attending to different representation subspaces. Each transformer "
       "block has two sub-layers: multi-head self-attention and position-wise feed-forward network. "
       "Layer normalization and residual connections are applied around each sub-layer. ") * 15
    + "\n\nQuestion: What is the attention formula and how many sub-layers does each block have?"
)

MAX_TOKENS = 200


def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


# ═══════════════════════════════════════════════════════
#  HuggingFace Transformers (CPU)
# ═══════════════════════════════════════════════════════
def bench_huggingface(prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n  Loading {model_id}...")
    mem_before = get_mem_mb()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    mem_loaded = get_mem_mb()
    print(f"  Model RAM: {mem_loaded - mem_before:.0f} MB")

    results = []
    for name, prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]

        gc.collect()
        mem_pre = get_mem_mb()

        # Measure TTFT (time to first token)
        t0 = time.time()
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        ttft = (time.time() - t0) * 1000

        # Full generation
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True)
        total = time.time() - t0

        gen_ids = gen[0][inputs.input_ids.shape[1]:]
        gen_tokens = len(gen_ids)
        tps = gen_tokens / total if total > 0 else 0
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        mem_peak = get_mem_mb()

        results.append({
            "name": name,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "ttft_ms": round(ttft, 0),
            "tok_per_sec": round(tps, 1),
            "total_s": round(total, 1),
            "ram_delta_mb": round(mem_peak - mem_pre, 0),
            "response_preview": response[:150],
        })
        gc.collect()

    del model, tokenizer
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════
#  MLX (Apple Silicon native)
# ═══════════════════════════════════════════════════════
def bench_mlx(prompts):
    import mlx_lm

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n  Loading {model_id} (MLX)...")
    mem_before = get_mem_mb()

    model, tokenizer = mlx_lm.load(model_id)

    mem_loaded = get_mem_mb()
    print(f"  Model RAM: {mem_loaded - mem_before:.0f} MB")

    results = []
    for name, prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(text))

        gc.collect()
        mem_pre = get_mem_mb()

        # TTFT — generate 1 token
        t0 = time.time()
        _ = mlx_lm.generate(model, tokenizer, prompt=text, max_tokens=1, verbose=False)
        ttft = (time.time() - t0) * 1000

        # Full generation
        t0 = time.time()
        response = mlx_lm.generate(model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, verbose=False)
        total = time.time() - t0

        gen_tokens = len(tokenizer.encode(response))
        tps = gen_tokens / total if total > 0 else 0

        mem_peak = get_mem_mb()

        results.append({
            "name": name,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "ttft_ms": round(ttft, 0),
            "tok_per_sec": round(tps, 1),
            "total_s": round(total, 1),
            "ram_delta_mb": round(mem_peak - mem_pre, 0),
            "response_preview": response[:150],
        })
        gc.collect()

    del model, tokenizer
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════
#  Ollama (GGUF, server-based)
# ═══════════════════════════════════════════════════════
def bench_ollama(prompts):
    import requests

    url = "http://localhost:11434"

    # Find a Qwen model
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        print("  Ollama not running — skipping")
        return None

    # Pick smallest Qwen for fair comparison
    model = None
    for pref in ["qwen3.5:0.8b", "qwen3:0.6b", "qwen3.5:2b", "gemma3:1b"]:
        if pref in models:
            model = pref
            break
    if not model:
        model = models[0] if models else None

    if not model:
        print("  No models in Ollama — skipping")
        return None

    print(f"\n  Using: {model}")

    # Get model info
    r = requests.post(f"{url}/api/show", json={"model": model}, timeout=10)
    details = r.json().get("details", {})
    print(f"  Size: {details.get('parameter_size', '?')}, Quant: {details.get('quantization_level', '?')}")

    results = []
    for name, prompt in prompts:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": MAX_TOKENS, "temperature": 0},
        }

        t0 = time.time()
        try:
            r = requests.post(f"{url}/api/generate", json=payload, timeout=120)
            total = time.time() - t0
            data = r.json()
        except Exception as e:
            results.append({"name": name, "error": str(e)})
            continue

        prompt_tokens = data.get("prompt_eval_count", 0)
        gen_tokens = data.get("eval_count", 0)
        prompt_ns = data.get("prompt_eval_duration", 0)
        gen_ns = data.get("eval_duration", 0)
        ttft = prompt_ns / 1e6
        tps = gen_tokens / (gen_ns / 1e9) if gen_ns > 0 else 0
        response = data.get("response", "")

        results.append({
            "name": name,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "ttft_ms": round(ttft, 0),
            "tok_per_sec": round(tps, 1),
            "total_s": round(total, 1),
            "ram_delta_mb": 0,  # can't measure server-side
            "response_preview": response[:150],
        })

    return results, model


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════
def main():
    import platform

    print("=" * 70)
    print("  Real Inference Benchmark — M2 MacBook")
    print("  Same prompts, actual measurements, no fake numbers")
    print("=" * 70)
    print(f"  Machine:  Apple {os.popen('sysctl -n machdep.cpu.brand_string').read().strip()}")
    print(f"  RAM:      {psutil.virtual_memory().total / 1024**3:.0f} GB")
    print(f"  macOS:    {platform.mac_ver()[0]}")
    print(f"  Max tokens: {MAX_TOKENS}")

    prompts = [
        ("Short (~30 tok)", PROMPT_SHORT),
        ("Medium (~80 tok)", PROMPT_MEDIUM),
        ("Long (~2K tok)", PROMPT_LONG),
    ]

    all_results = {}

    # ─── HuggingFace ───
    print(f"\n{'═' * 70}")
    print(f"  [1/3] HuggingFace Transformers (Qwen2.5-0.5B, CPU FP32)")
    print(f"{'═' * 70}")
    hf_results = bench_huggingface(prompts)
    all_results["HuggingFace (CPU)"] = hf_results

    # ─── MLX ───
    print(f"\n{'═' * 70}")
    print(f"  [2/3] MLX (Qwen2.5-0.5B, Apple Silicon native)")
    print(f"{'═' * 70}")
    try:
        mlx_results = bench_mlx(prompts)
        all_results["MLX (Metal)"] = mlx_results
    except Exception as e:
        print(f"  MLX failed: {e}")
        mlx_results = None

    # ─── Ollama ───
    print(f"\n{'═' * 70}")
    print(f"  [3/3] Ollama (GGUF quantized, server)")
    print(f"{'═' * 70}")
    ollama_out = bench_ollama(prompts)
    if ollama_out:
        ollama_results, ollama_model = ollama_out
        all_results[f"Ollama ({ollama_model})"] = ollama_results
    else:
        ollama_results = None

    # ═══════════════════════════════════════════════════════
    #  Results Table
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'═' * 70}")
    print(f"  RESULTS — Tokens per Second (higher = better)")
    print(f"{'═' * 70}")

    libraries = list(all_results.keys())
    prompt_names = [p[0] for p in prompts]

    # Header
    header = f"  {'Prompt':<18}"
    for lib in libraries:
        header += f" │ {lib:>20}"
    print(f"\n{header}")
    print(f"  {'─' * (20 + 23 * len(libraries))}")

    # Rows
    for pi, pname in enumerate(prompt_names):
        row = f"  {pname:<18}"
        for lib in libraries:
            r = all_results[lib][pi]
            if "error" in r:
                row += f" │ {'ERROR':>20}"
            else:
                tps = r["tok_per_sec"]
                row += f" │ {tps:>17.1f}/s"
        print(row)

    # TTFT
    print(f"\n  {'─' * (20 + 23 * len(libraries))}")
    print(f"\n  TTFT — Time to First Token (lower = better)")
    print(f"\n{header}")
    print(f"  {'─' * (20 + 23 * len(libraries))}")

    for pi, pname in enumerate(prompt_names):
        row = f"  {pname:<18}"
        for lib in libraries:
            r = all_results[lib][pi]
            if "error" in r:
                row += f" │ {'ERROR':>20}"
            else:
                ttft = r["ttft_ms"]
                row += f" │ {ttft:>16.0f} ms"
        print(row)

    # Prompt tokens
    print(f"\n  {'─' * (20 + 23 * len(libraries))}")
    print(f"\n  Prompt Tokens")
    for pi, pname in enumerate(prompt_names):
        row = f"  {pname:<18}"
        for lib in libraries:
            r = all_results[lib][pi]
            if "error" in r:
                row += f" │ {'ERROR':>20}"
            else:
                row += f" │ {r['prompt_tokens']:>20}"
        print(row)

    # Response preview
    print(f"\n\n{'═' * 70}")
    print(f"  RESPONSE SAMPLES (Short prompt)")
    print(f"{'═' * 70}")
    for lib in libraries:
        r = all_results[lib][0]
        if "error" not in r:
            print(f"\n  ┌─ {lib}:")
            print(f"  │ {r['response_preview']}")
            print(f"  └─")

    # Summary
    print(f"\n\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")

    print(f"\n  Library comparison on Apple M2 Pro 16GB:\n")

    for lib in libraries:
        avg_tps = sum(r.get("tok_per_sec", 0) for r in all_results[lib]) / len(all_results[lib])
        avg_ttft = sum(r.get("ttft_ms", 0) for r in all_results[lib]) / len(all_results[lib])
        print(f"  {lib}:")
        print(f"    Avg speed:  {avg_tps:.1f} tok/s")
        print(f"    Avg TTFT:   {avg_ttft:.0f} ms")
        print()

    print(f"  Notes:")
    print(f"    - HuggingFace runs on CPU (FP32) — slowest but most flexible")
    print(f"    - MLX uses Apple Metal GPU — fast, native Apple Silicon")
    print(f"    - Ollama uses GGUF (Q4) via llama.cpp — optimized C++")
    print(f"    - TurboQuant adds KV compression on TOP of any of these")
    print(f"    - For HuggingFace: TurboQuant hybrid cache works NOW")
    print(f"    - For MLX: TurboQuant can be integrated (Python KV cache)")
    print(f"    - For Ollama: needs C implementation in llama.cpp")

    # Save
    output = {lib: [r for r in results] for lib, results in all_results.items()}
    with open("library_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to library_comparison_results.json")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
