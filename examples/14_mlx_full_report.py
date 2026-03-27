#!/usr/bin/env python3
"""
TurboQuant + MLX — Full Report Generator
==========================================
Generates a complete comparison report:
  - Q4 model + FP16 KV cache (baseline)
  - Q4 model + TurboQuant KV cache (ours)
  - At context lengths: 1K, 2K, 4K, 8K, 16K, 32K, 64K
  - Measures: speed (tok/s), memory (MB), quality (cosine), TTFT

Usage:
  python examples/14_mlx_full_report.py
  python examples/14_mlx_full_report.py --model mlx-community/Qwen3.5-2B-4bit
  python examples/14_mlx_full_report.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

import time
import json
import gc
import sys
import os
import argparse
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turboquant.mlx_native import TurboQuantMLX, compress_kv_cache_mlx, get_model_config

# ═══════════════════════════════════════════════════════
#  Long Document Builder
# ═══════════════════════════════════════════════════════

DOCUMENT_BLOCK = (
    "The Transformer architecture was introduced by Vaswani et al. in 2017 in "
    "their paper 'Attention Is All You Need'. The core formula is "
    "Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. The model uses multi-head "
    "attention with 8 heads in the base configuration. Each transformer block "
    "contains multi-head self-attention and a position-wise feed-forward network. "
    "Layer normalization and residual connections stabilize training. "
    "BERT (Devlin 2018) introduced masked language modeling for bidirectional "
    "pretraining. GPT (Radford 2018) uses causal attention for autoregressive "
    "generation. ResNets (He 2016) enabled 152-layer networks with skip connections. "
    "AlexNet (Krizhevsky 2012) won ImageNet and started the deep learning revolution. "
    "GANs (Goodfellow 2014) introduced adversarial training for generative models. "
    "Dropout (Srivastava 2014) prevents overfitting. Batch normalization (Ioffe 2015) "
    "stabilizes training of deep networks. Support Vector Machines were dominant before "
    "deep learning. The KV-cache stores key and value vectors during generation and "
    "grows linearly with context length, becoming the primary memory bottleneck. "
)

QUESTION = (
    "Based on the document above, answer ALL of these:\n"
    "1. What is the exact attention formula?\n"
    "2. Who introduced the Transformer and in what year?\n"
    "3. List 5 other innovations mentioned with their authors and years."
)


def build_prompt(tokenizer, target_tokens):
    """Build a document prompt targeting approximately target_tokens."""
    doc = DOCUMENT_BLOCK
    while len(tokenizer.encode(doc)) < target_tokens:
        doc += "\n" + DOCUMENT_BLOCK
    # Trim to target
    words = doc.split()
    while len(tokenizer.encode(" ".join(words))) > target_tokens and len(words) > 100:
        words = words[:int(len(words) * 0.95)]
    doc = " ".join(words)

    prompt = f"Read this document carefully and answer the questions at the end.\n\n{doc}\n\n{QUESTION}"
    return prompt


# ═══════════════════════════════════════════════════════
#  Benchmark Functions
# ═══════════════════════════════════════════════════════

def bench_baseline(model, tokenizer, text, max_tokens=100):
    """Q4 model + FP16 KV cache (what users run today)."""
    mem_before = psutil.Process().memory_info().rss

    t0 = time.time()
    response = mlx_lm.generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0

    mem_after = psutil.Process().memory_info().rss
    gen_tokens = len(tokenizer.encode(response))
    tps = gen_tokens / elapsed if elapsed > 0 else 0

    for tag in ["<think>", "</think>"]:
        response = response.replace(tag, "")

    return {
        "response": response.strip(),
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(tps, 1),
        "total_s": round(elapsed, 2),
        "ram_mb": round((mem_after - mem_before) / 1024 / 1024, 0),
    }


def bench_turboquant(model, tokenizer, text, max_tokens=100, window=512, bits=4):
    """Q4 model + TurboQuant compressed KV cache."""
    ids = mx.array(tokenizer.encode(text))
    prompt_tokens = len(ids)

    mem_before = psutil.Process().memory_info().rss
    cache = make_prompt_cache(model)

    # Prefill
    t_prefill = time.time()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    prefill_ms = (time.time() - t_prefill) * 1000

    # Compress KV cache
    tq_result = compress_kv_cache_mlx(
        cache, model=model, window_size=window, bits=bits, min_context=512
    )

    # Generate
    t_gen = time.time()
    y = mx.argmax(logits[:, -1, :], axis=-1)
    tokens = [y.item()]
    for _ in range(max_tokens - 1):
        logits = model(y.reshape(1, -1), cache=cache)
        mx.eval(logits)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        tok = y.item()
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)
    gen_ms = (time.time() - t_gen) * 1000

    total_s = (time.time() - t_prefill)
    mem_after = psutil.Process().memory_info().rss

    response = tokenizer.decode(tokens)
    for tag in ["<think>", "</think>"]:
        response = response.replace(tag, "")

    gen_tokens = len(tokens)
    gen_tps = gen_tokens / (gen_ms / 1000) if gen_ms > 0 else 0

    return {
        "response": response.strip(),
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(gen_tps, 1),
        "total_s": round(total_s, 2),
        "prefill_ms": round(prefill_ms, 0),
        "compress_ms": tq_result["compress_ms"],
        "gen_ms": round(gen_ms, 0),
        "cosine": tq_result["cosine"],
        "layers_compressed": tq_result["layers_compressed"],
        "kv_original_mb": tq_result.get("original_mb", 0),
        "kv_compressed_mb": tq_result.get("compressed_mb", 0),
        "kv_saved_mb": tq_result.get("saved_mb", 0),
        "kv_ratio": tq_result.get("ratio", 0),
        "ram_mb": round((mem_after - mem_before) / 1024 / 1024, 0),
    }


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TurboQuant MLX Full Report")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit",
                        help="MLX model to benchmark")
    parser.add_argument("--bits", type=int, default=4, help="TurboQuant bits (2,3,4)")
    parser.add_argument("--window", type=int, default=512, help="FP16 window size")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max generation tokens")
    parser.add_argument("--contexts", default="1024,2048,4096,8192,16384",
                        help="Context lengths to test (comma-separated)")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.contexts.split(",")]

    print("=" * 75)
    print("  TurboQuant + MLX — Full Benchmark Report")
    print("  Q4 Weights + FP16 KV  vs  Q4 Weights + TurboQuant KV")
    print("=" * 75)

    # Load model
    print(f"\n  Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)

    config = get_model_config(model)
    model_mb = sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 1024 / 1024
    fp16_kv_per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * 2

    print(f"\n  Model:      {args.model}")
    print(f"  Size:       {model_mb:.0f} MB (4-bit weights)")
    print(f"  Arch:       {config['num_layers']}L × {config['num_kv_heads']}KV × {config['head_dim']}dim")
    print(f"  Heads:      {config['num_attention_heads']} attention, {config['num_kv_heads']} KV (GQA)")
    print(f"  KV/token:   {fp16_kv_per_token} bytes ({fp16_kv_per_token/1024:.1f} KB)")
    print(f"  TurboQuant: {args.bits}-bit, window={args.window}")
    print(f"  System:     Apple {os.popen('sysctl -n machdep.cpu.brand_string').read().strip()}")
    print(f"  RAM:        {psutil.virtual_memory().total / 1024**3:.0f} GB")

    # Run benchmarks
    results = []
    for target in context_lengths:
        print(f"\n{'═' * 75}")
        print(f"  Context: ~{target:,} tokens")
        print(f"{'═' * 75}")

        prompt = build_prompt(tokenizer, target)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        actual_tokens = len(tokenizer.encode(text))
        kv_mb = fp16_kv_per_token * actual_tokens / 1024 / 1024

        print(f"  Actual tokens: {actual_tokens:,}")
        print(f"  FP16 KV size:  {kv_mb:.0f} MB")

        # Baseline
        print(f"\n  [Q4 + FP16 KV] ...", end="", flush=True)
        try:
            bl = bench_baseline(model, tokenizer, text, args.max_tokens)
            print(f" {bl['gen_tokens']} tok  {bl['tok_per_sec']}/s  {bl['total_s']}s")
        except Exception as e:
            print(f" ERROR: {e}")
            bl = None
        gc.collect()

        # TurboQuant
        print(f"  [Q4 + TQ KV]   ...", end="", flush=True)
        try:
            tq = bench_turboquant(model, tokenizer, text, args.max_tokens, args.window, args.bits)
            extra = f"  compress={tq['compress_ms']}ms  cos={tq['cosine']}" if tq["layers_compressed"] > 0 else "  (no compression — below threshold)"
            print(f" {tq['gen_tokens']} tok  {tq['tok_per_sec']}/s  {tq['total_s']}s{extra}")
        except Exception as e:
            print(f" ERROR: {e}")
            tq = None
        gc.collect()

        # Show responses
        if bl and tq:
            print(f"\n  ┌─ Q4 + FP16 KV:")
            print(f"  │ {bl['response'][:300]}")
            print(f"  ├─ Q4 + TurboQuant KV:")
            print(f"  │ {tq['response'][:300]}")
            print(f"  └─")

        tq_kv_mb = kv_mb / 3.9 if tq and tq["layers_compressed"] > 0 else kv_mb
        results.append({
            "target_tokens": target,
            "actual_tokens": actual_tokens,
            "fp16_kv_mb": round(kv_mb, 1),
            "tq_kv_mb": round(tq_kv_mb, 1),
            "saved_mb": round(kv_mb - tq_kv_mb, 1),
            "baseline_tps": bl["tok_per_sec"] if bl else 0,
            "tq_tps": tq["tok_per_sec"] if tq else 0,
            "tq_gen_tps": tq["tok_per_sec"] if tq else 0,
            "compress_ms": tq["compress_ms"] if tq else 0,
            "cosine": tq["cosine"] if tq else 0,
            "compressed": tq["layers_compressed"] > 0 if tq else False,
            "baseline_response": bl["response"][:200] if bl else "",
            "tq_response": tq["response"][:200] if tq else "",
        })

    # ═══════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'═' * 75}")
    print(f"  FULL REPORT — {args.model}")
    print(f"  Q4 Weights ({model_mb:.0f}MB) + FP16 KV vs Q4 Weights + TurboQuant {args.bits}-bit KV")
    print(f"{'═' * 75}")

    # Speed table
    print(f"\n  SPEED (tok/s)")
    print(f"  {'Context':>8} │ {'Tokens':>7} │ {'Q4+FP16':>9} │ {'Q4+TQ':>9} │ {'Diff':>8} │ {'TQ Overhead':>12}")
    print(f"  {'─' * 65}")
    for r in results:
        diff = r["tq_tps"] - r["baseline_tps"]
        pct = (diff / r["baseline_tps"] * 100) if r["baseline_tps"] > 0 else 0
        overhead = f"{r['compress_ms']:.0f}ms" if r["compressed"] else "none"
        print(f"  {r['target_tokens']:>7} │ {r['actual_tokens']:>7} │ {r['baseline_tps']:>7.1f}/s │ "
              f"{r['tq_tps']:>7.1f}/s │ {pct:>+6.0f}%  │ {overhead:>12}")

    # Memory table
    print(f"\n  MEMORY (KV cache)")
    print(f"  {'Context':>8} │ {'FP16 KV':>9} │ {'TQ KV':>9} │ {'Saved':>9} │ {'Total FP16':>11} │ {'Total TQ':>11} │ {'KV > Model?':>12}")
    print(f"  {'─' * 82}")
    for r in results:
        total_fp16 = model_mb + r["fp16_kv_mb"]
        total_tq = model_mb + r["tq_kv_mb"]
        kv_bigger = "YES!" if r["fp16_kv_mb"] > model_mb else "no"
        print(f"  {r['target_tokens']:>7} │ {r['fp16_kv_mb']:>7.0f}MB │ {r['tq_kv_mb']:>7.0f}MB │ "
              f"{r['saved_mb']:>7.0f}MB │ {total_fp16:>9.0f}MB │ {total_tq:>9.0f}MB │ {kv_bigger:>12}")

    # Quality table
    print(f"\n  QUALITY")
    print(f"  {'Context':>8} │ {'Cosine':>8} │ {'Compressed?':>12} │ {'Output Match':>13}")
    print(f"  {'─' * 50}")
    for r in results:
        if r["baseline_response"] and r["tq_response"]:
            bl_words = set(r["baseline_response"].lower().split())
            tq_words = set(r["tq_response"].lower().split())
            overlap = len(bl_words & tq_words) / max(len(bl_words), 1) * 100
            match = f"{overlap:.0f}% overlap"
        else:
            match = "N/A"
        cos_str = f"{r['cosine']:.4f}" if r["compressed"] else "N/A"
        comp_str = "YES" if r["compressed"] else "no (< threshold)"
        print(f"  {r['target_tokens']:>7} │ {cos_str:>8} │ {comp_str:>12} │ {match:>13}")

    # Recommendation
    print(f"\n  RECOMMENDATION")
    print(f"  {'─' * 65}")
    for r in results:
        ctx = r["target_tokens"]
        if not r["compressed"]:
            print(f"  {ctx:>7} tokens: Skip TurboQuant — KV cache too small ({r['fp16_kv_mb']:.0f}MB)")
        elif r["saved_mb"] < 50:
            print(f"  {ctx:>7} tokens: Optional — saves {r['saved_mb']:.0f}MB")
        elif r["fp16_kv_mb"] > model_mb:
            print(f"  {ctx:>7} tokens: ESSENTIAL — KV ({r['fp16_kv_mb']:.0f}MB) > model ({model_mb:.0f}MB), saves {r['saved_mb']:.0f}MB")
        else:
            print(f"  {ctx:>7} tokens: RECOMMENDED — saves {r['saved_mb']:.0f}MB ({r['saved_mb']/r['fp16_kv_mb']*100:.0f}% of KV)")

    print(f"\n{'═' * 75}")

    # Save JSON
    output = {
        "model": args.model,
        "model_mb": round(model_mb),
        "config": config,
        "bits": args.bits,
        "window": args.window,
        "system": f"Apple {os.popen('sysctl -n machdep.cpu.brand_string').read().strip()}",
        "ram_gb": round(psutil.virtual_memory().total / 1024**3),
        "results": results,
    }
    out_path = "turboquant_mlx_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Report saved to {out_path}")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    main()
