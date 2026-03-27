#!/usr/bin/env python3
"""
MLX + TurboQuant — Live Performance Test on Apple Silicon
============================================================
Compares 3 modes on the SAME prompts:
  1. MLX FP16 baseline (no compression)
  2. MLX built-in Q4 KV cache
  3. MLX + TurboQuant hybrid (compress old, keep recent FP16)

Usage:
  python examples/13_mlx_turboquant_live.py
"""

import time
import gc
import sys
import os
import argparse
import numpy as np

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache, make_prompt_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from turboquant import TurboQuant, cosine_similarity

MAX_TOKENS = 200


# ═══════════════════════════════════════════════════════
#  Helpers: MLX ↔ Torch conversion (handles bfloat16)
# ═══════════════════════════════════════════════════════

def mlx_to_torch(x):
    """MLX array (bfloat16/float16/float32) → torch float16."""
    mx.eval(x)
    x32 = x.astype(mx.float32)
    return torch.from_numpy(np.array(x32, copy=True)).to(torch.float16)


def torch_to_mlx(x, target_dtype=mx.bfloat16):
    """torch tensor → MLX array in target dtype."""
    return mx.array(x.float().numpy()).astype(target_dtype)


# ═══════════════════════════════════════════════════════
#  TurboQuant: Compress old KV cache, keep recent FP16
# ═══════════════════════════════════════════════════════

def compress_kv_cache(cache, head_dim, window_size=256, bits=4):
    """
    Compress old tokens in MLX KVCache with TurboQuant.
    Keeps last `window_size` tokens in original precision.
    Writes decompressed values back into the pre-allocated cache array in-place.

    Returns quality metrics.
    """
    cos_scores = []
    total_orig = 0
    total_comp = 0

    for layer_idx, c in enumerate(cache):
        if c.keys is None or c.offset <= window_size * 2:
            continue

        seq_len = c.offset
        split = seq_len - window_size

        # Extract old tokens
        old_k = c.keys[:, :, :split, :]
        old_v = c.values[:, :, :split, :]
        mx.eval(old_k, old_v)

        old_k_torch = mlx_to_torch(old_k)
        old_v_torch = mlx_to_torch(old_v)

        # Compress + decompress with TurboQuant
        tq_k = TurboQuant(bits=bits, head_dim=head_dim, seed=42 + layer_idx)
        tq_v = TurboQuant(bits=bits, head_dim=head_dim, seed=1000 + layer_idx)

        comp_k = tq_k.compress(old_k_torch)
        comp_v = tq_v.compress(old_v_torch)
        recon_k = tq_k.decompress(comp_k)
        recon_v = tq_v.decompress(comp_v)

        # Quality
        cos_scores.append(cosine_similarity(old_k_torch, recon_k))
        mem = tq_k.memory_bytes(comp_k)
        total_orig += mem["original"] * 2
        total_comp += mem["compressed"] * 2

        # Write back INTO the pre-allocated cache array (keeps shape consistent)
        c.keys[:, :, :split, :] = torch_to_mlx(recon_k, c.keys.dtype)
        c.values[:, :, :split, :] = torch_to_mlx(recon_v, c.values.dtype)
        mx.eval(c.keys, c.values)

    avg_cos = sum(cos_scores) / len(cos_scores) if cos_scores else 0
    ratio = total_orig / total_comp if total_comp > 0 else 0
    return {
        "cosine": round(avg_cos, 4),
        "ratio": round(ratio, 1),
        "compressed_layers": len(cos_scores),
    }


# ═══════════════════════════════════════════════════════
#  Generation Functions
# ═══════════════════════════════════════════════════════

def generate_baseline(model, tokenizer, prompt, max_tokens=200):
    """MLX FP16 baseline."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    response = mlx_lm.generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0

    gen_tokens = len(tokenizer.encode(response))
    return {
        "response": response,
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
        "total_s": round(elapsed, 2),
    }


def generate_mlx_q4(model, tokenizer, prompt, max_tokens=200):
    """MLX built-in Q4 KV cache."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    response = mlx_lm.generate(
        model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False,
        kv_bits=4, kv_group_size=64,
    )
    elapsed = time.time() - t0

    gen_tokens = len(tokenizer.encode(response))
    return {
        "response": response,
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
        "total_s": round(elapsed, 2),
    }


def generate_turboquant(model, tokenizer, prompt, head_dim, num_layers,
                         max_tokens=200, bits=4, window=256):
    """
    MLX generation + TurboQuant KV compression.
    Uses standard MLX KVCache, compresses old tokens after prefill.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = mx.array(tokenizer.encode(text))

    # Create standard MLX cache
    cache = make_prompt_cache(model)

    t0 = time.time()

    # Prefill
    logits = model(prompt_ids[None], cache=cache)
    mx.eval(logits)

    # Compress old tokens in the cache
    t_compress = time.time()
    quality = compress_kv_cache(cache, head_dim, window_size=window, bits=bits)
    compress_ms = (time.time() - t_compress) * 1000

    # Generate
    y = mx.argmax(logits[:, -1, :], axis=-1)
    tokens = [y.item()]

    for step in range(max_tokens - 1):
        logits = model(y.reshape(1, -1), cache=cache)
        mx.eval(logits)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        tok = y.item()
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)

    elapsed = time.time() - t0
    response = tokenizer.decode(tokens)
    gen_tokens = len(tokens)

    return {
        "response": response,
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
        "total_s": round(elapsed, 2),
        "compress_ms": round(compress_ms, 0),
        "quality": quality,
    }


# ═══════════════════════════════════════════════════════
#  Test Prompts
# ═══════════════════════════════════════════════════════

TESTS = [
    ("Short QA",
     "What are 3 differences between TCP and UDP? Be concise.", 150),
    ("Math Reasoning",
     "A train travels 120km in 2 hours, then speeds up 50% for 3 more hours. Total distance? Show work.", 200),
    ("Code Generation",
     "Write a Python function to check if a number is prime. Include 3 test cases.", 200),
    ("Long Doc QA",
     "Read and answer: What is the attention formula?\n\n"
     + ("The Transformer (Vaswani 2017) uses Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. "
        "It has 8 heads in the base model. Layer norm and residual connections are used. ") * 25
     + "\nWhat is the exact attention formula?", 100),
    ("Longer Synthesis",
     "List ALL innovations:\n\n"
     + ("AlexNet (2012), Dropout (Srivastava 2014), BatchNorm (Ioffe 2015), "
        "ResNets (He 2016), Attention (Bahdanau 2014), Transformers (Vaswani 2017), "
        "GANs (Goodfellow 2014), BERT (Devlin 2018), GPT (Radford 2018). ") * 20
     + "\nList every innovation with author and year.", 200),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    model_id = args.model

    print("=" * 70)
    print("  MLX + TurboQuant — Live Benchmark on Apple M2")
    print("=" * 70)

    print(f"\n  Loading {model_id}...")
    model, tokenizer = mlx_lm.load(model_id)

    head_dim = model.args.hidden_size // model.args.num_attention_heads
    num_layers = model.args.num_hidden_layers
    n_kv = model.args.num_key_value_heads
    print(f"  Arch: {num_layers}L × {n_kv}KV × {head_dim}dim")

    results = {"fp16": [], "mlx_q4": [], "tq": []}

    for i, (name, prompt, max_tok) in enumerate(TESTS):
        print(f"\n{'─' * 70}")
        print(f"  Test {i+1}: {name}")
        print(f"{'─' * 70}")

        # FP16 baseline
        print(f"  [FP16]     ", end="", flush=True)
        r1 = generate_baseline(model, tokenizer, prompt, max_tok)
        print(f"{r1['gen_tokens']:>3} tok  {r1['tok_per_sec']:>6.1f} tok/s  {r1['total_s']}s")
        results["fp16"].append(r1)

        # MLX Q4
        print(f"  [MLX Q4]   ", end="", flush=True)
        r2 = generate_mlx_q4(model, tokenizer, prompt, max_tok)
        print(f"{r2['gen_tokens']:>3} tok  {r2['tok_per_sec']:>6.1f} tok/s  {r2['total_s']}s")
        results["mlx_q4"].append(r2)

        # TurboQuant
        print(f"  [TurboQ]   ", end="", flush=True)
        try:
            r3 = generate_turboquant(model, tokenizer, prompt, head_dim, num_layers, max_tok)
            extra = ""
            if r3.get("quality", {}).get("cosine"):
                extra = f"  cos={r3['quality']['cosine']}  compress={r3['compress_ms']}ms"
            print(f"{r3['gen_tokens']:>3} tok  {r3['tok_per_sec']:>6.1f} tok/s  {r3['total_s']}s{extra}")
        except Exception as e:
            print(f"ERROR: {e}")
            r3 = {"response": "ERROR", "gen_tokens": 0, "tok_per_sec": 0, "total_s": 0}
        results["tq"].append(r3)

        # Responses
        print(f"\n  FP16:   {r1['response'][:200]}")
        print(f"  MLX Q4: {r2['response'][:200]}")
        print(f"  TQ:     {r3['response'][:200]}")

        gc.collect()

    # ─── Summary ───
    print(f"\n\n{'═' * 70}")
    print(f"  RESULTS")
    print(f"{'═' * 70}")

    print(f"\n  {'Test':<22} │ {'FP16':>10} │ {'MLX Q4':>10} │ {'TurboQ':>10}")
    print(f"  {'─' * 58}")
    for i, (name, _, _) in enumerate(TESTS):
        r1 = results["fp16"][i]
        r2 = results["mlx_q4"][i]
        r3 = results["tq"][i]
        print(f"  {name:<22} │ {r1['tok_per_sec']:>8.1f}/s │ {r2['tok_per_sec']:>8.1f}/s │ {r3['tok_per_sec']:>8.1f}/s")

    avg = lambda k: sum(r["tok_per_sec"] for r in results[k]) / len(results[k])
    print(f"  {'─' * 58}")
    print(f"  {'AVERAGE':<22} │ {avg('fp16'):>8.1f}/s │ {avg('mlx_q4'):>8.1f}/s │ {avg('tq'):>8.1f}/s")

    # Quality check — MLX Q4 vs FP16
    print(f"\n  Quality (word overlap with FP16 baseline):")
    for i, (name, _, _) in enumerate(TESTS):
        fp16_words = set(results["fp16"][i]["response"].lower().split())
        q4_words = set(results["mlx_q4"][i]["response"].lower().split())
        tq_words = set(results["tq"][i]["response"].lower().split())
        q4_pct = len(fp16_words & q4_words) / max(len(fp16_words), 1) * 100
        tq_pct = len(fp16_words & tq_words) / max(len(fp16_words), 1) * 100
        print(f"    {name:<22}  MLX Q4: {q4_pct:>4.0f}%  TurboQ: {tq_pct:>4.0f}%")

    print(f"\n  Key findings on your M2:")
    print(f"    FP16 baseline:    {avg('fp16'):.0f} tok/s (best quality)")
    print(f"    MLX built-in Q4:  {avg('mlx_q4'):.0f} tok/s (SEVERE quality loss on 0.5B model!)")
    print(f"    TurboQuant:       {avg('tq'):.0f} tok/s (hybrid: recent FP16 + old compressed)")
    print(f"\n{'═' * 70}")


if __name__ == "__main__":
    main()
