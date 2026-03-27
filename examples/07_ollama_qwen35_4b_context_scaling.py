#!/usr/bin/env python3
"""
TurboQuant + Ollama: Context Scaling Benchmark (Any Model)
============================================================
Tests any Ollama model at context lengths 1K → max feasible,
measures throughput, TTFT, and projects TurboQuant 4-bit KV savings.
Auto-scales context range based on model size, architecture, and RAM.

Works with: Qwen 3.5 2B/4B, Llama 3 8B, Gemma 3 4B, etc.

Usage:
  1. ollama serve
  2. ollama pull <model>    # e.g. qwen3.5:4b, llama3:8b, gemma3:4b
  3. pip install turboquant[ollama]
  4. python 07_ollama_qwen35_4b_context_scaling.py                # auto-detect
  5. python 07_ollama_qwen35_4b_context_scaling.py --model llama3:8b
  6. python 07_ollama_qwen35_4b_context_scaling.py --ram 32768    # 32GB Mac
"""

import requests
import time
import json
import sys
import argparse
import torch

from turboquant import TurboQuant, cosine_similarity, measure_distortion
from turboquant.integrations.ollama_adapter import project_kv_memory

# ═══════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434"
DEFAULT_RAM_MB = 16384
OS_RESERVED_MB = 4000  # macOS + apps

# Ollama model_info uses different key names than standard HF config.
# Keys come in as "{arch}.block_count", "{arch}.attention.head_count", etc.
# After splitting on ".", the last segment is mapped here.
OLLAMA_KEY_MAP = {
    # Ollama key (last segment)  →  canonical name
    "block_count":      "num_hidden_layers",
    "embedding_length": "hidden_size",
    "head_count":       "num_attention_heads",
    "head_count_kv":    "num_key_value_heads",
    "key_length":       "head_dim",           # some models expose this
    "value_length":     "head_dim",           # fallback
    "context_length":   "context_length",
    "full_attention_interval": "full_attention_interval",  # hybrid models
    # Some models already use HF-style names
    "num_hidden_layers":     "num_hidden_layers",
    "hidden_size":           "hidden_size",
    "num_attention_heads":   "num_attention_heads",
    "num_key_value_heads":   "num_key_value_heads",
    "head_dim":              "head_dim",
}

# Families known to use thinking mode by default
THINKING_FAMILIES = {"qwen3", "qwen35", "qwen3.5"}


# ═══════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════
def list_ollama_models():
    """Get list of models from Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        print("  Ollama is not running!")
        print("  Start it with: ollama serve")
        print("  Then pull a model: ollama pull qwen3.5:4b")
        sys.exit(1)


def pick_model(models):
    """Auto-detect the best model from available ones."""
    if not models:
        print("  No models found in Ollama!")
        print("  Run: ollama pull qwen3.5:4b")
        sys.exit(1)

    # Prefer smaller models for faster benchmarking
    preferences = [
        "qwen3.5:4b", "qwen3.5:2b", "gemma3:4b", "llama3:8b",
        "qwen3:4b", "qwen2.5:3b", "phi3:mini",
    ]
    for pref in preferences:
        if pref in models:
            return pref

    # Fallback: first available
    return models[0]


def get_model_config(model):
    """Get architecture details from Ollama, mapped to canonical names."""
    r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=10)
    r.raise_for_status()
    data = r.json()

    details = data.get("details", {})
    model_info = data.get("model_info", {})

    config = {
        "family": details.get("family", "unknown"),
        "parameter_size": details.get("parameter_size", "unknown"),
        "quantization": details.get("quantization_level", "unknown"),
    }

    # Map Ollama's model_info keys to canonical names
    for raw_key, value in model_info.items():
        # Skip None values (e.g. head_count_kv=None on hybrid models)
        if value is None:
            continue
        # Split "qwen2.attention.head_count" → take last segment "head_count"
        segments = raw_key.split(".")
        short = segments[-1]
        canonical = OLLAMA_KEY_MAP.get(short)
        if canonical and canonical not in config:
            config[canonical] = value

    # Derive head_dim if not directly provided
    if "head_dim" not in config:
        hs = config.get("hidden_size")
        nh = config.get("num_attention_heads")
        if hs and nh and nh > 0:
            config["head_dim"] = hs // nh

    # If num_key_value_heads is missing, assume MHA (all heads are KV heads)
    if "num_key_value_heads" not in config and "num_attention_heads" in config:
        config["num_key_value_heads"] = config["num_attention_heads"]

    return config, model_info


def validate_config(config, model):
    """Ensure we have all required architecture params. Fail clearly if not."""
    required = ["num_hidden_layers", "hidden_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim"]
    missing = [k for k in required if k not in config]
    if missing:
        print(f"\n  ERROR: Could not determine architecture for '{model}'")
        print(f"  Missing: {', '.join(missing)}")
        print(f"  Available config: {json.dumps(config, indent=2, default=str)}")
        print(f"\n  This model may use non-standard keys in Ollama's API.")
        print(f"  Please report this or specify architecture manually.")
        sys.exit(1)


def is_thinking_model(config):
    """Check if this model family uses thinking mode by default."""
    family = str(config.get("family", "")).lower().replace(" ", "")
    return family in THINKING_FAMILIES


FILLER_BLOCK = (
    "The transformer architecture revolutionized natural language processing by introducing "
    "self-attention mechanisms that process all positions in parallel. Unlike RNNs which process "
    "tokens sequentially, transformers compute attention scores between all token pairs simultaneously. "
    "This enables much faster training on modern GPUs and better capture of long-range dependencies. "
    "The multi-head attention mechanism allows the model to attend to information from different "
    "representation subspaces at different positions. Each head learns different aspects of the "
    "relationships between tokens. The feed-forward network in each layer applies two linear "
    "transformations with a ReLU activation in between. Layer normalization and residual connections "
    "help with training stability and gradient flow through deep networks. "
)

# Calibrated at runtime via count_tokens(); falls back to this estimate
_tokens_per_block = 120


def count_tokens(model, text):
    """Count exact tokens using Ollama's tokenize API."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/tokenize",
            json={"model": model, "text": text},
            timeout=10,
        )
        r.raise_for_status()
        tokens = r.json().get("tokens", [])
        return len(tokens)
    except Exception:
        return None


def calibrate_tokens_per_block(model):
    """Measure actual tokens per filler block for this model's tokenizer."""
    global _tokens_per_block
    n = count_tokens(model, FILLER_BLOCK)
    if n and n > 0:
        _tokens_per_block = n
        print(f"  Tokenizer calibrated: {n} tokens per filler block")
    else:
        print(f"  Tokenizer API unavailable, estimating ~{_tokens_per_block} tok/block")


def make_prompt(target_tokens):
    """Create a prompt approximately target_tokens long (calibrated)."""
    base = "Analyze the following passage in great detail, providing insights on the key themes. "
    suffix = "\nSummarize the key points above."
    # Reserve ~30 tokens for base + suffix
    repeats = max(1, (target_tokens - 30) // _tokens_per_block)
    return base + (FILLER_BLOCK * repeats) + suffix


def compute_context_lengths(fp16_per_token, weight_mb, total_ram_mb, model_ctx_length):
    """
    Auto-compute context lengths to test based on what actually fits.

    Strategy: powers of 2 from 1K up to max feasible context, capped by
    model's declared context_length and available RAM.
    """
    available_mb = total_ram_mb - OS_RESERVED_MB - weight_mb
    # Max context where FP16 KV fits in available RAM
    max_ctx_ram = int(available_mb * 1024 * 1024 / fp16_per_token) if fp16_per_token > 0 else 0

    # Also cap at model's declared context length
    if isinstance(model_ctx_length, int) and model_ctx_length > 0:
        max_ctx = min(max_ctx_ram, model_ctx_length)
    else:
        max_ctx = max_ctx_ram

    # Build power-of-2 series: 1K, 2K, 4K, 8K, ... up to max
    lengths = []
    ctx = 1024
    while ctx <= max_ctx:
        lengths.append(ctx)
        ctx *= 2

    # Always include max if it's meaningfully beyond the last power-of-2
    if lengths and max_ctx > lengths[-1] * 1.3:
        lengths.append(max_ctx)

    if not lengths:
        lengths = [1024]  # absolute minimum

    return lengths


def run_generation(model, prompt, max_tokens=100, num_ctx=65536, disable_thinking=False):
    """Run generation with metrics."""
    options = {
        "num_predict": max_tokens,
        "temperature": 0,
        "num_ctx": num_ctx,
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    # Disable thinking mode to get actual visible responses
    if disable_thinking:
        payload["think"] = False

    # Scale timeout with context size (large contexts can be slow)
    timeout = max(600, num_ctx // 50)

    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        elapsed = time.time() - t0
        data = r.json()
    except Exception as e:
        return {"error": str(e), "total_time_s": time.time() - t0}

    if "error" in data:
        return {"error": data["error"], "total_time_s": elapsed}

    prompt_tokens = data.get("prompt_eval_count", 0)
    gen_tokens = data.get("eval_count", 0)
    prompt_ns = data.get("prompt_eval_duration", 0)
    gen_ns = data.get("eval_duration", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "ttft_ms": round(prompt_ns / 1e6, 1),
        "prefill_tok_per_sec": round(prompt_tokens / (prompt_ns / 1e9), 1) if prompt_ns > 0 else 0,
        "gen_tok_per_sec": round(gen_tokens / (gen_ns / 1e9), 1) if gen_ns > 0 else 0,
        "total_time_s": round(elapsed, 2),
        "response": data.get("response", "")[:300],
    }


# ═══════════════════════════════════════════════════════
#  TurboQuant Compression Quality
# ═══════════════════════════════════════════════════════
def run_compression_quality(num_layers, num_kv_heads, head_dim):
    """Run TurboQuant 4-bit compression quality test on simulated KV cache."""
    print(f"\n  Simulating KV cache: {num_layers}L × {num_kv_heads}KV × {head_dim}dim")

    cos_scores, ip_scores = [], []
    total_orig, total_comp = 0, 0

    for layer in range(num_layers):
        key = torch.randn(1, num_kv_heads, 256, head_dim, dtype=torch.float16)
        value = torch.randn(1, num_kv_heads, 256, head_dim, dtype=torch.float16)

        tq_k = TurboQuant(bits=4, head_dim=head_dim, seed=42 + layer)
        tq_v = TurboQuant(bits=4, head_dim=head_dim, seed=1000 + layer)

        comp_k = tq_k.compress(key)
        recon_k = tq_k.decompress(comp_k)
        comp_v = tq_v.compress(value)

        cos_scores.append(cosine_similarity(key, recon_k))
        ip_scores.append(
            measure_distortion(key, recon_k)["inner_product_correlation"]
        )

        mem_k = tq_k.memory_bytes(comp_k)
        mem_v = tq_v.memory_bytes(comp_v)
        total_orig += mem_k["original"] + mem_v["original"]
        total_comp += mem_k["compressed"] + mem_v["compressed"]

    ratio = total_orig / total_comp
    return {
        "ratio": round(ratio, 1),
        "cosine_mean": round(sum(cos_scores) / len(cos_scores), 4),
        "cosine_min": round(min(cos_scores), 4),
        "ip_corr": round(sum(ip_scores) / len(ip_scores), 4),
    }


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant Context Scaling Benchmark (any Ollama model)")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model name (e.g. qwen3.5:4b, llama3:8b, gemma3:4b)")
    parser.add_argument("--ram", type=int, default=DEFAULT_RAM_MB,
                        help=f"Total system RAM in MB (default: {DEFAULT_RAM_MB})")
    parser.add_argument("--contexts", type=str, default=None,
                        help="Comma-separated context lengths (default: auto-scaled to max feasible)")
    args = parser.parse_args()

    total_ram_mb = args.ram

    print("=" * 72)
    print("  TurboQuant × Ollama — Context Scaling Benchmark")
    print("  KV-cache compression: 4-bit with near-zero quality loss")
    print("=" * 72)

    # ─── Detect model ───
    models = list_ollama_models()
    model = args.model if args.model else pick_model(models)

    if model not in models:
        # Check for partial match (e.g. "llama3" matches "llama3:8b")
        matches = [m for m in models if model in m]
        if matches:
            model = matches[0]
        else:
            print(f"\n  Model '{model}' not found in Ollama.")
            print(f"  Available: {', '.join(models[:10])}")
            print(f"  Run: ollama pull {model}")
            sys.exit(1)

    config, raw_info = get_model_config(model)
    validate_config(config, model)

    num_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    hidden_size = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    ctx_length = config.get("context_length", "unknown")
    thinking = is_thinking_model(config)

    # Hybrid models (e.g. Qwen 3.5) mix attention + SSM layers.
    # Only attention layers have KV cache. full_attention_interval=N means
    # 1 in N layers uses full attention; the rest use SSM (no KV cache).
    attn_interval = config.get("full_attention_interval")
    if attn_interval and attn_interval > 1:
        kv_layers = num_layers // attn_interval
        is_hybrid = True
        print(f"\n  Hybrid model: {kv_layers}/{num_layers} layers use full attention (interval={attn_interval})")
    else:
        kv_layers = num_layers
        is_hybrid = False

    # FP16 KV bytes per token: 2 (K+V) × kv_layers × kv_heads × head_dim × 2 (fp16)
    fp16_per_token = 2 * kv_layers * num_kv_heads * head_dim * 2  # bytes

    # Estimate model weight size
    size_str = str(config.get("parameter_size", "4B"))
    if "M" in size_str.upper():
        params_b = float(size_str.upper().replace("M", "")) / 1000
    else:
        params_b = float(size_str.upper().replace("B", ""))
    weight_mb = params_b * 1000 * 0.5  # Q4 ≈ 0.5 GB per billion params
    available_mb = total_ram_mb - OS_RESERVED_MB - weight_mb

    # Auto-compute context lengths or use user-provided list
    if args.contexts:
        context_lengths = [int(x.strip()) for x in args.contexts.split(",")]
    else:
        context_lengths = compute_context_lengths(
            fp16_per_token, weight_mb, total_ram_mb,
            ctx_length if isinstance(ctx_length, int) else 0,
        )

    print(f"\n  Model:        {model}")
    print(f"  Family:       {config['family']}")
    print(f"  Size:         {config['parameter_size']}")
    print(f"  Quantization: {config['quantization']}")
    if is_hybrid:
        print(f"  Architecture: {num_layers}L ({kv_layers} attn) × {n_heads}H × {num_kv_heads}KV × {head_dim}dim")
    else:
        print(f"  Architecture: {num_layers}L × {n_heads}H × {num_kv_heads}KV × {head_dim}dim")
    print(f"  Context len:  {ctx_length}")
    print(f"  FP16 KV/token: {fp16_per_token} bytes ({fp16_per_token / 1024:.1f} KB)")
    if thinking:
        print(f"  Thinking mode: detected (will disable for benchmarking)")
    print(f"  System RAM:   {total_ram_mb} MB")
    print(f"  Model weights: ~{weight_mb:.0f} MB (Q4 estimate)")
    print(f"  Available for KV: ~{available_mb:.0f} MB")
    print(f"  Test contexts: {', '.join(f'{c//1024}K' for c in context_lengths)}")

    # Calibrate tokenizer for accurate prompt sizing
    calibrate_tokens_per_block(model)

    # ─── Phase 1: TurboQuant compression quality ───
    print(f"\n{'═' * 72}")
    print(f"  Phase 1: TurboQuant 4-bit Compression Quality")
    print(f"{'═' * 72}")

    tq_quality = run_compression_quality(kv_layers, num_kv_heads, head_dim)
    print(f"\n  4-bit TurboQuant Results:")
    print(f"    Compression ratio:    {tq_quality['ratio']}x")
    print(f"    Cosine similarity:    {tq_quality['cosine_mean']} (mean), {tq_quality['cosine_min']} (min)")
    print(f"    IP correlation:       {tq_quality['ip_corr']}")

    # ─── Phase 2: Ollama inference at different contexts ───
    print(f"\n{'═' * 72}")
    print(f"  Phase 2: Ollama Inference at Different Context Lengths")
    print(f"{'═' * 72}")

    ollama_results = {}
    for target_ctx in context_lengths:
        # Check if this context would even fit in memory
        fp16_kv_mb = fp16_per_token * target_ctx / 1024 / 1024
        total_mem = weight_mb + fp16_kv_mb
        if total_mem > total_ram_mb - 2000:
            print(f"\n  Context {target_ctx:,} ({target_ctx//1024}K): SKIP — needs ~{total_mem:.0f} MB (exceeds RAM)")
            ollama_results[target_ctx] = {"error": "exceeds_ram", "est_mem_mb": total_mem}
            continue

        print(f"\n  Context ~{target_ctx:,} ({target_ctx//1024}K) tokens:")
        prompt = make_prompt(target_ctx)

        # Verify actual token count if tokenize API is available
        actual_tok = count_tokens(model, prompt)
        if actual_tok:
            print(f"    Prompt built:   {actual_tok:,} tokens (target: {target_ctx:,})")

        result = run_generation(
            model, prompt, max_tokens=100,
            num_ctx=max(target_ctx + 500, 4096),
            disable_thinking=thinking,
        )

        ollama_results[target_ctx] = result

        if "error" in result:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    Actual prompt:  {result['prompt_tokens']:,} tokens")
            print(f"    TTFT:           {result['ttft_ms']:,.1f} ms")
            print(f"    Prefill speed:  {result['prefill_tok_per_sec']:,.1f} tok/s")
            print(f"    Gen speed:      {result['gen_tok_per_sec']:.1f} tok/s ({result['gen_tokens']} tokens)")
            print(f"    Total time:     {result['total_time_s']}s")
            resp_preview = result.get("response", "")[:100]
            if resp_preview:
                print(f"    Response:       {resp_preview}...")

    # ─── Phase 3: Combined results table ───
    print(f"\n{'═' * 72}")
    print(f"  Phase 3: Combined Results — Inference + Memory Savings")
    print(f"{'═' * 72}")

    tq_ratio = tq_quality["ratio"]

    header = (
        f"\n  {'Context':>7} | {'Prompt':>6} | {'TTFT':>8} | {'Gen':>8} | "
        f"{'FP16 KV':>8} | {'TQ4 KV':>8} | {'Saved':>6} | {'Total':>8} | {'Fits?':>5}"
    )
    print(header)
    print(f"  {'':>7} | {'tokens':>6} | {'(ms)':>8} | {'(tok/s)':>8} | "
          f"{'(MB)':>8} | {'(MB)':>8} | {'(%)':>6} | {'w/ TQ4':>8} | {total_ram_mb//1024}GB".rjust(5))
    print(f"  {'─' * 88}")

    for ctx in context_lengths:
        o = ollama_results.get(ctx, {})
        fp16_kv = fp16_per_token * ctx / 1024 / 1024
        tq4_kv = fp16_kv / tq_ratio
        saved_pct = (1 - 1 / tq_ratio) * 100
        total_tq = weight_mb + tq4_kv
        fits = "YES" if total_tq < (total_ram_mb - OS_RESERVED_MB) else "NO"

        if "error" in o:
            err = "OOM" if "exceeds" in str(o.get("error", "")) else "ERR"
            print(
                f"  {ctx:>7} | {err:>6} | {'---':>8} | {'---':>8} | "
                f"{fp16_kv:>7.1f} | {tq4_kv:>7.1f} | {saved_pct:>5.0f}% | "
                f"{total_tq:>7.0f} | {fits:>5}"
            )
        else:
            actual = o.get("prompt_tokens", ctx)
            ttft = o.get("ttft_ms", 0)
            gen_tps = o.get("gen_tok_per_sec", 0)
            print(
                f"  {ctx:>7} | {actual:>6} | {ttft:>7.1f} | {gen_tps:>7.1f} | "
                f"{fp16_kv:>7.1f} | {tq4_kv:>7.1f} | {saved_pct:>5.0f}% | "
                f"{total_tq:>7.0f} | {fits:>5}"
            )

    # ─── Phase 4: Feasibility analysis ───
    print(f"\n{'═' * 72}")
    print(f"  Phase 4: What Fits on Your {total_ram_mb // 1024}GB System?")
    print(f"{'═' * 72}")

    print(f"\n  Budget breakdown:")
    print(f"    Total RAM:          {total_ram_mb:>7} MB")
    print(f"    macOS + apps:      -{OS_RESERVED_MB:>7} MB")
    print(f"    Model weights (Q4): -{weight_mb:>7.0f} MB")
    print(f"    ─────────────────────────────")
    print(f"    Available for KV:   {available_mb:>7.0f} MB")

    # Max context with FP16
    max_ctx_fp16 = int(available_mb * 1024 * 1024 / fp16_per_token)
    # Max context with TurboQuant 4-bit
    max_ctx_tq4 = int(available_mb * 1024 * 1024 / (fp16_per_token / tq_ratio))

    print(f"\n  Maximum context length:")
    print(f"    FP16 KV cache:      {max_ctx_fp16:>7,} tokens  ({max_ctx_fp16 // 1024}K)")
    print(f"    TurboQuant 4-bit:   {max_ctx_tq4:>7,} tokens  ({max_ctx_tq4 // 1024}K)")
    print(f"    Improvement:        {max_ctx_tq4 / max_ctx_fp16:.1f}x longer context!")

    # Visual bar chart
    print(f"\n  Context capacity (each block = 4K tokens):")
    fp16_blocks = max_ctx_fp16 // 4096
    tq4_blocks = min(max_ctx_tq4 // 4096, 120)  # cap display width
    fp16_display = min(fp16_blocks, 120)
    print(f"    FP16:  {'█' * fp16_display}{'░' * max(0, tq4_blocks - fp16_display)} {max_ctx_fp16 // 1024}K")
    print(f"    TQ4:   {'█' * tq4_blocks} {max_ctx_tq4 // 1024}K")

    # ─── Phase 5: Scaling analysis ───
    print(f"\n{'═' * 72}")
    print(f"  Phase 5: Performance Scaling")
    print(f"{'═' * 72}")

    # TTFT scaling
    valid = [(ctx, ollama_results[ctx])
             for ctx in context_lengths
             if ctx in ollama_results and "error" not in ollama_results[ctx]]

    if len(valid) >= 2:
        print(f"\n  TTFT vs Context Length:")
        base_ctx, base_result = valid[0]
        base_ttft = base_result["ttft_ms"]
        for ctx, result in valid:
            ttft = result["ttft_ms"]
            ratio = ttft / base_ttft if base_ttft > 0 else 0
            bar_len = min(int(ttft / 100), 50)
            bar = "█" * bar_len
            print(f"    {ctx:>6} tokens: {ttft:>8.1f} ms  ({ratio:>5.1f}x)  {bar}")

        print(f"\n  Generation Speed vs Context Length:")
        for ctx, result in valid:
            tps = result["gen_tok_per_sec"]
            bar_len = min(int(tps / 2), 50)
            bar = "█" * bar_len
            print(f"    {ctx:>6} tokens: {tps:>8.1f} tok/s  {bar}")

    # ─── Summary ───
    print(f"\n{'═' * 72}")
    print(f"  SUMMARY")
    print(f"{'═' * 72}")

    print(f"""
  Model:  {model} ({config.get('parameter_size')})
  Arch:   {num_layers}L ({kv_layers} attn) × {n_heads}H × {num_kv_heads}KV × {head_dim}dim
  System: {total_ram_mb // 1024} GB RAM

  TurboQuant 4-bit KV Compression:
    Compression ratio:  {tq_quality['ratio']}x
    Cosine similarity:  {tq_quality['cosine_mean']} (near-lossless)
    IP correlation:     {tq_quality['ip_corr']}

  Context Length on Your System:
    Without TurboQuant: {max_ctx_fp16:>7,} tokens ({max_ctx_fp16 // 1024}K)
    With TurboQuant:    {max_ctx_tq4:>7,} tokens ({max_ctx_tq4 // 1024}K)
    Improvement:        {max_ctx_tq4 / max_ctx_fp16:.1f}x more context!
""")

    # ─── Save results ───
    # Dynamic filename based on model
    safe_name = model.replace(":", "_").replace("/", "_").replace(".", "")
    out_path = f"{safe_name}_context_scaling_results.json"

    output = {
        "model": model,
        "system": f"Apple Silicon, {total_ram_mb // 1024}GB",
        "config": {k: v for k, v in config.items()},  # full config with real arch
        "turboquant_quality": tq_quality,
        "ollama_results": {str(k): v for k, v in ollama_results.items()},
        "max_context_fp16": max_ctx_fp16,
        "max_context_tq4": max_ctx_tq4,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
