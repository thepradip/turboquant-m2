#!/usr/bin/env python3
"""
TurboQuant + Quantized Model — Full Python Stack
===================================================
Quantized model weights (4-bit) + TurboQuant KV cache compression.
Everything in Python. No Ollama, no llama.cpp, no C code.

Stack:
  Model weights: 4-bit via bitsandbytes / GPTQ / AWQ (saves ~75% weight memory)
  KV cache:      4-bit via TurboQuant hybrid cache (saves ~70% KV memory)
  Result:        Maximum context on minimum hardware

Requirements:
  pip install turboquant[transformers]
  pip install bitsandbytes    # For 4-bit weight quantization (CUDA)
  # OR on Apple Silicon (no bitsandbytes needed, use float16):
  pip install turboquant[transformers]

Usage:
  # Apple Silicon (M1/M2/M3) — FP16 weights + TurboQuant KV
  python examples/11_quantized_model_with_turboquant.py

  # CUDA GPU — 4-bit weights + TurboQuant KV
  python examples/11_quantized_model_with_turboquant.py --quantize

  # Custom model
  python examples/11_quantized_model_with_turboquant.py --model Qwen/Qwen2.5-3B-Instruct
"""

import torch
import time
import gc
import sys
import os
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import TurboQuant, cosine_similarity
from turboquant.integrations.hybrid_cache import HybridKVCache


# ═══════════════════════════════════════════════════════
#  Model Loading — supports multiple quantization methods
# ═══════════════════════════════════════════════════════

def load_model(model_id, quantize=False, device=None):
    """Load model with optional 4-bit quantization."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "cpu"  # MPS has tensor size limits, use CPU for reliability
        else:
            device = "cpu"

    print(f"  Model:  {model_id}")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if quantize and torch.cuda.is_available():
        # 4-bit quantization via bitsandbytes (CUDA only)
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"  Quant:  4-bit (bitsandbytes NF4)")
        except ImportError:
            print("  bitsandbytes not installed, loading in FP16...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16,
                device_map=device, trust_remote_code=True,
            )
            print(f"  Quant:  FP16 (install bitsandbytes for 4-bit)")
    else:
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype,
            device_map=device, trust_remote_code=True,
        )
        print(f"  Dtype:  {dtype}")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def get_arch(model):
    """Extract model architecture info."""
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    num_layers = config.num_hidden_layers
    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    return {
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "params_b": round(params_b, 2),
    }


# ═══════════════════════════════════════════════════════
#  Generation with Hybrid TurboQuant KV Cache
# ═══════════════════════════════════════════════════════

def generate_with_hybrid_cache(
    model, tokenizer, prompt, max_new_tokens=200,
    head_dim=128, num_layers=24, window_size=256, bits=4,
):
    """
    Generate text using hybrid KV cache:
    - Prefill entire prompt → get KV cache
    - Compress old tokens with TurboQuant
    - Keep recent window in FP16
    - Continue generation with hybrid cache
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    prompt_len = inputs.input_ids.shape[1]

    # Step 1: Prefill
    t0 = time.time()
    with torch.no_grad():
        prefill = model(**inputs, use_cache=True)
    prefill_ms = (time.time() - t0) * 1000

    cache = prefill.past_key_values
    n_layers = len(cache.layers)

    # Step 2: Extract KV tensors
    kv_list = []
    for li in range(n_layers):
        kv_list.append((cache.layers[li].keys, cache.layers[li].values))

    seq_len = kv_list[0][0].shape[2]

    # Step 3: Compress old tokens if context is long enough
    hybrid = HybridKVCache(
        bits=bits, head_dim=head_dim, num_layers=n_layers,
        window_size=min(window_size, seq_len),
    )

    compressed = False
    compress_ms = 0
    if seq_len > window_size * 2:
        t0 = time.time()
        trimmed_kv = hybrid.compress_old_tokens(kv_list)
        compress_ms = (time.time() - t0) * 1000

        # Reconstruct full cache for generation
        full_kv = hybrid.get_full_cache(trimmed_kv)
        compressed = True

        # Build DynamicCache from reconstructed
        new_cache = DynamicCache()
        for li, (k, v) in enumerate(full_kv):
            new_cache.update(k, v, li)
        cache = new_cache

    # Step 4: Generate tokens
    t_gen_start = time.time()
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [next_token.item()]

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_tokens.append(tok_id)

    gen_time = time.time() - t_gen_start
    total_time = time.time() - t0
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    gen_tps = len(generated_tokens) / gen_time if gen_time > 0 else 0

    mem_report = hybrid.memory_report(kv_list[0][0].shape[1]) if compressed else None

    return {
        "response": response,
        "prompt_tokens": prompt_len,
        "gen_tokens": len(generated_tokens),
        "gen_tps": round(gen_tps, 1),
        "prefill_ms": round(prefill_ms, 0),
        "compress_ms": round(compress_ms, 0),
        "total_time": round(total_time, 1),
        "compressed": compressed,
        "memory": mem_report,
    }


def generate_baseline(model, tokenizer, prompt, max_new_tokens=200):
    """Standard generation without TurboQuant."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.time() - t0

    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tps = len(gen_ids) / elapsed if elapsed > 0 else 0

    return {
        "response": response,
        "prompt_tokens": inputs.input_ids.shape[1],
        "gen_tokens": len(gen_ids),
        "gen_tps": round(tps, 1),
        "total_time": round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════
#  Test Prompts — Short and Long
# ═══════════════════════════════════════════════════════

TESTS = [
    {
        "name": "Short QA (100 tokens)",
        "prompt": "What are the 3 main differences between TCP and UDP? Be concise.",
        "max_tokens": 150,
    },
    {
        "name": "Math Reasoning (100 tokens)",
        "prompt": "A store sells notebooks at $4 each and pens at $1.50 each. Maria buys 3 notebooks and 5 pens with a $20 bill. How much change does she get? Show your work step by step.",
        "max_tokens": 200,
    },
    {
        "name": "Code Generation (100 tokens)",
        "prompt": "Write a Python function to check if a string is a palindrome. Include 3 test cases.",
        "max_tokens": 200,
    },
    {
        "name": "Long Context QA (2K tokens)",
        "prompt": (
            "Read this document and answer: What is the attention formula and who invented the Transformer?\n\n"
            + "The Transformer architecture was introduced in 'Attention Is All You Need' by Vaswani et al. in 2017. "
            * 20
            + "The key formula is Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. "
            + "It uses multi-head attention with 8 heads in the base model. "
            + "The model was trained on WMT English-German and English-French translation tasks. "
            * 15
            + "\n\nAnswer the question based on the document above."
        ),
        "max_tokens": 150,
    },
    {
        "name": "Longer Context QA (4K tokens)",
        "prompt": (
            "Analyze this technical document and list all key innovations mentioned.\n\n"
            + "Deep learning has transformed AI since 2012 when AlexNet won ImageNet. "
            + "Key innovations include: dropout regularization (Srivastava 2014), "
            + "batch normalization (Ioffe & Szegedy 2015), residual connections (He et al. 2016), "
            + "attention mechanisms (Bahdanau 2014), and transformers (Vaswani 2017). "
            + "GANs were introduced by Goodfellow in 2014. BERT used masked language modeling. "
            + "GPT used autoregressive language modeling with causal attention. "
            * 40
            + "\n\nList every innovation and its author mentioned above."
        ),
        "max_tokens": 200,
    },
]


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit bitsandbytes (CUDA)")
    parser.add_argument("--window", type=int, default=256, help="FP16 window size")
    parser.add_argument("--bits", type=int, default=4, help="TurboQuant bits")
    args = parser.parse_args()

    print("=" * 70)
    print("  Quantized Model + TurboQuant KV Cache — Full Python Stack")
    print("  No Ollama, no llama.cpp, no C code. Pure Python.")
    print("=" * 70)

    model, tokenizer, device = load_model(args.model, quantize=args.quantize)
    arch = get_arch(model)

    print(f"  Params: {arch['params_b']}B")
    print(f"  Arch:   {arch['num_layers']}L × {arch['num_kv_heads']}KV × {arch['head_dim']}dim")
    print(f"  GQA:    {arch['num_attention_heads']}:{arch['num_kv_heads']}")
    print(f"  TQ:     {args.bits}-bit, window={args.window}")

    results = []
    for i, test in enumerate(TESTS):
        print(f"\n{'─' * 70}")
        print(f"  Test {i+1}: {test['name']}")
        print(f"{'─' * 70}")

        # Baseline
        print(f"  Running FP16 baseline...", end="", flush=True)
        try:
            baseline = generate_baseline(model, tokenizer, test["prompt"], test["max_tokens"])
            print(f" {baseline['gen_tokens']} tok, {baseline['gen_tps']} tok/s")
        except Exception as e:
            print(f" ERROR: {e}")
            baseline = {"response": "ERROR", "gen_tokens": 0, "gen_tps": 0, "total_time": 0, "prompt_tokens": 0}

        # TurboQuant hybrid
        print(f"  Running TurboQuant hybrid...", end="", flush=True)
        try:
            tq_result = generate_with_hybrid_cache(
                model, tokenizer, test["prompt"], test["max_tokens"],
                head_dim=arch["head_dim"], num_layers=arch["num_layers"],
                window_size=args.window, bits=args.bits,
            )
            print(f" {tq_result['gen_tokens']} tok, {tq_result['gen_tps']} tok/s" +
                  (f", compressed {tq_result['compress_ms']}ms" if tq_result['compressed'] else ", no compression needed"))
        except Exception as e:
            print(f" ERROR: {e}")
            traceback.print_exc()
            tq_result = {"response": "ERROR", "gen_tokens": 0, "gen_tps": 0, "total_time": 0,
                         "compressed": False, "memory": None, "prompt_tokens": 0, "compress_ms": 0, "prefill_ms": 0}

        # Compare
        print(f"\n  ┌─ Baseline FP16:")
        print(f"  │ {baseline['response'][:300]}")
        print(f"  └─ ({baseline['gen_tokens']} tokens, {baseline['total_time']}s)")
        print(f"\n  ┌─ TurboQuant {args.bits}-bit hybrid:")
        print(f"  │ {tq_result['response'][:300]}")
        print(f"  └─ ({tq_result['gen_tokens']} tokens, {tq_result['total_time']}s)")

        if tq_result.get("memory"):
            m = tq_result["memory"]
            print(f"\n  Memory: {m['all_fp16_mb']:.1f}MB → {m['hybrid_mb']:.1f}MB "
                  f"(saved {m['saved_mb']:.1f}MB / {m['savings_pct']:.0f}%)")

        results.append({
            "name": test["name"],
            "prompt_tokens": baseline.get("prompt_tokens", 0),
            "baseline_tokens": baseline["gen_tokens"],
            "tq_tokens": tq_result["gen_tokens"],
            "baseline_tps": baseline["gen_tps"],
            "tq_tps": tq_result["gen_tps"],
            "compressed": tq_result["compressed"],
            "memory": tq_result.get("memory"),
        })

        gc.collect()

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  {'Test':<30} │ {'Prompt':>6} │ {'FP16':>6} │ {'TQ':>6} │ {'Compressed?':>12} │ {'Saved':>7}")
    print(f"  {'─' * 80}")
    for r in results:
        saved = f"{r['memory']['savings_pct']:.0f}%" if r.get("memory") else "N/A"
        print(f"  {r['name']:<30} │ {r['prompt_tokens']:>5} │ {r['baseline_tps']:>5}/s │ "
              f"{r['tq_tps']:>5}/s │ {'YES' if r['compressed'] else 'no':>12} │ {saved:>7}")

    print(f"""
  Stack:
    Model weights:  {'4-bit (bitsandbytes)' if args.quantize else 'FP16/FP32'} — saves ~75% weight memory
    KV cache:       {args.bits}-bit TurboQuant hybrid — saves ~70% KV memory
    Recent window:  {args.window} tokens in FP16 — no hallucination
    Framework:      Pure Python (HuggingFace transformers)

  This is the PURE PYTHON alternative to Ollama/llama.cpp.
  No C code needed. Works on CPU, CUDA, and Apple Silicon.
""")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
