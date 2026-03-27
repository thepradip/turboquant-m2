#!/usr/bin/env python3
"""
TurboQuant Real-World App — Chatbot, QA, RAG
==============================================
Tests TurboQuant in actual use cases, not just benchmarks.

Modes:
  1. Chatbot — multi-turn conversation (context grows each turn)
  2. Long Document QA — paste a document, ask questions
  3. RAG — retrieve chunks, stuff into context, answer
  4. Interactive — type your own prompts

Compares: Baseline (FP16 KV) vs TurboQuant (compressed KV)
Measures: Speed, memory, quality — at each turn/query

Usage:
  source /tmp/tq_real_test/bin/activate
  python examples/15_real_app.py --mode chatbot
  python examples/15_real_app.py --mode qa
  python examples/15_real_app.py --mode rag
  python examples/15_real_app.py --mode interactive
  python examples/15_real_app.py --mode all
"""

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

import time
import gc
import sys
import os
import argparse
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turboquant.mlx_native import compress_kv_cache_mlx, get_model_config


# ═══════════════════════════════════════════════════════
#  Generation helpers
# ═══════════════════════════════════════════════════════

def generate_baseline(model, tokenizer, messages, max_tokens=200):
    """Standard generation — FP16 KV cache."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(text))
    mem_before = psutil.Process().memory_info().rss

    t0 = time.time()
    resp = mlx_lm.generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0

    mem_after = psutil.Process().memory_info().rss
    gen_tokens = len(tokenizer.encode(resp))
    for tag in ["<think>", "</think>"]:
        resp = resp.replace(tag, "")

    return {
        "response": resp.strip(),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
        "time_s": round(elapsed, 2),
        "ram_delta_mb": round((mem_after - mem_before) / 1024 / 1024),
    }


def generate_turboquant(model, tokenizer, messages, max_tokens=200, window=512):
    """TurboQuant generation — compressed KV cache."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = mx.array(tokenizer.encode(text))
    prompt_tokens = len(ids)

    mem_before = psutil.Process().memory_info().rss
    cache = make_prompt_cache(model)

    t0 = time.time()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)

    tq = compress_kv_cache_mlx(cache, model=model, window_size=window, min_context=512)

    y = mx.argmax(logits[:, -1, :], axis=-1)
    tokens = [y.item()]
    for _ in range(max_tokens - 1):
        logits = model(y.reshape(1, -1), cache=cache)
        mx.eval(logits)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        if y.item() == tokenizer.eos_token_id:
            break
        tokens.append(y.item())

    elapsed = time.time() - t0
    mem_after = psutil.Process().memory_info().rss

    resp = tokenizer.decode(tokens)
    for tag in ["<think>", "</think>"]:
        resp = resp.replace(tag, "")

    return {
        "response": resp.strip(),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": len(tokens),
        "tok_per_sec": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
        "time_s": round(elapsed, 2),
        "ram_delta_mb": round((mem_after - mem_before) / 1024 / 1024),
        "cosine": tq["cosine"],
        "compress_ms": tq["compress_ms"],
        "saved_mb": tq.get("saved_mb", 0),
    }


def print_comparison(name, bl, tq):
    """Print side-by-side comparison."""
    print(f"\n  {'─' * 60}")
    print(f"  {name}")
    print(f"  Prompt: {bl['prompt_tokens']} tokens")
    print(f"  {'─' * 60}")
    print(f"  {'':>15} │ {'Baseline':>12} │ {'TurboQuant':>12}")
    print(f"  {'─' * 45}")
    print(f"  {'Speed':>15} │ {bl['tok_per_sec']:>10}/s │ {tq['tok_per_sec']:>10}/s")
    print(f"  {'Time':>15} │ {bl['time_s']:>10}s │ {tq['time_s']:>10}s")
    print(f"  {'Tokens':>15} │ {bl['gen_tokens']:>12} │ {tq['gen_tokens']:>12}")
    if tq.get("cosine"):
        print(f"  {'Cosine':>15} │ {'1.0000':>12} │ {tq['cosine']:>12}")
    if tq.get("compress_ms"):
        print(f"  {'Compress':>15} │ {'—':>12} │ {tq['compress_ms']:>9}ms")
    if tq.get("saved_mb"):
        print(f"  {'KV Saved':>15} │ {'—':>12} │ {tq['saved_mb']:>9}MB")
    print(f"\n  Baseline:   {bl['response'][:250]}")
    print(f"  TurboQuant: {tq['response'][:250]}")


# ═══════════════════════════════════════════════════════
#  Mode 1: Multi-turn Chatbot
# ═══════════════════════════════════════════════════════

def run_chatbot(model, tokenizer):
    print(f"\n{'═' * 60}")
    print(f"  MODE: Multi-turn Chatbot")
    print(f"  Context grows with each turn — TurboQuant shines here")
    print(f"{'═' * 60}")

    conversation = [
        "Hello! What can you help me with?",
        "Explain what a transformer is in machine learning.",
        "How does the attention mechanism work? Give me the formula.",
        "What is the difference between BERT and GPT?",
        "Now explain how KV-cache works during inference and why it uses so much memory.",
        "If I have a 7B model with 32 layers, 8 KV heads, and 128 head dim, how much memory does the KV cache use at 32K context?",
        "What are the best ways to reduce KV cache memory?",
        "Summarize everything we discussed in 5 bullet points.",
    ]

    messages_bl = []
    messages_tq = []

    for i, user_msg in enumerate(conversation):
        messages_bl.append({"role": "user", "content": user_msg})
        messages_tq.append({"role": "user", "content": user_msg})

        bl = generate_baseline(model, tokenizer, messages_bl, max_tokens=150)
        tq = generate_turboquant(model, tokenizer, messages_tq, max_tokens=150)

        print_comparison(f"Turn {i+1}/{len(conversation)}: {user_msg[:50]}...", bl, tq)

        messages_bl.append({"role": "assistant", "content": bl["response"]})
        messages_tq.append({"role": "assistant", "content": tq["response"]})
        gc.collect()


# ═══════════════════════════════════════════════════════
#  Mode 2: Long Document QA
# ═══════════════════════════════════════════════════════

LONG_DOCUMENT = """
# Machine Learning Infrastructure Guide

## Chapter 1: Model Serving

When deploying machine learning models in production, several key decisions must be made:

1. **Batch size**: Larger batches improve throughput but increase latency. For real-time applications, batch size of 1-4 is common. For offline processing, batches of 32-256 are typical.

2. **Hardware selection**: GPUs (NVIDIA A100, H100) offer the best performance for transformer models. TPUs are cost-effective for large-scale training. Apple Silicon (M2, M3) works well for on-device inference.

3. **Quantization**: Model weights can be compressed from FP16 (2 bytes) to INT4 (0.5 bytes), reducing memory by 4x with minimal quality loss. Common formats include GGUF (llama.cpp), AWQ, and GPTQ.

4. **KV-Cache Management**: During autoregressive generation, the model stores Key and Value vectors for every token. For a model with 32 layers, 8 KV heads, and 128 head dimension, each token requires 32 * 8 * 128 * 2 * 2 = 131,072 bytes (128 KB). At 32K context, this is 4 GB — often larger than the model itself.

## Chapter 2: Optimization Techniques

### PagedAttention (vLLM)
vLLM manages KV-cache like virtual memory pages. Instead of pre-allocating contiguous memory for each request, it allocates fixed-size blocks on demand. This reduces memory waste from internal fragmentation by up to 60%.

### Prefix Caching
When multiple requests share the same system prompt, the KV-cache for that prefix can be computed once and reused. This is especially valuable for RAG applications where the retrieved documents change but the instruction prefix stays the same.

### KV-Cache Quantization
TurboQuant (Google Research, ICLR 2026) compresses KV-cache from FP16 to 4-bit using random rotation + Lloyd-Max quantization. Key properties:
- 3.9x compression ratio at 4-bit
- 0.9954 cosine similarity (near-lossless)
- Data-independent: no calibration needed
- Zero overhead for normalization (unlike per-group quantization)

### Speculative Decoding
Uses a smaller "draft" model to generate candidate tokens, then verifies them with the main model in a single forward pass. Can achieve 2-3x speedup without quality loss.

## Chapter 3: Scaling Considerations

| Metric | 1B Model | 7B Model | 70B Model |
|--------|----------|----------|-----------|
| Weight Memory (FP16) | 2 GB | 14 GB | 140 GB |
| Weight Memory (Q4) | 0.5 GB | 3.5 GB | 35 GB |
| KV Cache (32K, FP16) | 0.5 GB | 4 GB | 40 GB |
| KV Cache (32K, TQ4) | 0.13 GB | 1 GB | 10 GB |
| Min GPU for inference | T4 (16GB) | A100 (40GB) | 2x H100 (160GB) |

The total memory requirement is: Model Weights + KV Cache + Activations + Overhead.

For a 7B model at 32K context: 3.5 GB (Q4 weights) + 4 GB (FP16 KV) = 7.5 GB. With TurboQuant: 3.5 GB + 1 GB = 4.5 GB — fits on a single T4!
"""

QA_QUESTIONS = [
    "How much memory does the KV cache use per token for a 32-layer model with 8 KV heads and 128 dim?",
    "What is the total memory needed for a 7B Q4 model at 32K context with TurboQuant?",
    "What are the 4 main optimization techniques mentioned? List each with one sentence.",
    "Compare PagedAttention and TurboQuant — what does each solve?",
    "If I have a T4 GPU (16GB), what is the maximum model size I can run at 32K context with TurboQuant?",
]


def run_qa(model, tokenizer):
    print(f"\n{'═' * 60}")
    print(f"  MODE: Long Document QA")
    print(f"  Same document, multiple questions")
    print(f"{'═' * 60}")

    for i, question in enumerate(QA_QUESTIONS):
        messages = [{"role": "user", "content": f"Document:\n{LONG_DOCUMENT}\n\nQuestion: {question}\nAnswer:"}]

        bl = generate_baseline(model, tokenizer, messages, max_tokens=150)
        tq = generate_turboquant(model, tokenizer, messages, max_tokens=150)

        print_comparison(f"Q{i+1}: {question[:55]}...", bl, tq)
        gc.collect()


# ═══════════════════════════════════════════════════════
#  Mode 3: RAG (Retrieval Augmented Generation)
# ═══════════════════════════════════════════════════════

RAG_CHUNKS = [
    "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "JavaScript was created by Brendan Eich in 1995 for Netscape Navigator. It is the primary language of the web, running in all modern browsers. Node.js (2009) brought JavaScript to server-side development.",
    "Rust was first released in 2015 by Mozilla Research. It focuses on memory safety without garbage collection. Rust uses a borrow checker to prevent data races at compile time. It is increasingly used in systems programming.",
    "Go (Golang) was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson in 2009. It features built-in concurrency with goroutines and channels. Go compiles to native code and has a garbage collector.",
    "TypeScript was created by Microsoft in 2012. It adds static typing to JavaScript and compiles to plain JavaScript. TypeScript has become the standard for large-scale web applications.",
]

RAG_QUERIES = [
    ("Which language focuses on memory safety?", [2]),
    ("Compare Python and JavaScript — who created them and when?", [0, 1]),
    ("Which languages were created by major tech companies?", [1, 3, 4]),
    ("I need a language for systems programming with no garbage collector. Which one and why?", [2, 3]),
    ("Rank all 5 languages by release year from oldest to newest.", [0, 1, 2, 3, 4]),
]


def run_rag(model, tokenizer):
    print(f"\n{'═' * 60}")
    print(f"  MODE: RAG (Retrieval Augmented Generation)")
    print(f"  Retrieved chunks stuffed into context")
    print(f"{'═' * 60}")

    for i, (query, chunk_ids) in enumerate(RAG_QUERIES):
        retrieved = "\n\n".join(f"[Document {j+1}]: {RAG_CHUNKS[j]}" for j in chunk_ids)
        prompt = (
            f"Use ONLY the following retrieved documents to answer the question.\n\n"
            f"{retrieved}\n\n"
            f"Question: {query}\n"
            f"Answer based on the documents above:"
        )
        messages = [{"role": "user", "content": prompt}]

        bl = generate_baseline(model, tokenizer, messages, max_tokens=150)
        tq = generate_turboquant(model, tokenizer, messages, max_tokens=150)

        print_comparison(f"RAG Q{i+1} ({len(chunk_ids)} chunks): {query[:45]}...", bl, tq)
        gc.collect()


# ═══════════════════════════════════════════════════════
#  Mode 4: Interactive
# ═══════════════════════════════════════════════════════

def run_interactive(model, tokenizer):
    print(f"\n{'═' * 60}")
    print(f"  MODE: Interactive — Type your own prompts")
    print(f"  Each prompt runs with both Baseline and TurboQuant")
    print(f"  Type 'quit' to exit")
    print(f"{'═' * 60}\n")

    messages = []
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})

        print(f"  Running baseline...", end="", flush=True)
        bl = generate_baseline(model, tokenizer, messages, max_tokens=200)
        print(f" {bl['tok_per_sec']}/s")

        print(f"  Running TurboQuant...", end="", flush=True)
        tq = generate_turboquant(model, tokenizer, messages, max_tokens=200)
        print(f" {tq['tok_per_sec']}/s")

        print_comparison(f"Your prompt ({bl['prompt_tokens']} tokens)", bl, tq)

        messages.append({"role": "assistant", "content": bl["response"]})
        gc.collect()
        print()


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    parser.add_argument("--mode", default="all", choices=["chatbot", "qa", "rag", "interactive", "all"])
    args = parser.parse_args()

    print("=" * 60)
    print("  TurboQuant Real-World App")
    print("  Chatbot | QA | RAG — Baseline vs TurboQuant")
    print("=" * 60)

    print(f"\n  Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    config = get_model_config(model)
    print(f"  Config: {config['num_layers']}L × {config['num_kv_heads']}KV × {config['head_dim']}dim")
    print(f"  RAM:    {psutil.virtual_memory().total / 1024**3:.0f} GB")

    if args.mode in ("chatbot", "all"):
        run_chatbot(model, tokenizer)

    if args.mode in ("qa", "all"):
        run_qa(model, tokenizer)

    if args.mode in ("rag", "all"):
        run_rag(model, tokenizer)

    if args.mode == "interactive":
        run_interactive(model, tokenizer)

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
