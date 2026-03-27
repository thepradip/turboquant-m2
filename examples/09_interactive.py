#!/usr/bin/env python3
"""
TurboQuant Interactive — Type your own prompts, see FP16 vs Compressed output.

Usage:
  python examples/09_interactive.py

Requires: pip install turboquant[transformers]
"""

import torch
import time
import gc
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import TurboQuant, cosine_similarity, measure_distortion

# ─── Config ───
MODEL_ID = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = "cpu"
MAX_NEW_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))
BITS = int(os.environ.get("BITS", "4"))


def load_model():
    print(f"\n  Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    num_layers = model.config.num_hidden_layers

    print(f"  Model:  {MODEL_ID}")
    print(f"  Arch:   {num_layers}L × {num_kv_heads}KV × {head_dim}dim")
    print(f"  Device: {DEVICE}")
    print(f"  Bits:   {BITS}-bit TurboQuant\n")

    return model, tokenizer, head_dim, num_layers


def generate_fp16(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    elapsed = time.time() - t0

    generated = outputs[0][inputs.input_ids.shape[1]:]
    n_tokens = len(generated)
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response, n_tokens, elapsed


def generate_turboquant(model, tokenizer, prompt, head_dim, bits):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Prefill
    t0 = time.time()
    with torch.no_grad():
        prefill = model(**inputs, use_cache=True)

    original_cache = prefill.past_key_values
    num_layers = len(original_cache.layers)

    # Compress → Decompress KV cache
    t_compress = time.time()
    reconstructed_cache = DynamicCache()
    cos_scores = []
    total_orig, total_comp = 0, 0

    for li in range(num_layers):
        key = original_cache.layers[li].keys
        value = original_cache.layers[li].values

        tq_k = TurboQuant(bits=bits, head_dim=head_dim, seed=42 + li)
        tq_v = TurboQuant(bits=bits, head_dim=head_dim, seed=1000 + li)

        comp_k = tq_k.compress(key)
        comp_v = tq_v.compress(value)
        recon_k = tq_k.decompress(comp_k)
        recon_v = tq_v.decompress(comp_v)

        cos_scores.append(cosine_similarity(key, recon_k))
        mem = tq_k.memory_bytes(comp_k)
        total_orig += mem["original"] * 2
        total_comp += mem["compressed"] * 2

        reconstructed_cache.update(recon_k, recon_v, li)

    compress_ms = (time.time() - t_compress) * 1000

    # Generate with reconstructed cache
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [next_token.item()]

    for _ in range(MAX_NEW_TOKENS - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=reconstructed_cache, use_cache=True)
        reconstructed_cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_tokens.append(tok_id)

    elapsed = time.time() - t0
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    avg_cos = sum(cos_scores) / len(cos_scores)
    ratio = total_orig / total_comp if total_comp else 0

    return response, len(generated_tokens), elapsed, avg_cos, compress_ms, ratio


def print_side_by_side(label1, text1, label2, text2, width=70):
    print(f"\n  ┌─ {label1}:")
    for line in text1.split("\n"):
        while len(line) > width:
            print(f"  │ {line[:width]}")
            line = line[width:]
        print(f"  │ {line}")
    print(f"  └─\n")
    print(f"  ┌─ {label2}:")
    for line in text2.split("\n"):
        while len(line) > width:
            print(f"  │ {line[:width]}")
            line = line[width:]
        print(f"  │ {line}")
    print(f"  └─")


def main():
    print("=" * 70)
    print("  TurboQuant Interactive — Type your prompts!")
    print("=" * 70)
    print(f"  Settings: MODEL={MODEL_ID} BITS={BITS} MAX_TOKENS={MAX_NEW_TOKENS}")
    print(f"  Change with: MODEL=... BITS=... MAX_TOKENS=... python {sys.argv[0]}")

    model, tokenizer, head_dim, num_layers = load_model()

    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("  Each prompt runs TWICE: FP16 baseline vs TurboQuant compressed.\n")

    prompt_num = 0
    while True:
        try:
            prompt = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break

        prompt_num += 1
        print(f"\n{'─' * 70}")
        print(f"  Prompt #{prompt_num}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'─' * 70}")

        # FP16 baseline
        print("\n  Running FP16 baseline...", end="", flush=True)
        fp16_resp, fp16_tokens, fp16_time = generate_fp16(model, tokenizer, prompt)
        fp16_tps = fp16_tokens / fp16_time if fp16_time > 0 else 0
        print(f" done ({fp16_tokens} tokens, {fp16_tps:.1f} tok/s)")

        # TurboQuant
        print(f"  Running TurboQuant {BITS}-bit...", end="", flush=True)
        tq_resp, tq_tokens, tq_time, avg_cos, compress_ms, ratio = generate_turboquant(
            model, tokenizer, prompt, head_dim, BITS
        )
        tq_tps = tq_tokens / tq_time if tq_time > 0 else 0
        print(f" done ({tq_tokens} tokens, {tq_tps:.1f} tok/s)")

        # Show outputs
        print_side_by_side(
            f"FP16 Baseline ({fp16_tokens} tok, {fp16_time:.1f}s)",
            fp16_resp,
            f"TurboQuant {BITS}-bit ({tq_tokens} tok, {tq_time:.1f}s)",
            tq_resp,
        )

        # Stats
        print(f"\n  ┌─ Stats:")
        print(f"  │ KV cosine similarity:  {avg_cos:.4f}")
        print(f"  │ KV compression ratio:  {ratio:.1f}x")
        print(f"  │ KV compress time:      {compress_ms:.1f} ms")
        print(f"  │ FP16 speed:            {fp16_tps:.1f} tok/s")
        print(f"  │ TQ speed:              {tq_tps:.1f} tok/s")
        print(f"  └─")
        print()

        gc.collect()


if __name__ == "__main__":
    main()
