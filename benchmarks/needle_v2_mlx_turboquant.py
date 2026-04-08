#!/usr/bin/env python3
"""
Needle-in-a-Haystack v2: Real MLX inference with OUR TurboQuant.

End-to-end: model loads -> prefill -> compress_cache() with TurboQuant -> generate.
Tests FP16 baseline vs TurboQuant 4-bit vs TurboQuant 3-bit on real model output.

6 context lengths: 1K, 4K, 8K, 16K, 32K, 60K
3 difficulty levels per context (easy, medium, hard) = 18 total needles
Real text sources: CNN/DailyMail (1K-4K), ArXiv (8K-16K), PG19 books (32K-60K)
Hard questions include distractor facts to test retrieval precision.

Author: Pradip Tivhale, April 2026
Hardware: Apple M2 Pro, 16 GB
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LLM_in_Prod", "turboquant"))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

# TurboQuant compress_cache from LLM_in_Prod/turboquant
import importlib
_tq = importlib.import_module("src.turboquant")
compress_cache = _tq.compress_cache
chunked_prefill = _tq.chunked_prefill

# Custom 1-bit loader for Bonsai models
from turboquant.bonsai_loader import load_bonsai_1bit


def load_model_auto(model_name):
    """Load model — uses custom 1-bit loader for Bonsai, standard mlx_lm for others."""
    try:
        return load_bonsai_1bit(model_name)
    except Exception:
        return mlx_load(model_name)


# ============================================================================
# NEEDLES: 3 per context (easy, medium, hard), 18 total
# Hard questions include distractor facts that contradict the needle.
# ============================================================================

TESTS = [
    # --- 1K context (news style) ---
    {
        "context": "1K", "tokens": 1024, "difficulty": "easy",
        "needle": "According to internal documents, the company allocated exactly $47.3 million to Project Nightingale, approved on March 14, 2023.",
        "question": "How much money was allocated to Project Nightingale and when was it approved?",
        "check": lambda a: "47.3" in a and ("march" in a.lower() or "2023" in a),
        "position": 0.5,
    },
    {
        "context": "1K", "tokens": 1024, "difficulty": "medium",
        "needle": "The investigation revealed that 2,847 customer accounts were affected between January 8 and January 12, resulting in unauthorized transfers totaling $1.92 million.",
        "question": "How many customer accounts were affected and what was the total amount of unauthorized transfers?",
        "check": lambda a: ("2,847" in a or "2847" in a),
        "position": 0.3,
    },
    {
        "context": "1K", "tokens": 1024, "difficulty": "hard",
        "needle": "Chief Technology Officer Maria Santos confirmed that the backup recovery time was 4 hours and 22 minutes, exceeding the 2-hour SLA by 142 minutes.",
        "distractor": "The company spokesperson noted that recovery procedures were completed within the standard 2-hour window as outlined in their service agreement.",
        "question": "What was the actual backup recovery time reported by the CTO?",
        "check": lambda a: ("4 hour" in a.lower() or "22 minute" in a.lower() or "142" in a),
        "position": 0.7,
    },
    # --- 4K context (news style) ---
    {
        "context": "4K", "tokens": 4096, "difficulty": "easy",
        "needle": "Federal investigators confirmed the shipment weighed exactly 3,215 kilograms and departed on vessel MV Castellano from Rotterdam on November 3, 2024.",
        "question": "What was the exact weight of the shipment and which vessel carried it?",
        "check": lambda a: ("3,215" in a or "3215" in a) and "castellano" in a.lower(),
        "position": 0.5,
    },
    {
        "context": "4K", "tokens": 4096, "difficulty": "medium",
        "needle": "Dr. Kenji Watanabe published findings showing that the compound reduced inflammation markers by 73.6% in the Phase IIb trial involving 1,284 participants.",
        "question": "What percentage did the compound reduce inflammation markers by and how many participants?",
        "check": lambda a: "73.6" in a and ("1,284" in a or "1284" in a),
        "position": 0.25,
    },
    {
        "context": "4K", "tokens": 4096, "difficulty": "hard",
        "needle": "The internal audit found that Branch 7 in Phoenix processed 14,891 transactions on December 19 alone, triggering compliance alert CA-2024-0892.",
        "distractor": "Branch operations across the Southwest region reported normal transaction volumes throughout the holiday period, with no unusual activity flagged by automated monitoring systems.",
        "question": "Which branch processed unusual transactions on December 19 and what alert was triggered?",
        "check": lambda a: ("14,891" in a or "14891" in a) and "0892" in a,
        "position": 0.6,
    },
    # --- 8K context (scientific style) ---
    {
        "context": "8K", "tokens": 8192, "difficulty": "easy",
        "needle": "Model C achieved a BLEU score of 42.7 on the WMT-2024 benchmark, surpassing the previous state-of-the-art by 3.1 points.",
        "question": "What BLEU score did Model C achieve on WMT-2024?",
        "check": lambda a: "42.7" in a,
        "position": 0.5,
    },
    {
        "context": "8K", "tokens": 8192, "difficulty": "medium",
        "needle": "Cross-validation on the held-out test set of 5,372 samples yielded an F1 score of 0.891 for the transformer variant with statistical significance at p < 0.001.",
        "question": "What F1 score did the transformer variant achieve and what was the sample size?",
        "check": lambda a: "0.891" in a or ".891" in a,
        "position": 0.35,
    },
    {
        "context": "8K", "tokens": 8192, "difficulty": "hard",
        "needle": "The ablation study in Table 4 showed removing attention pruning decreased throughput from 847 tokens/sec to 312 tokens/sec while improving perplexity from 8.34 to 7.91.",
        "distractor": "As shown in our supplementary analysis, the pruning mechanism maintained consistent throughput across all evaluation benchmarks without measurable impact on perplexity scores.",
        "question": "What happened to throughput and perplexity when attention pruning was removed?",
        "check": lambda a: ("847" in a or "312" in a) and ("8.34" in a or "7.91" in a),
        "position": 0.75,
    },
    # --- 16K context (scientific style) ---
    {
        "context": "16K", "tokens": 16384, "difficulty": "easy",
        "needle": "The clinical trial registered as NCT-2024-88431 enrolled patients from 37 hospitals across 12 countries, with a primary endpoint of overall survival at 24 months.",
        "question": "What was the clinical trial registration number, how many hospitals participated, and across how many countries?",
        "check": lambda a: "88431" in a and "37" in a and "12" in a,
        "position": 0.5,
    },
    {
        "context": "16K", "tokens": 16384, "difficulty": "medium",
        "needle": "Sensor array deployed at coordinates 34.0522N, 118.2437W recorded a peak vibration amplitude of 0.0847g at exactly 02:14:33 UTC on February 9, 2024, which correlated with the seismic event catalogued as LA-2024-0041.",
        "question": "What was the peak vibration amplitude recorded by the sensor array, at what exact time, and what was the catalogue number of the seismic event?",
        "check": lambda a: "0.0847" in a and ("02:14" in a or "2:14" in a) and "0041" in a,
        "position": 0.2,
    },
    {
        "context": "16K", "tokens": 16384, "difficulty": "hard",
        "needle": "The retrospective analysis of Patient Cohort D (n=892) found that the combination therapy of Drug X at 150mg and Drug Y at 75mg administered bi-weekly resulted in a 5-year remission rate of 68.3%, compared to 41.7% for Drug X monotherapy.",
        "distractor": "Previous studies in similar patient populations demonstrated that standard monotherapy protocols achieved remission rates comparable to or exceeding those observed in combination therapy arms across multiple Phase III trials.",
        "question": "In Patient Cohort D, what was the 5-year remission rate for the combination therapy versus monotherapy, and what were the dosages?",
        "check": lambda a: "68.3" in a and "41.7" in a and ("150" in a or "75" in a),
        "position": 0.65,
    },
    # --- 32K context (book/narrative style) ---
    {
        "context": "32K", "tokens": 32768, "difficulty": "easy",
        "needle": "The old merchant whispered to his apprentice that the combination to the iron vault was 7-34-89-12 and that it must never be written down or shared with anyone outside the guild.",
        "question": "What was the combination to the iron vault that the merchant whispered to his apprentice?",
        "check": lambda a: "7" in a and "34" in a and "89" in a and "12" in a,
        "position": 0.5,
    },
    {
        "context": "32K", "tokens": 32768, "difficulty": "medium",
        "needle": "Among the cargo manifests recovered from the wreck, one entry stood out: 847 bolts of Venetian silk, valued at 12,400 ducats, consigned to the House of Medici and bearing the seal of Captain Lorenzo Vettori.",
        "question": "How many bolts of Venetian silk were listed in the cargo manifest, what was their value, and who was the captain?",
        "check": lambda a: "847" in a and ("12,400" in a or "12400" in a) and "vettori" in a.lower(),
        "position": 0.4,
    },
    {
        "context": "32K", "tokens": 32768, "difficulty": "hard",
        "needle": "The surveyor's report, dated October 17, 1847, recorded that the north boundary of the estate measured exactly 2,341 feet and the south boundary measured 2,187 feet, with the discrepancy attributed to the creek bed shifting 154 feet eastward since the original 1802 survey.",
        "distractor": "Local records confirmed that the estate boundaries had remained unchanged since the original survey, with both the north and south boundaries measuring within standard tolerances of the 1802 measurements.",
        "question": "According to the surveyor's 1847 report, what were the exact measurements of the north and south boundaries, and how far had the creek shifted?",
        "check": lambda a: ("2,341" in a or "2341" in a),
        "position": 0.8,
    },
    # --- 60K context (book/narrative style) ---
    {
        "context": "60K", "tokens": 60000, "difficulty": "easy",
        "needle": "The letter, postmarked from Vienna on June 23, 1891, contained a bank draft for exactly 4,750 Austrian florins payable to one Friedrich Engel of 14 Bergstrasse, Salzburg.",
        "question": "What amount was the bank draft for in the Vienna letter, and who was the payee?",
        "check": lambda a: ("4,750" in a or "4750" in a),
        "position": 0.5,
    },
    {
        "context": "60K", "tokens": 60000, "difficulty": "medium",
        "needle": "Hidden in the third drawer of the oak escritoire was a folded map showing that the mine entrance lay at precisely 47 degrees 12 minutes north, 11 degrees 23 minutes east, marked with a red cross and the initials J.K.S.",
        "question": "What were the exact coordinates of the mine entrance shown on the hidden map, and whose initials were on it?",
        "check": lambda a: ("47" in a and "12" in a) and "j.k.s" in a.lower().replace(" ", ""),
        "position": 0.15,
    },
    {
        "context": "60K", "tokens": 60000, "difficulty": "hard",
        "needle": "The inventory of the apothecary, taken after the fire of September 1847, listed 2,891 glass vials destroyed, 147 intact, and exactly 23 containing a mercury compound that Inspector Hoffman ordered sealed under case number V-1847-0334.",
        "distractor": "Fire brigade records indicated that the apothecary's stock was largely preserved through the efforts of the volunteer company, with only minor losses reported to the municipal authorities.",
        "question": "After the apothecary fire, how many glass vials were destroyed, how many were intact, and what case number did Inspector Hoffman assign?",
        "check": lambda a: ("2,891" in a or "2891" in a) and "147" in a and "0334" in a,
        "position": 0.85,
    },
]


# ============================================================================
# TEXT SOURCES: Real text from HuggingFace (no repetition)
# ============================================================================

TEXT_SOURCES = {
    1024: "cnn_dailymail",
    4096: "cnn_dailymail",
    8192: "arxiv",
    16384: "arxiv",
    32768: "pg19",
    60000: "pg19",
}


def fetch_filler(target_tokens, source=None):
    """Fetch real text from HuggingFace. No repetition.

    Sources:
    - cnn_dailymail: news articles (1K-4K)
    - arxiv: scientific papers (8K-16K)
    - pg19: public domain books (32K-60K)
    """
    from datasets import load_dataset
    target_chars = target_tokens * 4

    if source is None:
        source = TEXT_SOURCES.get(target_tokens, "cnn_dailymail")

    if source == "cnn_dailymail":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
        parts = []
        total = 0
        for sample in ds:
            parts.append(sample["article"])
            total += len(sample["article"])
            if total >= target_chars:
                break
        return "\n\n".join(parts)[:target_chars]

    elif source == "arxiv":
        ds = load_dataset("ccdv/arxiv-summarization", split="test", streaming=True)
        parts = []
        total = 0
        for sample in ds:
            parts.append(sample["article"])
            total += len(sample["article"])
            if total >= target_chars:
                break
        return "\n\n".join(parts)[:target_chars]

    elif source == "pg19":
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        for sample in ds:
            if len(sample["text"]) >= target_chars:
                # Take a chunk from the middle (skip headers/TOC)
                start = len(sample["text"]) // 4
                return sample["text"][start:start + target_chars]
        # Fallback: concatenate multiple books
        parts = []
        total = 0
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        for sample in ds:
            parts.append(sample["text"])
            total += len(sample["text"])
            if total >= target_chars:
                break
        return "\n\n".join(parts)[:target_chars]

    raise ValueError(f"Unknown source: {source}")


def insert_needle(text, needle, position, distractor=None):
    """Insert needle (and optional distractor) into text at specified position."""
    paras = text.split("\n\n")
    if len(paras) < 3:
        paras = text.split("\n")

    # Insert needle
    idx = max(1, min(int(len(paras) * position), len(paras) - 1))
    paras.insert(idx, needle)

    # Insert distractor at an earlier position (before the needle)
    if distractor:
        dist_idx = max(1, int(len(paras) * max(0, position - 0.3)))
        paras.insert(dist_idx, distractor)

    return "\n\n".join(paras)


def run_single(model, tokenizer, test_info, filler_cache, kv_mode, bits):
    """Run one needle test with real MLX inference."""
    ctx_label = test_info["context"]
    filler = filler_cache[test_info["tokens"]]
    doc = insert_needle(
        filler, test_info["needle"], test_info["position"],
        test_info.get("distractor"),
    )

    prompt = (
        f"Read the following document carefully. Answer the question at the end "
        f"using ONLY information found in the document. Be specific and include "
        f"exact numbers, names, and dates from the document.\n\n"
        f"--- DOCUMENT START ---\n{doc}\n--- DOCUMENT END ---\n\n"
        f"Question: {test_info['question']}\nAnswer:"
    )

    ids = mx.array(tokenizer.encode(prompt))
    prompt_tokens = len(ids)

    # Create fresh cache
    cache = make_prompt_cache(model)

    # Prefill
    t_prefill_start = time.time()
    if prompt_tokens > 2048:
        logits = chunked_prefill(model, ids, cache, chunk_size=2048)
    else:
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
    t_prefill = time.time() - t_prefill_start

    # Compress KV cache with TurboQuant (skip for FP16 baseline)
    compress_result = None
    t_compress = 0
    if kv_mode != "fp16":
        t_comp_start = time.time()
        compress_result = compress_cache(cache, model=model, bits=bits)
        t_compress = time.time() - t_comp_start

    # Generate
    t_gen_start = time.time()
    tokens_generated = []
    for _ in range(150):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        tokens_generated.append(tok_id)
        logits = model(next_token[:, None], cache=cache)
        mx.eval(logits)
    t_gen = time.time() - t_gen_start

    answer = tokenizer.decode(tokens_generated)
    gen_count = len(tokens_generated)
    wall_time = t_prefill + t_compress + t_gen
    found = test_info["check"](answer)

    return {
        "found": found,
        "answer_preview": answer[:300],
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_count,
        "wall_time_s": round(wall_time, 2),
        "prefill_s": round(t_prefill, 2),
        "compress_s": round(t_compress, 2),
        "gen_s": round(t_gen, 2),
        "prompt_tps": round(prompt_tokens / t_prefill, 1) if t_prefill > 0 else 0,
        "gen_tps": round(gen_count / t_gen, 1) if t_gen > 0 else 0,
        "ttft_ms": round(t_prefill * 1000, 0),
        "compress_info": compress_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack v2: MLX + TurboQuant")
    parser.add_argument("--model",
                        default="/Users/pradip/Desktop/Learning/Claude/PrismML/Bonsai-demo/models/Bonsai-8B-mlx",
                        help="HuggingFace model or local path (supports 1-bit Bonsai)")
    parser.add_argument("--contexts", nargs="+",
                        default=["1K", "4K", "8K", "16K", "32K", "60K"],
                        help="Context sizes to test")
    parser.add_argument("--kv-configs", nargs="+",
                        default=["fp16", "turboquant_4bit", "turboquant_3bit"],
                        help="KV cache configs to test")
    parser.add_argument("--output", default="benchmarks/needle_v2_mlx_turboquant_results.json")
    args = parser.parse_args()

    model_name = args.model
    selected_contexts = set(args.contexts)

    # Map kv-config names to bits
    kv_config_map = {
        "fp16": 0,
        "turboquant_4bit": 4,
        "turboquant_3bit": 3,
    }
    kv_configs = [(name, kv_config_map[name]) for name in args.kv_configs]

    print("=" * 70)
    print("Needle-in-a-Haystack v2: MLX + TurboQuant (Real Inference)")
    print(f"Model: {model_name}")
    print(f"Contexts: {', '.join(sorted(selected_contexts))}")
    print(f"KV configs: {', '.join(args.kv_configs)}")
    print(f"Date: {datetime.now().isoformat()}")
    print("Hardware: Apple M2 Pro, 16 GB")
    print("=" * 70)

    # Filter tests by selected contexts
    active_tests = [t for t in TESTS if t["context"] in selected_contexts]

    # Load model (auto-detects 1-bit Bonsai vs standard models)
    print("\nLoading model...", flush=True)
    model, tokenizer = load_model_auto(model_name)
    print(f"Model loaded.\n")

    # Fetch real text for each context size
    print("Fetching real text from HuggingFace...")
    filler_cache = {}
    needed_sizes = sorted(set(t["tokens"] for t in active_tests))
    for size in needed_sizes:
        source = TEXT_SOURCES.get(size, "cnn_dailymail")
        label = f"{size//1024}K" if size >= 1024 else f"{size}"
        print(f"  {label} context ({source})...", end=" ", flush=True)
        filler_cache[size] = fetch_filler(size, source)
        print(f"got ~{len(filler_cache[size])//4} tokens")

    results = {
        "benchmark": "Needle-in-a-Haystack v2 (MLX + TurboQuant real inference)",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hardware": "Apple M2 Pro, 16 GB",
        "design": {
            "text_sources": {
                "1K-4K": "CNN/DailyMail (real news articles)",
                "8K-16K": "ArXiv (real scientific papers)",
                "32K-60K": "PG19 (real public domain books)",
            },
            "needles_per_context": "3 (easy, medium, hard)",
            "total_needles": len(active_tests),
            "distractor_facts": "present in all hard questions",
            "verification": "programmatic lambda checks on specific details",
        },
        "tests": [],
        "summary": {},
        "scores": {},
    }

    total = len(active_tests) * len(kv_configs)
    current = 0

    for kv_name, bits in kv_configs:
        print(f"\n{'='*60}")
        print(f"  {kv_name}")
        print(f"{'='*60}")
        config_tests = []
        scores_key = f"kv_{kv_name}"
        results["scores"][scores_key] = {}

        for test in active_tests:
            ctx = test["context"]
            diff = test["difficulty"]

            # Skip FP16 at 60K (won't fit in 16GB with Bonsai-8B)
            if kv_name == "fp16" and ctx == "60K":
                current += 1
                print(f"[{current}/{total}] SKIP: {kv_name} @ {ctx} (FP16 at 60K won't fit in 16GB)")
                results["tests"].append({
                    "kv_config": kv_name, "context": ctx, "difficulty": diff,
                    "status": "SKIPPED", "found": False,
                })
                continue

            current += 1
            print(f"[{current}/{total}] {kv_name} | {ctx} | {diff}", end=" ... ", flush=True)

            try:
                r = run_single(model, tokenizer, test, filler_cache, kv_name, bits)
                r["kv_config"] = kv_name
                r["context"] = ctx
                r["difficulty"] = diff
                r["status"] = "OK"
                config_tests.append(r)
                results["tests"].append(r)

                status = "FOUND" if r["found"] else "MISSED"
                print(f"{status} | wall={r['wall_time_s']}s PP={r['prompt_tps']}tok/s Gen={r['gen_tps']}tok/s TTFT={r['ttft_ms']}ms")
                if r.get("compress_info"):
                    ci = r["compress_info"]
                    print(f"         compress={ci.get('compress_ms',0)}ms cos={ci.get('cosine','?')} "
                          f"{ci.get('original_mb',0):.0f}MB->{ci.get('compressed_mb',0):.0f}MB ({ci.get('ratio','?')}x)")

            except Exception as e:
                print(f"ERROR: {e}")
                results["tests"].append({
                    "kv_config": kv_name, "context": ctx, "difficulty": diff,
                    "status": f"ERROR: {e}", "found": False,
                })

        # Per-config scores by context
        ok = [t for t in config_tests if t.get("status") == "OK"]
        for ctx_label in selected_contexts:
            ctx_tests = [t for t in ok if t["context"] == ctx_label]
            if ctx_tests:
                found_count = sum(1 for t in ctx_tests if t["found"])
                results["scores"][scores_key][ctx_label] = {
                    "found": found_count,
                    "total": len(ctx_tests),
                    "pct": round(100 * found_count / len(ctx_tests), 1),
                    "avg_wall_s": round(sum(t["wall_time_s"] for t in ctx_tests) / len(ctx_tests), 2),
                }

        # Per-config summary
        if ok:
            found_count = sum(1 for t in ok if t["found"])
            results["summary"][kv_name] = {
                "accuracy": f"{found_count}/{len(ok)}",
                "accuracy_pct": round(100 * found_count / len(ok), 1),
                "avg_wall_s": round(sum(t["wall_time_s"] for t in ok) / len(ok), 2),
                "avg_pp_tps": round(sum(t["prompt_tps"] for t in ok) / len(ok), 1),
                "avg_gen_tps": round(sum(t["gen_tps"] for t in ok) / len(ok), 1),
                "avg_ttft_ms": round(sum(t["ttft_ms"] for t in ok) / len(ok), 0),
                "by_difficulty": {},
                "by_context": {},
            }
            for diff in ["easy", "medium", "hard"]:
                diff_tests = [t for t in ok if t["difficulty"] == diff]
                if diff_tests:
                    results["summary"][kv_name]["by_difficulty"][diff] = {
                        "found": sum(1 for t in diff_tests if t["found"]),
                        "total": len(diff_tests),
                        "pct": round(100 * sum(1 for t in diff_tests if t["found"]) / len(diff_tests), 1),
                    }
            for ctx_label in sorted(selected_contexts):
                ctx_tests = [t for t in ok if t["context"] == ctx_label]
                if ctx_tests:
                    results["summary"][kv_name]["by_context"][ctx_label] = {
                        "found": sum(1 for t in ctx_tests if t["found"]),
                        "total": len(ctx_tests),
                        "avg_wall_s": round(sum(t["wall_time_s"] for t in ctx_tests) / len(ctx_tests), 2),
                        "avg_pp_tps": round(sum(t["prompt_tps"] for t in ctx_tests) / len(ctx_tests), 1),
                        "avg_gen_tps": round(sum(t["gen_tps"] for t in ctx_tests) / len(ctx_tests), 1),
                    }

    # Save
    outfile = args.output
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Needle-in-a-Haystack v2")
    print("=" * 70)
    for kv_name, s in results["summary"].items():
        print(f"\n  {kv_name}: {s['accuracy']} ({s['accuracy_pct']}%)")
        print(f"    wall={s['avg_wall_s']}s  PP={s['avg_pp_tps']}tok/s  Gen={s['avg_gen_tps']}tok/s  TTFT={s['avg_ttft_ms']}ms")
        if "by_difficulty" in s:
            print(f"    By difficulty:")
            for diff, d in s["by_difficulty"].items():
                print(f"      {diff}: {d['found']}/{d['total']} ({d['pct']}%)")
        if "by_context" in s:
            print(f"    By context:")
            for ctx, c in s["by_context"].items():
                print(f"      {ctx}: {c['found']}/{c['total']} | wall={c['avg_wall_s']}s PP={c['avg_pp_tps']}tok/s")


if __name__ == "__main__":
    main()
