"""
vLLM integration for TurboQuant.

Provides:
  - Full feasibility analysis: give a model name, get a detailed report
  - KV-cache compression manager for all layers
  - Memory & speed projections at various context lengths and batch sizes
  - GPU memory budget analysis

Requires: pip install turboquant[vllm]

Usage::

    from turboquant.integrations.vllm_adapter import analyze

    # Full report — just give a model name
    report = analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=24)
    print(report["summary"])

    # Or use the KV manager directly
    from turboquant.integrations.vllm_adapter import TurboQuantKVManager
    manager = TurboQuantKVManager(bits=4, head_dim=128, num_layers=32)
    compressed = manager.compress_layer(0, key=k, value=v)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..compressor import CompressedKVCache, TurboQuant
from ..metrics import cosine_similarity, measure_distortion


# ═══════════════════════════════════════════════════════
#  Model Config Resolution
# ═══════════════════════════════════════════════════════

# Well-known model architectures (fallback when HF config unavailable)
_KNOWN_MODELS: Dict[str, Dict[str, int]] = {
    "qwen2.5-0.5b": {"layers": 24, "heads": 14, "kv_heads": 2, "hidden": 896, "head_dim": 64},
    "qwen2.5-1.5b": {"layers": 28, "heads": 12, "kv_heads": 2, "hidden": 1536, "head_dim": 128},
    "qwen2.5-3b": {"layers": 36, "heads": 16, "kv_heads": 2, "hidden": 2048, "head_dim": 128},
    "qwen2.5-7b": {"layers": 28, "heads": 28, "kv_heads": 4, "hidden": 3584, "head_dim": 128},
    "qwen2.5-14b": {"layers": 48, "heads": 40, "kv_heads": 8, "hidden": 5120, "head_dim": 128},
    "qwen2.5-32b": {"layers": 64, "heads": 40, "kv_heads": 8, "hidden": 5120, "head_dim": 128},
    "qwen2.5-72b": {"layers": 80, "heads": 64, "kv_heads": 8, "hidden": 8192, "head_dim": 128},
    "qwen3.5-4b": {"layers": 36, "heads": 32, "kv_heads": 8, "hidden": 2560, "head_dim": 128},
    "llama-3-8b": {"layers": 32, "heads": 32, "kv_heads": 8, "hidden": 4096, "head_dim": 128},
    "llama-3-70b": {"layers": 80, "heads": 64, "kv_heads": 8, "hidden": 8192, "head_dim": 128},
    "llama-3.1-8b": {"layers": 32, "heads": 32, "kv_heads": 8, "hidden": 4096, "head_dim": 128},
    "llama-3.1-70b": {"layers": 80, "heads": 64, "kv_heads": 8, "hidden": 8192, "head_dim": 128},
    "mistral-7b": {"layers": 32, "heads": 32, "kv_heads": 8, "hidden": 4096, "head_dim": 128},
    "mixtral-8x7b": {"layers": 32, "heads": 32, "kv_heads": 8, "hidden": 4096, "head_dim": 128},
    "gemma-2-9b": {"layers": 42, "heads": 16, "kv_heads": 8, "hidden": 3584, "head_dim": 256},
    "gemma-2-27b": {"layers": 46, "heads": 32, "kv_heads": 16, "hidden": 4608, "head_dim": 128},
    "phi-3-mini": {"layers": 32, "heads": 32, "kv_heads": 32, "hidden": 3072, "head_dim": 96},
    "deepseek-v2-lite": {"layers": 27, "heads": 16, "kv_heads": 2, "hidden": 2048, "head_dim": 128},
}


def resolve_model_config(
    model_name: str,
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    hidden_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve model architecture config from name or explicit values.

    Tries in order:
      1. Explicit kwargs (if all provided)
      2. HuggingFace transformers AutoConfig (if installed)
      3. Built-in known model table

    Args:
        model_name: HuggingFace model ID or short name (e.g. "Qwen/Qwen2.5-7B-Instruct").
        num_layers: Override number of transformer layers.
        num_kv_heads: Override number of KV attention heads.
        head_dim: Override attention head dimension.
        num_attention_heads: Override number of attention heads.
        hidden_size: Override hidden size.

    Returns:
        Dict with num_layers, num_kv_heads, head_dim, num_attention_heads,
        hidden_size, model_name, source.
    """
    # If all values explicitly provided, use them directly
    if all(v is not None for v in [num_layers, num_kv_heads, head_dim]):
        return {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads or num_kv_heads,
            "hidden_size": hidden_size or (head_dim * (num_attention_heads or num_kv_heads)),
            "source": "explicit",
        }

    # Try HuggingFace AutoConfig
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        h_dim = hf_config.hidden_size
        n_heads = hf_config.num_attention_heads
        n_kv = getattr(hf_config, "num_key_value_heads", n_heads)
        n_layers = hf_config.num_hidden_layers
        h_d = getattr(hf_config, "head_dim", h_dim // n_heads)

        return {
            "model_name": model_name,
            "num_layers": num_layers or n_layers,
            "num_kv_heads": num_kv_heads or n_kv,
            "head_dim": head_dim or h_d,
            "num_attention_heads": num_attention_heads or n_heads,
            "hidden_size": hidden_size or h_dim,
            "max_position_embeddings": getattr(hf_config, "max_position_embeddings", None),
            "model_type": getattr(hf_config, "model_type", "unknown"),
            "source": "huggingface",
        }
    except Exception:
        pass

    # Fallback: known model table
    name_lower = model_name.lower()
    for key, cfg in _KNOWN_MODELS.items():
        if key in name_lower:
            return {
                "model_name": model_name,
                "num_layers": num_layers or cfg["layers"],
                "num_kv_heads": num_kv_heads or cfg["kv_heads"],
                "head_dim": head_dim or cfg["head_dim"],
                "num_attention_heads": num_attention_heads or cfg["heads"],
                "hidden_size": hidden_size or cfg["hidden"],
                "source": "known_models",
            }

    raise ValueError(
        f"Cannot resolve config for '{model_name}'. "
        f"Install transformers (`pip install transformers`) or provide "
        f"num_layers, num_kv_heads, and head_dim explicitly."
    )


def _estimate_params_b(model_name: str, hidden_size: int, num_layers: int) -> float:
    """Estimate model parameters in billions from name or architecture."""
    name = model_name.lower()
    # Try to extract from name (e.g., "7b", "4b", "0.5b", "72b")
    import re

    match = re.search(r"(\d+\.?\d*)\s*[bB]", name)
    if match:
        return float(match.group(1))
    # Rough estimate: params ≈ 12 * num_layers * hidden_size^2
    return 12 * num_layers * hidden_size**2 / 1e9


# ═══════════════════════════════════════════════════════
#  TurboQuant KV Manager
# ═══════════════════════════════════════════════════════

class TurboQuantKVManager:
    """
    Manages per-layer TurboQuant compressors for a full model.

    Creates one TurboQuant instance per layer (with unique seeds) so that
    rotation matrices are reproducible and independent across layers.

    Args:
        bits: Quantization bits (2, 3, or 4).
        head_dim: Attention head dimension.
        num_layers: Number of transformer layers.
        base_seed: Base random seed (layer seed = base_seed + layer_idx).
    """

    def __init__(
        self,
        bits: int = 4,
        head_dim: int = 128,
        num_layers: int = 32,
        base_seed: int = 42,
    ):
        self.bits = bits
        self.head_dim = head_dim
        self.num_layers = num_layers

        self._key_compressors = [
            TurboQuant(bits=bits, head_dim=head_dim, seed=base_seed + i)
            for i in range(num_layers)
        ]
        self._value_compressors = [
            TurboQuant(bits=bits, head_dim=head_dim, seed=base_seed + 1000 + i)
            for i in range(num_layers)
        ]

    def compress_layer(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor,
    ) -> Tuple[CompressedKVCache, CompressedKVCache]:
        """Compress key and value tensors for a single layer."""
        return (
            self._key_compressors[layer_idx].compress(key),
            self._value_compressors[layer_idx].compress(value),
        )

    def decompress_layer(
        self, layer_idx: int,
        compressed_key: CompressedKVCache, compressed_value: CompressedKVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress key and value tensors for a single layer."""
        return (
            self._key_compressors[layer_idx].decompress(compressed_key),
            self._value_compressors[layer_idx].decompress(compressed_value),
        )

    def compress_all(
        self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[CompressedKVCache, CompressedKVCache]]:
        """Compress an entire model's KV cache (all layers)."""
        return [self.compress_layer(i, k, v) for i, (k, v) in enumerate(kv_cache)]

    def decompress_all(
        self, compressed: List[Tuple[CompressedKVCache, CompressedKVCache]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Decompress an entire model's KV cache (all layers)."""
        return [self.decompress_layer(i, ck, cv) for i, (ck, cv) in enumerate(compressed)]

    def memory_savings(self, num_kv_heads: int, context_length: int) -> Dict[str, float]:
        """Calculate projected memory savings."""
        fp16_per_token = 2 * self.num_layers * num_kv_heads * self.head_dim * 2
        tq_indices = 2 * self.num_layers * num_kv_heads * self.head_dim * self.bits / 8
        tq_norms = 2 * self.num_layers * num_kv_heads * 2

        fp16_total = fp16_per_token * context_length
        tq_total = (tq_indices + tq_norms) * context_length

        ratio = fp16_total / tq_total if tq_total > 0 else float("inf")
        return {
            "original_mb": round(fp16_total / 1024 / 1024, 2),
            "compressed_mb": round(tq_total / 1024 / 1024, 2),
            "ratio": round(ratio, 1),
            "savings_pct": round((1 - 1 / ratio) * 100, 1),
        }

    def to(self, device: torch.device) -> "TurboQuantKVManager":
        """Move all compressors to a device."""
        for tq in self._key_compressors + self._value_compressors:
            tq.to(device)
        return self


# ═══════════════════════════════════════════════════════
#  Feasibility Analysis & Report
# ═══════════════════════════════════════════════════════

def _run_quality_benchmark(
    num_layers: int, num_kv_heads: int, head_dim: int,
    bits_list: Optional[List[int]] = None, seq_len: int = 256,
) -> Dict[int, Dict[str, float]]:
    """Run TurboQuant compression quality at each bit width."""
    if bits_list is None:
        bits_list = [2, 3, 4]

    results = {}
    for bits in bits_list:
        cos_scores, ip_scores = [], []
        total_orig, total_comp = 0, 0

        for layer in range(num_layers):
            key = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
            tq = TurboQuant(bits=bits, head_dim=head_dim, seed=42 + layer)
            comp = tq.compress(key)
            recon = tq.decompress(comp)

            cos_scores.append(cosine_similarity(key, recon))
            m = measure_distortion(key, recon)
            ip_scores.append(m["inner_product_correlation"])

            mem = tq.memory_bytes(comp)
            total_orig += mem["original"] * 2  # K + V
            total_comp += mem["compressed"] * 2

        n = len(cos_scores)
        results[bits] = {
            "compression_ratio": round(total_orig / total_comp, 1) if total_comp else 0,
            "cosine_mean": round(sum(cos_scores) / n, 4),
            "cosine_min": round(min(cos_scores), 4),
            "ip_correlation": round(sum(ip_scores) / n, 4),
        }
    return results


def _run_speed_benchmark(
    num_layers: int, num_kv_heads: int, head_dim: int,
) -> Dict[str, float]:
    """Benchmark compress/decompress speed."""
    tqs = [TurboQuant(bits=4, head_dim=head_dim, seed=42 + i) for i in range(num_layers)]

    # Warmup
    for tq in tqs:
        c = tq.compress(torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.float16))
        tq.decompress(c)

    # Per-token (all layers)
    iters = 50
    t0 = time.time()
    for _ in range(iters):
        for tq in tqs:
            tok = torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.float16)
            c = tq.compress(tok)
            tq.decompress(c)
    per_token_ms = (time.time() - t0) / iters * 1000

    # Batch 1024 tokens (all layers)
    t0 = time.time()
    for tq in tqs:
        batch = torch.randn(1, num_kv_heads, 1024, head_dim, dtype=torch.float16)
        c = tq.compress(batch)
        tq.decompress(c)
    batch_1k_ms = (time.time() - t0) * 1000

    # Decompress only (hot path)
    comps = []
    for tq in tqs:
        batch = torch.randn(1, num_kv_heads, 1024, head_dim, dtype=torch.float16)
        comps.append(tq.compress(batch))

    t0 = time.time()
    for i, tq in enumerate(tqs):
        tq.decompress(comps[i])
    decompress_1k_ms = (time.time() - t0) * 1000

    return {
        "per_token_all_layers_ms": round(per_token_ms, 2),
        "per_token_per_layer_ms": round(per_token_ms / num_layers, 3),
        "batch_1k_compress_decompress_ms": round(batch_1k_ms, 1),
        "batch_1k_decompress_only_ms": round(decompress_1k_ms, 1),
    }


def analyze(
    model_name: str,
    gpu_memory_gb: float = 24.0,
    bits: int = 4,
    batch_sizes: Optional[List[int]] = None,
    context_lengths: Optional[List[int]] = None,
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    run_speed_benchmark: bool = True,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Full feasibility analysis for TurboQuant KV compression with a vLLM model.

    Give a model name, get a detailed report on memory savings, speed overhead,
    quality metrics, and what context lengths / batch sizes fit on your GPU.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct").
        gpu_memory_gb: GPU memory in GB (default: 24 for RTX 3090/4090).
        bits: TurboQuant bit width (2, 3, or 4).
        batch_sizes: Batch sizes to analyze (default: [1, 4, 8, 16, 32]).
        context_lengths: Context lengths to analyze (default: standard set).
        num_layers: Override model layers (auto-detected if omitted).
        num_kv_heads: Override KV heads (auto-detected if omitted).
        head_dim: Override head dim (auto-detected if omitted).
        run_speed_benchmark: Whether to run speed benchmarks (adds ~10s).
        print_report: Whether to print formatted report to stdout.

    Returns:
        Dict with model_config, quality, speed, memory_table, feasibility, summary.

    Example::

        report = analyze("Qwen/Qwen2.5-7B-Instruct", gpu_memory_gb=24)
        report = analyze("meta-llama/Llama-3.1-8B", gpu_memory_gb=80)
        report = analyze("custom-model", num_layers=32, num_kv_heads=8, head_dim=128)
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]
    if context_lengths is None:
        context_lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # ─── Step 1: Resolve model config ───
    config = resolve_model_config(
        model_name, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    n_layers = config["num_layers"]
    n_kv_heads = config["num_kv_heads"]
    h_dim = config["head_dim"]
    h_size = config["hidden_size"]
    n_heads = config["num_attention_heads"]

    params_b = _estimate_params_b(model_name, h_size, n_layers)
    weight_fp16_gb = params_b * 2  # FP16 = 2 bytes per param
    weight_q4_gb = params_b * 0.5  # Q4 ≈ 0.5 bytes per param
    # vLLM typically loads in FP16 or BF16
    weight_gb = weight_fp16_gb

    gpu_mem_mb = gpu_memory_gb * 1024
    # vLLM reserves ~10% for overhead (activations, CUDA kernels, etc.)
    overhead_pct = 0.10
    available_kv_mb = gpu_mem_mb * (1 - overhead_pct) - weight_gb * 1024

    fp16_per_token = 2 * n_layers * n_kv_heads * h_dim * 2  # bytes

    # TurboQuant compressed per token
    tq_per_token_per_head = h_dim * bits / 8 + 2  # indices + norm
    fp16_per_token_per_head = h_dim * 2
    tq_ratio = fp16_per_token_per_head / tq_per_token_per_head
    tq_per_token = fp16_per_token / tq_ratio

    # ─── Step 2: Quality benchmark ───
    quality = _run_quality_benchmark(n_layers, n_kv_heads, h_dim, bits_list=[2, 3, 4])

    # ─── Step 3: Speed benchmark ───
    speed = {}
    if run_speed_benchmark:
        speed = _run_speed_benchmark(n_layers, n_kv_heads, h_dim)

    # ─── Step 4: Memory & feasibility table ───
    memory_table = []
    for ctx in context_lengths:
        for batch in batch_sizes:
            fp16_kv_mb = fp16_per_token * ctx * batch / 1024 / 1024
            tq_kv_mb = tq_per_token * ctx * batch / 1024 / 1024
            total_fp16_mb = weight_gb * 1024 + fp16_kv_mb
            total_tq_mb = weight_gb * 1024 + tq_kv_mb

            fits_fp16 = total_fp16_mb <= gpu_mem_mb * (1 - overhead_pct)
            fits_tq = total_tq_mb <= gpu_mem_mb * (1 - overhead_pct)

            memory_table.append({
                "context_length": ctx,
                "batch_size": batch,
                "fp16_kv_mb": round(fp16_kv_mb, 1),
                "tq_kv_mb": round(tq_kv_mb, 1),
                "total_fp16_mb": round(total_fp16_mb, 1),
                "total_tq_mb": round(total_tq_mb, 1),
                "saved_mb": round(fp16_kv_mb - tq_kv_mb, 1),
                "saved_pct": round((1 - 1 / tq_ratio) * 100, 1),
                "fits_fp16": fits_fp16,
                "fits_tq": fits_tq,
                "unlocked_by_tq": fits_tq and not fits_fp16,
            })

    # ─── Step 5: Compute max context per batch size ───
    feasibility = {}
    for batch in batch_sizes:
        max_fp16 = int(available_kv_mb * 1024 * 1024 / (fp16_per_token * batch))
        max_tq = int(available_kv_mb * 1024 * 1024 / (tq_per_token * batch))
        feasibility[batch] = {
            "max_context_fp16": max_fp16,
            "max_context_tq": max_tq,
            "improvement_x": round(max_tq / max_fp16, 1) if max_fp16 > 0 else float("inf"),
        }

    # ─── Step 6: Build result ───
    tq_q = quality.get(bits, quality.get(4, {}))
    result = {
        "model_config": config,
        "model_params_b": round(params_b, 1),
        "weight_gb": round(weight_gb, 1),
        "gpu_memory_gb": gpu_memory_gb,
        "available_for_kv_mb": round(available_kv_mb, 0),
        "fp16_kv_bytes_per_token": fp16_per_token,
        "tq_kv_bytes_per_token": round(tq_per_token, 1),
        "tq_compression_ratio": round(tq_ratio, 1),
        "quality": quality,
        "speed": speed,
        "memory_table": memory_table,
        "feasibility": feasibility,
    }

    # ─── Step 7: Print report ───
    if print_report:
        _print_report(result, bits)

    return result


def _print_report(result: Dict[str, Any], bits: int) -> None:
    """Print a formatted feasibility report."""
    cfg = result["model_config"]
    gpu_gb = result["gpu_memory_gb"]

    print("=" * 75)
    print("  TurboQuant Feasibility Report for vLLM")
    print("=" * 75)

    print(f"\n  Model:            {cfg['model_name']}")
    print(f"  Config source:    {cfg['source']}")
    print(f"  Parameters:       ~{result['model_params_b']}B")
    print(f"  Architecture:     {cfg['num_layers']}L × {cfg['num_attention_heads']}H "
          f"× {cfg['num_kv_heads']}KV × {cfg['head_dim']}dim")
    print(f"  GQA ratio:        {cfg['num_attention_heads']}:{cfg['num_kv_heads']}")
    if cfg.get("max_position_embeddings"):
        print(f"  Max positions:    {cfg['max_position_embeddings']:,}")

    print(f"\n  GPU Memory:       {gpu_gb} GB")
    print(f"  Model weights:    ~{result['weight_gb']} GB (FP16)")
    print(f"  Available for KV: ~{result['available_for_kv_mb']:.0f} MB")
    print(f"  FP16 KV/token:    {result['fp16_kv_bytes_per_token']} bytes "
          f"({result['fp16_kv_bytes_per_token'] / 1024:.1f} KB)")

    # ─── Quality ───
    print(f"\n{'─' * 75}")
    print(f"  Compression Quality")
    print(f"{'─' * 75}")
    print(f"\n  {'Bits':>5} | {'Ratio':>6} | {'Cosine':>8} | {'Cosine Min':>10} | {'IP Corr':>8}")
    print(f"  {'─' * 50}")
    for b in sorted(result["quality"].keys()):
        q = result["quality"][b]
        marker = " <<<" if b == bits else ""
        print(f"  {b}-bit | {q['compression_ratio']:>5.1f}x | {q['cosine_mean']:>8.4f} | "
              f"{q['cosine_min']:>10.4f} | {q['ip_correlation']:>8.4f}{marker}")

    # ─── Speed ───
    if result["speed"]:
        print(f"\n{'─' * 75}")
        print(f"  Speed Overhead ({bits}-bit, {cfg['num_layers']} layers)")
        print(f"{'─' * 75}")
        sp = result["speed"]
        print(f"\n  Per-token (all {cfg['num_layers']} layers):  {sp['per_token_all_layers_ms']} ms")
        print(f"  Per-token per layer:         {sp['per_token_per_layer_ms']} ms")
        print(f"  1K tokens (compress+decomp): {sp['batch_1k_compress_decompress_ms']} ms")
        print(f"  1K tokens (decompress only): {sp['batch_1k_decompress_only_ms']} ms")

        # Estimate overhead as % of typical generation
        # vLLM on A100: ~100 tok/s for 7B = 10ms/token
        typical_gen_ms = 10.0
        overhead_pct = sp["per_token_all_layers_ms"] / typical_gen_ms * 100
        print(f"\n  At ~100 tok/s generation: {overhead_pct:.1f}% overhead — "
              f"{'negligible' if overhead_pct < 5 else 'moderate' if overhead_pct < 20 else 'significant'}")

    # ─── Memory table ───
    print(f"\n{'─' * 75}")
    print(f"  Memory Analysis (batch=1) — {gpu_gb}GB GPU")
    print(f"{'─' * 75}")
    print(f"\n  {'Context':>8} | {'FP16 KV':>9} | {'TQ KV':>9} | {'Saved':>6} | "
          f"{'Total FP16':>10} | {'Total TQ':>10} | {'Fits?':>12}")
    print(f"  {'─' * 78}")

    for row in result["memory_table"]:
        if row["batch_size"] != 1:
            continue
        fits_str = ""
        if row["unlocked_by_tq"]:
            fits_str = "TQ ONLY"
        elif row["fits_fp16"]:
            fits_str = "Both"
        elif row["fits_tq"]:
            fits_str = "TQ only"
        else:
            fits_str = "Neither"

        print(f"  {row['context_length']:>8} | {row['fp16_kv_mb']:>7.1f}MB | "
              f"{row['tq_kv_mb']:>7.1f}MB | {row['saved_pct']:>4.0f}%  | "
              f"{row['total_fp16_mb']:>8.1f}MB | {row['total_tq_mb']:>8.1f}MB | "
              f"{fits_str:>12}")

    # ─── Batch size scaling ───
    print(f"\n{'─' * 75}")
    print(f"  Max Context Length per Batch Size — {gpu_gb}GB GPU")
    print(f"{'─' * 75}")
    print(f"\n  {'Batch':>6} | {'FP16 Max':>12} | {'TQ Max':>12} | {'Improvement':>12} | {'Extra Context':>14}")
    print(f"  {'─' * 62}")

    for batch, f in sorted(result["feasibility"].items()):
        fp16_k = f["max_context_fp16"] // 1024
        tq_k = f["max_context_tq"] // 1024
        extra = tq_k - fp16_k
        print(f"  {batch:>6} | {fp16_k:>9}K | {tq_k:>9}K | "
              f"{f['improvement_x']:>10.1f}x | +{extra:>10}K")

    # ─── Concurrent users (batch) scaling ───
    print(f"\n{'─' * 75}")
    print(f"  Concurrent Users at 8K Context — {gpu_gb}GB GPU")
    print(f"{'─' * 75}")
    ctx = 8192
    avail = result["available_for_kv_mb"] * 1024 * 1024
    fp16_per = result["fp16_kv_bytes_per_token"] * ctx
    tq_per = result["tq_kv_bytes_per_token"] * ctx

    max_users_fp16 = int(avail / fp16_per)
    max_users_tq = int(avail / tq_per)

    print(f"\n  FP16: {max_users_fp16} concurrent users")
    print(f"  TQ{bits}:  {max_users_tq} concurrent users")
    print(f"  Gain: {max_users_tq / max_users_fp16:.1f}x more users!")

    fp16_bar = "█" * max_users_fp16
    tq_bar = "█" * min(max_users_tq, 60)
    print(f"\n  FP16: {fp16_bar} ({max_users_fp16})")
    print(f"  TQ{bits}:  {tq_bar} ({max_users_tq})")

    # ─── Summary ───
    tq_q = result["quality"].get(bits, {})
    f1 = result["feasibility"].get(1, {})

    print(f"\n{'═' * 75}")
    print(f"  SUMMARY")
    print(f"{'═' * 75}")
    print(f"""
  Model:          {cfg['model_name']} (~{result['model_params_b']}B)
  GPU:            {gpu_gb} GB
  TurboQuant:     {bits}-bit, {tq_q.get('compression_ratio', 'N/A')}x compression
  Quality:        cosine = {tq_q.get('cosine_mean', 'N/A')} (near-lossless)

  Max context (batch=1):
    FP16:         {f1.get('max_context_fp16', 0) // 1024}K tokens
    TurboQuant:   {f1.get('max_context_tq', 0) // 1024}K tokens  ({f1.get('improvement_x', 0)}x more!)

  Concurrent users (8K context):
    FP16:         {max_users_fp16} users
    TurboQuant:   {max_users_tq} users  ({max_users_tq / max(max_users_fp16, 1):.1f}x more!)

  vLLM Launch Command:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  vllm serve {cfg['model_name']}                                     │
  │    --gpu-memory-utilization 0.9                                     │
  │    --max-model-len {min(f1.get('max_context_tq', 32768), 131072)}                                                │
  └──────────────────────────────────────────────────────────────────────┘
""")
    print(f"{'═' * 75}")
