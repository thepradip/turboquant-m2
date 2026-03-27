"""
llama.cpp / GGUF integration for TurboQuant.

Provides utilities to:
  - Compare TurboQuant KV compression against standard GGUF Q4 KV quantization
  - Project memory savings for llama.cpp deployments
  - Benchmark the quality difference between methods

llama.cpp uses per-group INT4 quantization for KV cache (scale + zero per group).
TurboQuant uses rotation + Lloyd-Max with near-zero overhead.

Usage::

    from turboquant.integrations.llamacpp_adapter import compare_kv_methods

    results = compare_kv_methods(num_layers=28, num_kv_heads=4, head_dim=128)
    print(results)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from ..compressor import StandardQ4Quantizer, TurboQuant
from ..metrics import cosine_similarity, inner_product_correlation


def compare_kv_methods(
    num_layers: int = 28,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    seq_len: int = 256,
    group_size: int = 32,
    tq_bits: int = 4,
) -> Dict[str, Dict]:
    """
    Fair comparison between standard Q4 and TurboQuant at the same bit budget.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Attention head dimension.
        seq_len: Sequence length for test data.
        group_size: Group size for standard Q4 quantization.
        tq_bits: TurboQuant bit width.

    Returns:
        Dict with 'standard_q4' and 'turboquant' sub-dicts containing
        quality metrics, memory stats, and speed benchmarks.
    """
    std_q4 = StandardQ4Quantizer(group_size=group_size)
    tq = TurboQuant(bits=tq_bits, head_dim=head_dim)

    std_cos, std_ip = [], []
    tq_cos, tq_ip = [], []

    for _layer in range(num_layers):
        key = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float16)

        # Standard Q4
        s_comp = std_q4.compress(key)
        s_recon = std_q4.decompress(s_comp)
        std_cos.append(cosine_similarity(key, s_recon))
        std_ip.append(inner_product_correlation(key, s_recon))

        # TurboQuant
        t_comp = tq.compress(key)
        t_recon = tq.decompress(t_comp)
        tq_cos.append(cosine_similarity(key, t_recon))
        tq_ip.append(inner_product_correlation(key, t_recon))

    # Memory
    sample = torch.randn(1, 1, seq_len, head_dim, dtype=torch.float16)
    s_mem = std_q4.memory_bytes(std_q4.compress(sample))
    t_comp_sample = tq.compress(sample)
    t_mem = tq.memory_bytes(t_comp_sample)

    # Speed (10 iterations)
    bench_data = torch.randn(1, num_kv_heads, 1024, head_dim, dtype=torch.float16)

    t0 = time.time()
    for _ in range(10):
        c = std_q4.compress(bench_data)
        std_q4.decompress(c)
    std_time = (time.time() - t0) / 10 * 1000

    t0 = time.time()
    for _ in range(10):
        c = tq.compress(bench_data)
        tq.decompress(c)
    tq_time = (time.time() - t0) / 10 * 1000

    n = len(std_cos)
    return {
        "standard_q4": {
            "cosine_mean": round(sum(std_cos) / n, 4),
            "cosine_min": round(min(std_cos), 4),
            "ip_corr_mean": round(sum(std_ip) / n, 4),
            "actual_bits_per_element": s_mem["actual_bits_per_element"],
            "compression_ratio": round(s_mem["ratio"], 1),
            "overhead_pct": s_mem["overhead_pct"],
            "speed_ms_per_1k_tokens": round(std_time, 1),
        },
        "turboquant": {
            "cosine_mean": round(sum(tq_cos) / n, 4),
            "cosine_min": round(min(tq_cos), 4),
            "ip_corr_mean": round(sum(tq_ip) / n, 4),
            "bits": tq_bits,
            "compression_ratio": round(t_mem["ratio"], 1),
            "overhead_pct": round(t_mem["savings_pct"], 1),
            "speed_ms_per_1k_tokens": round(tq_time, 1),
        },
    }


def project_gguf_plus_tq_memory(
    model_params_b: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_lengths: Optional[List[int]] = None,
    tq_bits: int = 4,
) -> List[Dict]:
    """
    Project total memory for GGUF model weights + TurboQuant KV cache.

    Args:
        model_params_b: Model size in billions of parameters.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Attention head dimension.
        context_lengths: Context lengths to project.
        tq_bits: TurboQuant bit width.

    Returns:
        List of dicts per context length with memory breakdown.
    """
    if context_lengths is None:
        context_lengths = [4096, 16384, 32768, 65536, 131072, 262144]

    model_weight_mb = model_params_b * 1000 * 0.5  # Q4 ~ 0.5 GB/B
    fp16_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

    tq_per_token_per_head = head_dim * tq_bits / 8 + 2
    fp16_per_token_per_head = head_dim * 2
    tq_ratio = fp16_per_token_per_head / tq_per_token_per_head

    rows = []
    for ctx in context_lengths:
        fp16_kv_mb = fp16_per_token * ctx / 1024 / 1024
        tq_kv_mb = fp16_kv_mb / tq_ratio
        total_fp16 = model_weight_mb + fp16_kv_mb
        total_tq = model_weight_mb + tq_kv_mb

        rows.append({
            "context_length": ctx,
            "model_weight_mb": round(model_weight_mb, 0),
            "fp16_kv_mb": round(fp16_kv_mb, 1),
            "tq_kv_mb": round(tq_kv_mb, 1),
            "total_fp16_mb": round(total_fp16, 1),
            "total_tq_mb": round(total_tq, 1),
            "saved_mb": round(total_fp16 - total_tq, 1),
            "saved_pct": round((total_fp16 - total_tq) / total_fp16 * 100, 1),
        })

    return rows
