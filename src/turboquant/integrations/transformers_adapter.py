"""
HuggingFace Transformers integration.

Provides utilities to:
  - Extract KV caches from HuggingFace models
  - Compress/decompress KV caches with TurboQuant
  - Benchmark compression quality on real model KV caches
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..compressor import CompressedKVCache, TurboQuant
from ..metrics import measure_distortion


def extract_kv_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Run a forward pass and extract the KV cache tensors.

    Args:
        model: A HuggingFace CausalLM model.
        tokenizer: The corresponding tokenizer.
        prompt: Text prompt to run through the model.
        device: Device override (defaults to model.device).

    Returns:
        Tuple of (past_key_values, inputs) where past_key_values is a tuple
        of (key, value) tensors per layer.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    target_device = device or str(next(model.parameters()).device)
    inputs = tokenizer(text, return_tensors="pt").to(target_device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    return outputs.past_key_values, inputs


def compress_kv_cache(
    kv_cache: Any,
    bits: int = 4,
    head_dim: Optional[int] = None,
) -> List[Tuple[CompressedKVCache, CompressedKVCache]]:
    """
    Compress an entire KV cache (all layers) with TurboQuant.

    Args:
        kv_cache: Tuple of (key, value) tensors per layer from a HF model.
        bits: Quantization bits (2, 3, or 4).
        head_dim: Head dimension (auto-detected from tensors if not given).

    Returns:
        List of (compressed_key, compressed_value) per layer.
    """
    compressed_layers = []
    for layer_idx, (key, value) in enumerate(kv_cache):
        dim = head_dim or key.shape[-1]
        tq = TurboQuant(bits=bits, head_dim=dim, seed=42 + layer_idx)
        tq_v = TurboQuant(bits=bits, head_dim=dim, seed=1000 + layer_idx)

        comp_k = tq.compress(key.cpu())
        comp_v = tq_v.compress(value.cpu())
        compressed_layers.append((comp_k, comp_v))

    return compressed_layers


def decompress_kv_cache(
    compressed_layers: List[Tuple[CompressedKVCache, CompressedKVCache]],
    bits: int = 4,
    head_dim: Optional[int] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Decompress an entire KV cache (all layers).

    Args:
        compressed_layers: Output of compress_kv_cache().
        bits: Must match the bits used for compression.
        head_dim: Must match the head_dim used for compression.

    Returns:
        List of (key, value) tensor tuples per layer.
    """
    result = []
    for layer_idx, (comp_k, comp_v) in enumerate(compressed_layers):
        dim = head_dim or comp_k.shape[-1]
        tq = TurboQuant(bits=bits, head_dim=dim, seed=42 + layer_idx)
        tq_v = TurboQuant(bits=bits, head_dim=dim, seed=1000 + layer_idx)

        key = tq.decompress(comp_k)
        value = tq_v.decompress(comp_v)
        result.append((key, value))

    return result


def benchmark_kv_compression(
    kv_cache: Any,
    bits_list: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark TurboQuant compression quality on a real KV cache.

    Args:
        kv_cache: Tuple of (key, value) tensors per layer from a HF model.
        bits_list: List of bit widths to test. Default: [2, 3, 4].

    Returns:
        Dict mapping bits -> quality metrics.
    """
    if bits_list is None:
        bits_list = [2, 3, 4]

    results = {}
    for bits in bits_list:
        k_cos_list, v_cos_list, ip_list = [], [], []
        total_orig, total_comp = 0, 0

        for layer_idx, (key, value) in enumerate(kv_cache):
            key_cpu = key.cpu()
            value_cpu = value.cpu()
            head_dim = key_cpu.shape[-1]

            tq_k = TurboQuant(bits=bits, head_dim=head_dim, seed=42 + layer_idx)
            tq_v = TurboQuant(bits=bits, head_dim=head_dim, seed=1000 + layer_idx)

            comp_k = tq_k.compress(key_cpu)
            recon_k = tq_k.decompress(comp_k)
            comp_v = tq_v.compress(value_cpu)
            recon_v = tq_v.decompress(comp_v)

            km = measure_distortion(key_cpu, recon_k)
            vm = measure_distortion(value_cpu, recon_v)

            k_cos_list.append(km["cosine_similarity_mean"])
            v_cos_list.append(vm["cosine_similarity_mean"])
            ip_list.append(km["inner_product_correlation"])

            orig = (key_cpu.numel() + value_cpu.numel()) * 2
            num_kv_heads = key_cpu.shape[1]
            comp = (
                (key_cpu.numel() + value_cpu.numel()) * bits / 8
                + (key_cpu.shape[2] + value_cpu.shape[2]) * num_kv_heads * 2
            )
            total_orig += orig
            total_comp += comp

        n = len(k_cos_list)
        results[bits] = {
            "compression_ratio": round(total_orig / total_comp, 1) if total_comp else 0,
            "key_cosine_mean": round(sum(k_cos_list) / n, 4),
            "key_cosine_min": round(min(k_cos_list), 4),
            "value_cosine_mean": round(sum(v_cos_list) / n, 4),
            "ip_correlation": round(sum(ip_list) / n, 4),
            "original_mb": round(total_orig / 1024 / 1024, 2),
            "compressed_mb": round(total_comp / 1024 / 1024, 2),
            "num_layers": n,
        }

    return results


def get_model_kv_config(model: Any) -> Dict[str, int]:
    """
    Extract KV-cache-relevant config from a HuggingFace model.

    Returns:
        Dict with head_dim, num_kv_heads, num_layers, fp16_kv_bytes_per_token.
    """
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    num_kv_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    num_layers = config.num_hidden_layers
    fp16_kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

    return {
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "fp16_kv_bytes_per_token": fp16_kv_per_token,
    }
