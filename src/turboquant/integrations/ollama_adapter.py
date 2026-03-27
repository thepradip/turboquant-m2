"""
Ollama integration for TurboQuant.

Provides utilities to:
  - Query Ollama for available models and model info
  - Run generation benchmarks through the Ollama REST API
  - Project KV-cache memory savings with TurboQuant

Requires: pip install turboquant[ollama]
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

OLLAMA_DEFAULT_URL = "http://localhost:11434"


def _get_session():
    """Lazy import requests to avoid hard dependency."""
    import requests
    return requests.Session()


def list_models(base_url: str = OLLAMA_DEFAULT_URL) -> List[str]:
    """
    List available models in Ollama.

    Args:
        base_url: Ollama server URL.

    Returns:
        List of model name strings.
    """
    session = _get_session()
    r = session.get(f"{base_url}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


_OLLAMA_KEY_MAP = {
    # Ollama model_info uses arch-prefixed keys like "qwen2.block_count"
    # or "llama.attention.head_count". After splitting on ".", the last
    # segment is mapped to a canonical name.
    "block_count":           "num_hidden_layers",
    "embedding_length":      "hidden_size",
    "head_count":            "num_attention_heads",
    "head_count_kv":         "num_key_value_heads",
    "key_length":            "head_dim",
    "value_length":          "head_dim",
    "context_length":        "context_length",
    # Some models already use HF-style names
    "num_hidden_layers":     "num_hidden_layers",
    "hidden_size":           "hidden_size",
    "num_attention_heads":   "num_attention_heads",
    "num_key_value_heads":   "num_key_value_heads",
    "head_dim":              "head_dim",
}


def get_model_info(model: str, base_url: str = OLLAMA_DEFAULT_URL) -> Dict[str, Any]:
    """
    Get model architecture details from Ollama.

    Args:
        model: Model name (e.g. "qwen3.5:2b").
        base_url: Ollama server URL.

    Returns:
        Dict with family, parameter_size, quantization, and architecture params
        using canonical names (num_hidden_layers, hidden_size, etc.).
    """
    session = _get_session()
    r = session.post(f"{base_url}/api/show", json={"model": model}, timeout=10)
    r.raise_for_status()
    data = r.json()

    details = data.get("details", {})
    params = data.get("model_info", {})

    info: Dict[str, Any] = {
        "family": details.get("family", "unknown"),
        "parameter_size": details.get("parameter_size", "unknown"),
        "quantization": details.get("quantization_level", "unknown"),
        "format": details.get("format", "unknown"),
    }

    for raw_key, value in params.items():
        if value is None:
            continue
        short = raw_key.split(".")[-1]
        canonical = _OLLAMA_KEY_MAP.get(short)
        if canonical and canonical not in info:
            info[canonical] = value

    # Derive head_dim if not directly available
    if "head_dim" not in info:
        hs = info.get("hidden_size")
        nh = info.get("num_attention_heads")
        if hs and nh and nh > 0:
            info["head_dim"] = hs // nh

    # If num_key_value_heads is missing, assume MHA
    if "num_key_value_heads" not in info and "num_attention_heads" in info:
        info["num_key_value_heads"] = info["num_attention_heads"]

    return info


def generate(
    model: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0,
    base_url: str = OLLAMA_DEFAULT_URL,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Generate text via Ollama and return performance metrics.

    Args:
        model: Model name.
        prompt: Input prompt text.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        base_url: Ollama server URL.
        timeout: Request timeout in seconds.

    Returns:
        Dict with response text, token counts, throughput, and latency.
    """
    session = _get_session()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }

    t0 = time.time()
    r = session.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()

    data = r.json()
    response = data.get("response", "")
    eval_count = data.get("eval_count", len(response.split()))
    eval_duration_ns = data.get("eval_duration", elapsed * 1e9)
    prompt_eval_ns = data.get("prompt_eval_duration", 0)

    tok_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

    return {
        "response": response,
        "tokens": eval_count,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "total_time_s": round(elapsed, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "ttft_ms": round(prompt_eval_ns / 1e6, 1),
    }


def project_kv_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_lengths: Optional[List[int]] = None,
    tq_bits: int = 4,
) -> List[Dict[str, Any]]:
    """
    Project KV-cache memory at various context lengths.

    Compares FP16 KV cache vs TurboQuant compressed KV cache.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Attention head dimension.
        context_lengths: List of context lengths to project.
        tq_bits: TurboQuant bit width for projection.

    Returns:
        List of dicts with fp16_mb, tq_mb, saved_pct per context length.
    """
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

    fp16_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # bytes

    # Effective compression ratio for TurboQuant:
    # indices: bits/8 bytes per element, norms: 2 bytes per token
    # Per token: head_dim * bits/8 + 2 bytes (for one KV head)
    # vs FP16: head_dim * 2 bytes
    tq_per_token_per_head = head_dim * tq_bits / 8 + 2  # indices + norm
    fp16_per_token_per_head = head_dim * 2
    tq_ratio = fp16_per_token_per_head / tq_per_token_per_head

    rows = []
    for ctx in context_lengths:
        fp16_mb = fp16_per_token * ctx / 1024 / 1024
        tq_mb = fp16_mb / tq_ratio
        rows.append({
            "context_length": ctx,
            "fp16_kv_mb": round(fp16_mb, 2),
            "tq_kv_mb": round(tq_mb, 2),
            "compression_ratio": round(tq_ratio, 1),
            "saved_pct": round((1 - 1 / tq_ratio) * 100, 1),
            "saved_mb": round(fp16_mb - tq_mb, 2),
        })

    return rows
