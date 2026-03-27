"""
Distortion and quality metrics for KV-cache compression.

Provides cosine similarity, inner-product correlation, MSE, and a
combined distortion report for evaluating quantization quality.
"""

from __future__ import annotations

from typing import Dict

import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Average cosine similarity between corresponding vectors.

    Args:
        a: Original tensor (..., dim).
        b: Reconstructed tensor (..., dim).

    Returns:
        Mean cosine similarity as a Python float.
    """
    a_flat = a.float().reshape(-1, a.shape[-1])
    b_flat = b.float().reshape(-1, b.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def inner_product_correlation(
    orig: torch.Tensor, recon: torch.Tensor, n_queries: int = 200
) -> float:
    """
    Measure how well inner products are preserved (key metric for attention).

    Samples random query vectors and checks correlation between
    inner products with original vs reconstructed KV vectors.

    Args:
        orig: Original tensor (..., dim).
        recon: Reconstructed tensor (..., dim).
        n_queries: Number of random query vectors to sample.

    Returns:
        Pearson correlation coefficient as a Python float.
    """
    d = orig.shape[-1]
    q = torch.randn(n_queries, d, device=orig.device)
    o = orig.float().reshape(-1, d)
    r = recon.float().reshape(-1, d)
    dots_o = torch.matmul(q, o.T).flatten()
    dots_r = torch.matmul(q, r.T).flatten()
    corr = torch.corrcoef(torch.stack([dots_o, dots_r]))[0, 1].item()
    return corr


def measure_distortion(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """
    Comprehensive distortion measurement between original and reconstructed tensors.

    Args:
        original: Original tensor (..., dim).
        reconstructed: Reconstructed tensor (..., dim).

    Returns:
        Dictionary with MSE, cosine similarity stats, and inner product metrics.
    """
    original = original.float()
    reconstructed = reconstructed.float()

    mse = ((original - reconstructed) ** 2).mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(
        original.reshape(-1, original.shape[-1]),
        reconstructed.reshape(-1, reconstructed.shape[-1]),
        dim=-1,
    )

    d = original.shape[-1]
    queries = torch.randn(100, d, device=original.device)
    orig_flat = original.reshape(-1, d)
    recon_flat = reconstructed.reshape(-1, d)

    orig_dots = torch.matmul(queries, orig_flat.T)
    recon_dots = torch.matmul(queries, recon_flat.T)
    dot_error = ((orig_dots - recon_dots) ** 2).mean().item()
    dot_corr = torch.corrcoef(
        torch.stack([orig_dots.flatten(), recon_dots.flatten()])
    )[0, 1].item()

    return {
        "mse": mse,
        "cosine_similarity_mean": cos_sim.mean().item(),
        "cosine_similarity_min": cos_sim.min().item(),
        "cosine_similarity_std": cos_sim.std().item(),
        "inner_product_mse": dot_error,
        "inner_product_correlation": dot_corr,
    }
