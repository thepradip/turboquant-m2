"""
TurboQuant — Near-optimal KV-cache compression for LLM inference.

Based on: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
          Google Research, ICLR 2026 (arXiv:2504.19874)

Usage::

    from turboquant import TurboQuant

    tq = TurboQuant(bits=4, head_dim=128)
    compressed = tq.compress(kv_tensor)  # (batch, heads, seq, head_dim)
    reconstructed = tq.decompress(compressed)
    print(tq.stats())
"""

__version__ = "0.2.0"

from .codebook import LloydMaxCodebook
from .compressor import CompressedKVCache, StandardQ4Quantizer, TurboQuant
from .metrics import cosine_similarity, inner_product_correlation, measure_distortion

__all__ = [
    "TurboQuant",
    "CompressedKVCache",
    "StandardQ4Quantizer",
    "LloydMaxCodebook",
    "cosine_similarity",
    "inner_product_correlation",
    "measure_distortion",
    "__version__",
]
