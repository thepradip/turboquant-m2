"""
Lloyd-Max optimal scalar quantizer for the Beta distribution.

After random orthogonal rotation of unit-sphere vectors, each coordinate
follows Beta(d/2 - 1/2, d/2 - 1/2) on [-1, 1]. This module precomputes
the optimal quantization centroids and boundaries for that distribution.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
           Google Research, ICLR 2026 (arXiv:2504.19874)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.stats import beta as beta_dist


class LloydMaxCodebook:
    """
    Precompute optimal Lloyd-Max scalar quantizer for the Beta distribution
    that arises after random rotation of unit-sphere vectors.

    For a d-dimensional vector on the unit sphere, each coordinate after
    random rotation follows Beta(d/2 - 1/2, d/2 - 1/2) on [-1, 1].

    Args:
        bits: Number of quantization bits (2, 3, or 4).
        dim: Dimension of the vectors (typically head_dim, e.g. 128).
        iterations: Number of Lloyd-Max iterations for codebook optimization.
    """

    def __init__(self, bits: int, dim: int, iterations: int = 300):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")

        self.bits = bits
        self.num_centroids = 2 ** bits
        self.dim = dim
        self.centroids, self.boundaries = self._build(dim, iterations)

    def _build(self, dim: int, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal centroids via Lloyd-Max iteration."""
        alpha = max(dim / 2 - 0.5, 1.0)

        centroids = np.linspace(-0.95, 0.95, self.num_centroids)

        # Dense grid for numerical integration
        x = np.linspace(-0.999, 0.999, 10_000)
        pdf = beta_dist.pdf((x + 1) / 2, alpha, alpha) / 2  # Jacobian for [-1,1]

        for _ in range(iters):
            boundaries = np.zeros(self.num_centroids + 1)
            boundaries[0], boundaries[-1] = -1.0, 1.0
            for i in range(1, self.num_centroids):
                boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

            for i in range(self.num_centroids):
                mask = (x >= boundaries[i]) & (x < boundaries[i + 1])
                if mask.sum() > 0:
                    w = pdf[mask]
                    if w.sum() > 0:
                        centroids[i] = np.sum(x[mask] * w) / w.sum()

        return (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries, dtype=torch.float32),
        )

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map continuous values to nearest centroid index.

        Args:
            x: Tensor of any shape with values in [-1, 1].

        Returns:
            Tensor of same shape with uint8 indices.
        """
        dists = (x.unsqueeze(-1) - self.centroids.to(x.device)).abs()
        return dists.argmin(dim=-1).to(torch.uint8)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map centroid indices back to centroid values.

        Args:
            indices: Tensor of uint8 centroid indices.

        Returns:
            Tensor of float32 centroid values.
        """
        return self.centroids.to(indices.device)[indices.long()]

    def to(self, device: torch.device) -> "LloydMaxCodebook":
        """Move codebook tensors to a device."""
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)
        return self
