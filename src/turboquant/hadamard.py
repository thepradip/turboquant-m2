"""
Fast Walsh-Hadamard Transform for MLX.

O(d log d) butterfly operations — 18x fewer FLOPs than dense rotation.
Ported from turboquant-vllm/turboquant/core/hadamard.py (our own code).

Used by PolarQuant to Gaussianize KV cache coordinates before quantization.
"""

import math
import mlx.core as mx


class HadamardTransform:
    """Randomized Hadamard Transform: y = H @ diag(signs) @ x / sqrt(d).

    H is the Walsh-Hadamard matrix computed via butterfly operations.
    signs is a random ±1 vector for randomization.

    Args:
        dim: dimension (must be power of 2)
        seed: random seed for sign generation
    """

    def __init__(self, dim: int, seed: int = 42):
        if dim < 1 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Dimension must be power of 2, got {dim}")
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)

        # Random ±1 signs
        import numpy as np
        rng = np.random.RandomState(seed)
        signs = rng.randint(0, 2, size=(dim,)) * 2 - 1
        self.signs = mx.array(signs.astype(float))

    def _fwht(self, x):
        """Fast Walsh-Hadamard Transform via butterfly. O(d log d)."""
        orig_shape = x.shape
        n = x.shape[-1]
        x = x.reshape(-1, n)
        batch = x.shape[0]

        h = 1
        while h < n:
            # Reshape for butterfly: (batch, n/(2h), 2, h)
            x = x.reshape(batch, n // (2 * h), 2, h)
            a = x[:, :, 0, :]
            b = x[:, :, 1, :]
            # Butterfly: add/subtract pairs
            top = a + b
            bot = a - b
            x = mx.stack([top, bot], axis=2)
            x = x.reshape(batch, n)
            h *= 2

        x = x * self.scale
        return x.reshape(orig_shape)

    def forward(self, x):
        """Forward transform: multiply by signs, then WHT."""
        return self._fwht(x * self.signs)

    def inverse(self, y):
        """Inverse transform: WHT then multiply by signs (WHT is self-inverse up to scale)."""
        return self._fwht(y) * self.signs
