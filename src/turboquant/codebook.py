"""Lloyd-Max optimal scalar quantizer for Beta distribution (shared by MLX and numpy)."""

import numpy as np
from scipy.stats import beta as beta_dist


def build_codebook(bits: int, dim: int, iterations: int = 300) -> np.ndarray:
    """
    Precompute Lloyd-Max centroids for the Beta(d/2-0.5, d/2-0.5) distribution.
    After random rotation of unit-sphere vectors, each coordinate follows this distribution.

    Returns numpy float32 array of centroids.
    """
    n = 2 ** bits
    alpha = max(dim / 2 - 0.5, 1.0)
    centroids = np.linspace(-0.95, 0.95, n)
    x = np.linspace(-0.999, 0.999, 10_000)
    pdf = beta_dist.pdf((x + 1) / 2, alpha, alpha) / 2

    for _ in range(iterations):
        b = np.zeros(n + 1)
        b[0], b[-1] = -1.0, 1.0
        for i in range(1, n):
            b[i] = (centroids[i - 1] + centroids[i]) / 2
        for i in range(n):
            mask = (x >= b[i]) & (x < b[i + 1])
            if mask.sum() > 0:
                w = pdf[mask]
                if w.sum() > 0:
                    centroids[i] = np.sum(x[mask] * w) / w.sum()

    return centroids.astype(np.float32)


def build_rotation(dim: int, seed: int) -> np.ndarray:
    """
    Generate random orthogonal rotation matrix via QR decomposition.
    Ensures proper rotation (det=+1) by fixing sign ambiguity.
    """
    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim)
    Q, R = np.linalg.qr(G)
    # Fix sign ambiguity to ensure det(Q) = +1
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign[np.newaxis, :]
    return Q.astype(np.float32)


def build_qjl_matrix(dim: int, m: int = None, seed: int = 43) -> np.ndarray:
    """
    Generate random Gaussian projection matrix for QJL.
    S has i.i.d. N(0,1) entries, shape (m, dim). Default m=dim.
    """
    if m is None:
        m = dim
    rng = np.random.RandomState(seed)
    return rng.randn(m, dim).astype(np.float32)
