"""Tests for LloydMaxCodebook."""

import pytest
import torch

from turboquant.codebook import LloydMaxCodebook


class TestLloydMaxCodebook:
    def test_init_valid(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        assert cb.bits == 4
        assert cb.num_centroids == 16
        assert cb.centroids.shape == (16,)
        assert cb.boundaries.shape == (17,)

    def test_init_2bit(self):
        cb = LloydMaxCodebook(bits=2, dim=64)
        assert cb.num_centroids == 4
        assert cb.centroids.shape == (4,)

    def test_init_3bit(self):
        cb = LloydMaxCodebook(bits=3, dim=128)
        assert cb.num_centroids == 8

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be in"):
            LloydMaxCodebook(bits=0, dim=128)
        with pytest.raises(ValueError, match="bits must be in"):
            LloydMaxCodebook(bits=9, dim=128)

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be >= 2"):
            LloydMaxCodebook(bits=4, dim=1)

    def test_centroids_sorted(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        # Centroids should be roughly sorted (increasing)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] < cb.centroids[i + 1]

    def test_centroids_in_range(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        assert cb.centroids.min() >= -1.0
        assert cb.centroids.max() <= 1.0

    def test_boundaries_in_range(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        assert cb.boundaries[0] == -1.0
        assert cb.boundaries[-1] == 1.0

    def test_quantize_shape(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        x = torch.randn(2, 8, 32, 128)
        idx = cb.quantize(x)
        assert idx.shape == x.shape
        assert idx.dtype == torch.uint8

    def test_quantize_range(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        x = torch.randn(100, 128)
        idx = cb.quantize(x)
        assert idx.min() >= 0
        assert idx.max() < 16

    def test_dequantize_shape(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        idx = torch.randint(0, 16, (2, 8, 32, 128), dtype=torch.uint8)
        vals = cb.dequantize(idx)
        assert vals.shape == idx.shape
        assert vals.dtype == torch.float32

    def test_roundtrip_preserves_shape(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        x = torch.randn(4, 128)
        idx = cb.quantize(x)
        vals = cb.dequantize(idx)
        assert vals.shape == x.shape

    def test_to_device(self):
        cb = LloydMaxCodebook(bits=4, dim=128)
        cb.to(torch.device("cpu"))
        assert cb.centroids.device == torch.device("cpu")

    def test_low_dim(self):
        """Test with low dimension where alpha would be < 1."""
        cb = LloydMaxCodebook(bits=2, dim=2)
        assert cb.num_centroids == 4
        x = torch.randn(10, 2)
        idx = cb.quantize(x)
        assert idx.shape == (10, 2)
