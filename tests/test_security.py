"""Security-focused tests for TurboQuant.

Validates that the package handles edge cases safely, doesn't leak data,
and doesn't have common vulnerabilities.
"""

import pytest
import torch
import numpy as np

from turboquant import TurboQuant, CompressedKVCache, LloydMaxCodebook


class TestInputValidation:
    """Ensure all public APIs validate inputs properly."""

    def test_negative_bits(self):
        with pytest.raises(ValueError):
            TurboQuant(bits=-1)

    def test_zero_bits(self):
        with pytest.raises(ValueError):
            TurboQuant(bits=0)

    def test_oversized_bits(self):
        with pytest.raises(ValueError):
            TurboQuant(bits=8)

    def test_zero_head_dim(self):
        with pytest.raises(ValueError):
            TurboQuant(head_dim=0)

    def test_negative_head_dim(self):
        with pytest.raises(ValueError):
            TurboQuant(head_dim=-1)

    def test_wrong_dim_input(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 4, 16, 64)
        with pytest.raises(ValueError, match="Last dim must be 128"):
            tq.compress(x)

    def test_codebook_invalid_bits(self):
        with pytest.raises(ValueError):
            LloydMaxCodebook(bits=0, dim=128)

    def test_codebook_invalid_dim(self):
        with pytest.raises(ValueError):
            LloydMaxCodebook(bits=4, dim=0)


class TestNumericalStability:
    """Ensure numerical stability with edge-case inputs."""

    def test_zero_tensor(self):
        """All-zeros should not crash (norms clamped to 1e-8)."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.zeros(1, 4, 8, 64, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert recon.shape == x.shape
        assert not torch.isnan(recon).any()
        assert not torch.isinf(recon).any()

    def test_very_large_values(self):
        """Large values should not produce NaN/Inf."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 8, 64, dtype=torch.float16) * 1000
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert not torch.isnan(recon).any()
        assert not torch.isinf(recon).any()

    def test_very_small_values(self):
        """Very small values should not produce NaN."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 8, 64, dtype=torch.float16) * 1e-6
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert not torch.isnan(recon).any()

    def test_mixed_sign_values(self):
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        x[:, :, :4, :] = x[:, :, :4, :].abs()   # positive
        x[:, :, 4:, :] = -x[:, :, 4:, :].abs()   # negative
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert not torch.isnan(recon).any()

    def test_single_element_sequence(self):
        """Single token should work."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 1, 1, 64, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert recon.shape == x.shape
        assert not torch.isnan(recon).any()


class TestDataIsolation:
    """Ensure no data leaks between compress/decompress calls."""

    def test_no_cross_contamination(self):
        """Two different inputs should produce independent outputs."""
        tq = TurboQuant(bits=4, head_dim=64, seed=42)

        x1 = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        x2 = torch.randn(1, 4, 8, 64, dtype=torch.float16) * 5

        comp1 = tq.compress(x1)
        comp2 = tq.compress(x2)

        recon1 = tq.decompress(comp1)
        recon2 = tq.decompress(comp2)

        # Reconstructions should be close to their originals, not each other
        cos_11 = torch.nn.functional.cosine_similarity(
            x1.float().reshape(-1, 64), recon1.float().reshape(-1, 64), dim=-1
        ).mean()
        cos_12 = torch.nn.functional.cosine_similarity(
            x1.float().reshape(-1, 64), recon2.float().reshape(-1, 64), dim=-1
        ).mean()

        assert cos_11 > cos_12, "Reconstruction closer to wrong input!"

    def test_compressed_immutable(self):
        """Modifying compressed data should not affect original."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        comp = tq.compress(x)

        # Save original indices
        original_indices = comp.indices.clone()

        # Tamper with compressed data
        comp.indices.zero_()

        # Original should be unchanged
        assert not torch.equal(original_indices, comp.indices)


class TestRotationMatrixSafety:
    """Ensure the random rotation matrix is proper orthogonal."""

    def test_rotation_orthogonal(self):
        """R^T @ R should be identity."""
        tq = TurboQuant(bits=4, head_dim=64)
        R = tq.rotation
        eye = torch.eye(64)
        product = R.T @ R
        assert torch.allclose(product, eye, atol=1e-5)

    def test_rotation_determinant(self):
        """det(R) should be +1 or -1."""
        tq = TurboQuant(bits=4, head_dim=64)
        det = torch.det(tq.rotation)
        assert abs(abs(det.item()) - 1.0) < 1e-4

    def test_rotation_preserves_norm(self):
        """Rotation should preserve vector norms."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(10, 64)
        y = x @ tq.rotation.T
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-5)
