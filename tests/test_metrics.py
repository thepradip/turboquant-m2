"""Tests for distortion metrics."""

import pytest
import torch

from turboquant.metrics import cosine_similarity, inner_product_correlation, measure_distortion


class TestCosineSimilarity:
    def test_identical(self):
        x = torch.randn(10, 64)
        assert cosine_similarity(x, x) == pytest.approx(1.0, abs=1e-5)

    def test_range(self):
        a = torch.randn(10, 64)
        b = torch.randn(10, 64)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_high_sim_on_small_perturbation(self):
        x = torch.randn(100, 128)
        noise = torch.randn_like(x) * 0.01
        sim = cosine_similarity(x, x + noise)
        assert sim > 0.99

    def test_4d_input(self):
        a = torch.randn(1, 8, 32, 128)
        b = torch.randn(1, 8, 32, 128)
        sim = cosine_similarity(a, b)
        assert isinstance(sim, float)

    def test_fp16_input(self):
        a = torch.randn(10, 64, dtype=torch.float16)
        b = torch.randn(10, 64, dtype=torch.float16)
        sim = cosine_similarity(a, b)
        assert isinstance(sim, float)


class TestInnerProductCorrelation:
    def test_identical(self):
        x = torch.randn(10, 64)
        corr = inner_product_correlation(x, x)
        assert corr == pytest.approx(1.0, abs=1e-4)

    def test_range(self):
        a = torch.randn(50, 64)
        b = torch.randn(50, 64)
        corr = inner_product_correlation(a, b)
        assert -1.0 <= corr <= 1.0

    def test_custom_n_queries(self):
        a = torch.randn(20, 64)
        b = a + torch.randn_like(a) * 0.1
        corr = inner_product_correlation(a, b, n_queries=50)
        assert corr > 0.9

    def test_4d_input(self):
        a = torch.randn(1, 4, 16, 64)
        b = a + torch.randn_like(a) * 0.05
        corr = inner_product_correlation(a, b)
        assert corr > 0.95


class TestMeasureDistortion:
    def test_identical(self):
        x = torch.randn(1, 4, 16, 64)
        d = measure_distortion(x, x)
        assert d["mse"] == pytest.approx(0.0, abs=1e-6)
        assert d["cosine_similarity_mean"] == pytest.approx(1.0, abs=1e-4)
        assert d["inner_product_correlation"] == pytest.approx(1.0, abs=1e-3)

    def test_keys_present(self):
        x = torch.randn(1, 4, 16, 64)
        y = x + torch.randn_like(x) * 0.1
        d = measure_distortion(x, y)

        expected_keys = {
            "mse", "cosine_similarity_mean", "cosine_similarity_min",
            "cosine_similarity_std", "inner_product_mse", "inner_product_correlation",
        }
        assert set(d.keys()) == expected_keys

    def test_all_floats(self):
        x = torch.randn(2, 4, 8, 64)
        y = x + torch.randn_like(x) * 0.1
        d = measure_distortion(x, y)
        for v in d.values():
            assert isinstance(v, float)

    def test_small_perturbation(self):
        x = torch.randn(1, 8, 32, 128)
        y = x + torch.randn_like(x) * 0.01
        d = measure_distortion(x, y)
        assert d["mse"] < 0.001
        assert d["cosine_similarity_mean"] > 0.999
