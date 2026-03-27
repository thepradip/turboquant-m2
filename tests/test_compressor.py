"""Tests for TurboQuant compressor."""

import pytest
import torch

from turboquant import TurboQuant, CompressedKVCache, StandardQ4Quantizer


class TestTurboQuant:
    def test_init_default(self):
        tq = TurboQuant()
        assert tq.bits == 4
        assert tq.head_dim == 128

    def test_init_custom(self):
        tq = TurboQuant(bits=3, head_dim=64, seed=123)
        assert tq.bits == 3
        assert tq.head_dim == 64
        assert tq.seed == 123

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be 2, 3, or 4"):
            TurboQuant(bits=5)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim must be >= 2"):
            TurboQuant(head_dim=1)

    def test_compress_returns_compressed_kv(self):
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 16, 64, dtype=torch.float16)
        comp = tq.compress(x)
        assert isinstance(comp, CompressedKVCache)

    def test_compress_shape(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 8, 32, 128, dtype=torch.float16)
        comp = tq.compress(x)
        assert comp.indices.shape == (1, 8, 32, 128)
        assert comp.norms.shape == (1, 8, 32)
        assert comp.shape == x.shape
        assert comp.dtype == torch.float16

    def test_compress_indices_dtype(self):
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 16, 64)
        comp = tq.compress(x)
        assert comp.indices.dtype == torch.uint8

    def test_compress_dim_mismatch(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 4, 16, 64)
        with pytest.raises(ValueError, match="Last dim must be 128"):
            tq.compress(x)

    def test_decompress_shape(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 8, 32, 128, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert recon.shape == x.shape
        assert recon.dtype == x.dtype

    def test_roundtrip_quality_4bit(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 8, 64, 128, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)

        cos = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1, 128), recon.float().reshape(-1, 128), dim=-1
        ).mean()
        assert cos > 0.99, f"4-bit cosine sim too low: {cos}"

    def test_roundtrip_quality_3bit(self):
        tq = TurboQuant(bits=3, head_dim=128)
        x = torch.randn(1, 8, 64, 128, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)

        cos = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1, 128), recon.float().reshape(-1, 128), dim=-1
        ).mean()
        assert cos > 0.97, f"3-bit cosine sim too low: {cos}"

    def test_roundtrip_quality_2bit(self):
        tq = TurboQuant(bits=2, head_dim=128)
        x = torch.randn(1, 8, 64, 128, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)

        cos = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1, 128), recon.float().reshape(-1, 128), dim=-1
        ).mean()
        assert cos > 0.90, f"2-bit cosine sim too low: {cos}"

    def test_memory_bytes(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 8, 32, 128, dtype=torch.float16)
        comp = tq.compress(x)
        mem = tq.memory_bytes(comp)

        assert mem["original"] > mem["compressed"]
        assert mem["ratio"] > 1.0
        assert 0 < mem["savings_pct"] < 100

    def test_stats(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 4, 16, 128, dtype=torch.float16)
        tq.compress(x)
        stats = tq.stats()

        assert stats["bits"] == 4
        assert stats["head_dim"] == 128
        assert stats["original_bytes"] > 0
        assert stats["compressed_bytes"] > 0

    def test_reset_stats(self):
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 4, 16, 128, dtype=torch.float16)
        tq.compress(x)
        tq.reset_stats()

        assert tq._original_bytes == 0
        assert tq._compressed_bytes == 0

    def test_compression_ratio(self):
        tq = TurboQuant(bits=4, head_dim=128)
        assert tq.compression_ratio() == 0.0

        x = torch.randn(1, 4, 16, 128, dtype=torch.float16)
        tq.compress(x)
        ratio = tq.compression_ratio()
        assert ratio > 3.0  # 4-bit should give ~3.8x

    def test_reproducibility(self):
        """Same seed should produce same results."""
        x = torch.randn(1, 4, 16, 128, dtype=torch.float16)

        tq1 = TurboQuant(bits=4, head_dim=128, seed=42)
        comp1 = tq1.compress(x)
        recon1 = tq1.decompress(comp1)

        tq2 = TurboQuant(bits=4, head_dim=128, seed=42)
        comp2 = tq2.compress(x)
        recon2 = tq2.decompress(comp2)

        assert torch.equal(comp1.indices, comp2.indices)
        assert torch.equal(recon1, recon2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        x = torch.randn(1, 4, 16, 128, dtype=torch.float16)

        tq1 = TurboQuant(bits=4, head_dim=128, seed=42)
        comp1 = tq1.compress(x)

        tq2 = TurboQuant(bits=4, head_dim=128, seed=99)
        comp2 = tq2.compress(x)

        assert not torch.equal(comp1.indices, comp2.indices)

    def test_fp32_input(self):
        """Should handle FP32 input."""
        tq = TurboQuant(bits=4, head_dim=64)
        x = torch.randn(1, 4, 16, 64, dtype=torch.float32)
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert recon.dtype == torch.float32
        assert recon.shape == x.shape

    def test_to_device(self):
        tq = TurboQuant(bits=4, head_dim=64)
        tq.to(torch.device("cpu"))
        assert tq.rotation.device == torch.device("cpu")

    def test_batch_sizes(self):
        """Should work with various batch sizes."""
        tq = TurboQuant(bits=4, head_dim=64)
        for batch in [1, 2, 4]:
            x = torch.randn(batch, 4, 16, 64, dtype=torch.float16)
            comp = tq.compress(x)
            recon = tq.decompress(comp)
            assert recon.shape == x.shape

    def test_single_token(self):
        """Should work with seq_len=1."""
        tq = TurboQuant(bits=4, head_dim=128)
        x = torch.randn(1, 8, 1, 128, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert recon.shape == x.shape

    def test_small_head_dim(self):
        """Should work with small head dimensions."""
        tq = TurboQuant(bits=4, head_dim=8)
        x = torch.randn(1, 4, 16, 8, dtype=torch.float16)
        comp = tq.compress(x)
        recon = tq.decompress(comp)
        assert recon.shape == x.shape


class TestCompressedKVCache:
    def test_device_property(self):
        indices = torch.zeros(1, dtype=torch.uint8)
        norms = torch.zeros(1, dtype=torch.float16)
        comp = CompressedKVCache(indices, norms, torch.Size([1, 1]), torch.float16)
        assert comp.device == torch.device("cpu")

    def test_to_device(self):
        indices = torch.zeros(1, dtype=torch.uint8)
        norms = torch.zeros(1, dtype=torch.float16)
        comp = CompressedKVCache(indices, norms, torch.Size([1, 1]), torch.float16)
        comp2 = comp.to(torch.device("cpu"))
        assert comp2.device == torch.device("cpu")


class TestStandardQ4Quantizer:
    def test_compress_decompress(self):
        q4 = StandardQ4Quantizer(group_size=32)
        x = torch.randn(1, 4, 16, 128, dtype=torch.float16)
        comp = q4.compress(x)
        recon = q4.decompress(comp)
        assert recon.shape == x.shape

    def test_quality(self):
        q4 = StandardQ4Quantizer(group_size=32)
        x = torch.randn(1, 4, 64, 128, dtype=torch.float16)
        comp = q4.compress(x)
        recon = q4.decompress(comp)

        cos = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1, 128), recon.float().reshape(-1, 128), dim=-1
        ).mean()
        assert cos > 0.99

    def test_memory_bytes(self):
        q4 = StandardQ4Quantizer(group_size=32)
        x = torch.randn(1, 1, 32, 128, dtype=torch.float16)
        comp = q4.compress(x)
        mem = q4.memory_bytes(comp)

        assert mem["original"] > mem["compressed"]
        assert mem["ratio"] > 1.0
        assert mem["overhead_pct"] > 0

    def test_dim_not_divisible(self):
        q4 = StandardQ4Quantizer(group_size=32)
        x = torch.randn(1, 1, 1, 100, dtype=torch.float16)
        with pytest.raises(ValueError, match="not divisible"):
            q4.compress(x)
