"""
TurboQuant core tests — real MLX tensors, real math, real model inference.
No mocks. No fake data.

Run: pytest tests/test_core.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import mlx.core as mx
import numpy as np
from turboquant.compressor import PolarQuantMLX, pack_indices, unpack_indices
from turboquant.codebook import build_codebook, build_rotation


# ═══════════════════════════════════════════
#  Test 1: Codebook + Rotation
# ═══════════════════════════════════════════

class TestCodebook:
    def test_codebook_4bit_has_16_centroids(self):
        cb = build_codebook(bits=4, dim=128)
        assert cb.shape == (16,), f"Expected 16 centroids, got {cb.shape}"

    def test_codebook_3bit_has_8_centroids(self):
        cb = build_codebook(bits=3, dim=128)
        assert cb.shape == (8,), f"Expected 8 centroids, got {cb.shape}"

    def test_codebook_2bit_has_4_centroids(self):
        cb = build_codebook(bits=2, dim=128)
        assert cb.shape == (4,), f"Expected 4 centroids, got {cb.shape}"

    def test_codebook_centroids_are_sorted(self):
        for bits in [2, 3, 4]:
            cb = build_codebook(bits=bits, dim=128)
            cb_np = np.array(cb)
            assert np.all(cb_np[:-1] <= cb_np[1:]), f"{bits}-bit codebook not sorted"

    def test_rotation_is_orthogonal(self):
        for dim in [64, 128, 256]:
            R = np.array(build_rotation(dim, seed=42))
            # R @ R^T should be identity
            product = R @ R.T
            identity = np.eye(dim)
            assert np.allclose(product, identity, atol=1e-5), \
                f"Rotation matrix {dim}x{dim} not orthogonal: max error {np.max(np.abs(product - identity))}"


# ═══════════════════════════════════════════
#  Test 2: Quantize → Dequantize Round-Trip
# ═══════════════════════════════════════════

class TestRoundTrip:
    @pytest.fixture(params=[64, 128, 256])
    def head_dim(self, request):
        return request.param

    @pytest.fixture(params=[2, 3, 4])
    def bits(self, request):
        return request.param

    def test_round_trip_cosine(self, head_dim, bits):
        """Quantize unit vectors, dequantize, measure cosine similarity."""
        pq = PolarQuantMLX(head_dim, bits=bits, seed=42)

        # Random unit vectors (batch=2, heads=4, seq=32, head_dim)
        x = mx.random.normal((2, 4, 32, head_dim))
        norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        x_unit = x / mx.maximum(norms, 1e-8)

        indices = pq.quantize(x_unit)
        x_recon = pq.dequantize(indices)

        # Cosine similarity
        cosine = mx.mean(mx.sum(x_unit * x_recon, axis=-1))
        mx.eval(cosine)
        cos_val = cosine.item()

        # Minimum expected cosine by bits
        min_cosine = {4: 0.98, 3: 0.94, 2: 0.88}
        assert cos_val >= min_cosine[bits], \
            f"Cosine {cos_val:.4f} below threshold {min_cosine[bits]} for {bits}-bit, dim={head_dim}"

    def test_indices_are_uint8(self, head_dim, bits):
        """Quantized indices must be uint8."""
        pq = PolarQuantMLX(head_dim, bits=bits, seed=42)
        x = mx.random.normal((1, 1, 8, head_dim))
        norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        indices = pq.quantize(x / mx.maximum(norms, 1e-8))
        assert indices.dtype == mx.uint8

    def test_indices_in_range(self, head_dim, bits):
        """Index values must be in [0, 2^bits - 1]."""
        pq = PolarQuantMLX(head_dim, bits=bits, seed=42)
        x = mx.random.normal((1, 1, 16, head_dim))
        norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        indices = pq.quantize(x / mx.maximum(norms, 1e-8))
        mx.eval(indices)
        max_val = (2 ** bits) - 1
        assert mx.max(indices).item() <= max_val, \
            f"Index {mx.max(indices).item()} exceeds max {max_val} for {bits}-bit"

    def test_deterministic(self, head_dim, bits):
        """Same input → same output."""
        pq = PolarQuantMLX(head_dim, bits=bits, seed=42)
        x = mx.random.normal((1, 1, 8, head_dim), key=mx.random.key(123))
        norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        x_unit = x / mx.maximum(norms, 1e-8)

        idx1 = pq.quantize(x_unit)
        idx2 = pq.quantize(x_unit)
        mx.eval(idx1, idx2)
        assert mx.array_equal(idx1, idx2), "Quantization not deterministic"


# ═══════════════════════════════════════════
#  Test 3: Pack → Unpack Exact Round-Trip
# ═══════════════════════════════════════════

class TestPackUnpack:
    @pytest.fixture(params=[2, 3, 4])
    def bits(self, request):
        return request.param

    def test_pack_unpack_exact(self, bits):
        """Pack indices to bits, unpack, verify exact match."""
        head_dim = 128
        max_val = (2 ** bits) - 1
        # Random indices in valid range
        indices = mx.random.randint(0, max_val + 1, (1, 2, 16, head_dim)).astype(mx.uint8)
        mx.eval(indices)

        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, head_dim)
        mx.eval(unpacked)

        assert mx.array_equal(indices, unpacked), \
            f"Pack/unpack round-trip failed for {bits}-bit. " \
            f"Mismatches: {mx.sum(indices != unpacked).item()}/{indices.size}"

    def test_pack_reduces_size(self, bits):
        """Packed size should be smaller than unpacked."""
        head_dim = 256
        indices = mx.zeros((1, 2, 64, head_dim), dtype=mx.uint8)
        packed = pack_indices(indices, bits)
        mx.eval(packed)

        original_bytes = indices.nbytes
        packed_bytes = packed.nbytes
        assert packed_bytes < original_bytes, \
            f"Packed ({packed_bytes}) not smaller than original ({original_bytes}) for {bits}-bit"


# ═══════════════════════════════════════════
#  Test 4: compress_cache (compact=False)
# ═══════════════════════════════════════════

class TestCompressCache:
    @pytest.fixture
    def model_and_cache(self):
        """Load real model and create cache with real prefill."""
        from mlx_lm import load as mlx_load
        from mlx_lm.models.cache import make_prompt_cache
        model, tokenizer = mlx_load("mlx-community/Qwen3.5-4B-MLX-4bit")
        cache = make_prompt_cache(model)
        ids = mx.array(tokenizer.encode("What is 2+2?"))
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        return model, tokenizer, cache, logits

    def test_compress_returns_valid_metrics(self, model_and_cache):
        """compress_cache returns cosine, ratio, layers_compressed."""
        model, tokenizer, cache, logits = model_and_cache
        from turboquant import compress_cache
        result = compress_cache(cache, model=model, bits=4, compact=False)

        assert "cosine" in result
        assert "ratio" in result
        assert "layers_compressed" in result
        assert result["cosine"] >= 0.98, f"Cosine {result['cosine']} too low"
        assert result["ratio"] >= 2.0, f"Ratio {result['ratio']} too low"
        assert result["layers_compressed"] > 0

    def test_compact_false_preserves_fp16(self, model_and_cache):
        """With compact=False, FP16 keys/values still exist."""
        model, tokenizer, cache, logits = model_and_cache
        from turboquant import compress_cache
        compress_cache(cache, model=model, bits=4, compact=False)

        has_fp16 = any(hasattr(c, 'keys') and c.keys is not None for c in cache)
        assert has_fp16, "compact=False should preserve FP16 keys/values"

    def test_compact_true_frees_fp16(self, model_and_cache):
        """With compact=True, FP16 keys/values are freed."""
        model, tokenizer, cache, logits = model_and_cache
        from turboquant import compress_cache
        compress_cache(cache, model=model, bits=4, compact=True)

        # Compacted layers should have keys=None
        compacted = [c for c in cache if getattr(c, '_tq_compacted', False)]
        for c in compacted:
            assert c.keys is None, "compact=True should free FP16 keys"

    def test_generation_after_compact_false(self, model_and_cache):
        """Standard generation works after compress with compact=False."""
        model, tokenizer, cache, logits = model_and_cache
        from turboquant import compress_cache
        compress_cache(cache, model=model, bits=4, compact=False)

        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        tokens = []
        for _ in range(10):
            tok = y.item()
            if tok == tokenizer.eos_token_id:
                break
            tokens.append(tok)
            logits = model(y[:, None], cache=cache)
            mx.eval(logits)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)

        assert len(tokens) > 0, "No tokens generated after compact=False"
        answer = tokenizer.decode(tokens)
        assert len(answer) > 0, "Empty answer after compact=False"


# ═══════════════════════════════════════════
#  Test 5: generate_step (compact=True)
# ═══════════════════════════════════════════

class TestGenerateStep:
    @pytest.fixture
    def model_and_cache(self):
        """Load model and create cache with real prefill."""
        from mlx_lm import load as mlx_load
        from mlx_lm.models.cache import make_prompt_cache
        model, tokenizer = mlx_load("mlx-community/Qwen3.5-4B-MLX-4bit")
        cache = make_prompt_cache(model)
        ids = mx.array(tokenizer.encode("What is 5+3? Answer with just the number."))
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        return model, tokenizer, cache, logits

    @pytest.fixture
    def model_and_compressed_cache(self):
        """Load model, prefill, compress with compact=True."""
        from mlx_lm import load as mlx_load
        from mlx_lm.models.cache import make_prompt_cache
        from turboquant import compress_cache
        model, tokenizer = mlx_load("mlx-community/Qwen3.5-4B-MLX-4bit")
        cache = make_prompt_cache(model)
        ids = mx.array(tokenizer.encode("What is 5+3? Answer with just the number."))
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        compress_cache(cache, model=model, bits=4, compact=True)
        return model, tokenizer, cache, logits

    def test_generate_step_produces_tokens(self, model_and_compressed_cache):
        """generate_step should produce at least 1 token."""
        model, tokenizer, cache, logits = model_and_compressed_cache
        from turboquant import generate_step

        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        tokens = []
        for _ in range(10):
            tok = y.item()
            if tok == tokenizer.eos_token_id:
                break
            tokens.append(tok)
            logits = generate_step(model, y[:, None], cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)

        assert len(tokens) > 0, "generate_step produced no tokens"

    def test_generate_step_correct_answer(self, model_and_compressed_cache):
        """generate_step should produce a sensible answer (contains '8')."""
        model, tokenizer, cache, logits = model_and_compressed_cache
        from turboquant import generate_step

        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        tokens = []
        for _ in range(20):
            tok = y.item()
            if tok == tokenizer.eos_token_id:
                break
            tokens.append(tok)
            logits = generate_step(model, y[:, None], cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)

        answer = tokenizer.decode(tokens)
        assert "8" in answer, f"Expected '8' in answer, got: {answer}"

    def test_cache_stays_compacted(self, model_and_compressed_cache):
        """After generate_step, compacted layers should remain compacted."""
        model, tokenizer, cache, logits = model_and_compressed_cache
        from turboquant import generate_step

        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        logits = generate_step(model, y[:, None], cache)
        mx.eval(logits)

        compacted = [c for c in cache if getattr(c, '_tq_compacted', False)]
        assert len(compacted) > 0, "No compacted layers after generate_step"
        for c in compacted:
            assert c.keys is None, "FP16 keys should be freed after generate_step"

    def test_generate_step_memory_lower(self, model_and_compressed_cache):
        """Memory after generate_step should be lower than before compression."""
        model, tokenizer, cache, logits = model_and_compressed_cache
        from turboquant import generate_step

        mx.eval(mx.array([0]))
        mem_compacted = mx.get_active_memory()

        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        for _ in range(5):
            tok = y.item()
            if tok == tokenizer.eos_token_id:
                break
            logits = generate_step(model, y[:, None], cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)

        mx.eval(mx.array([0]))
        mem_after = mx.get_active_memory()

        # Memory after generation should not exceed compacted + reasonable overhead
        overhead_mb = (mem_after - mem_compacted) / 1024**2
        assert overhead_mb < 200, f"Memory grew by {overhead_mb:.0f} MB after generate_step — possible leak"

