"""Tests for integration adapters (unit tests without external dependencies)."""

import pytest
import torch

from turboquant.integrations.vllm_adapter import (
    TurboQuantKVManager,
    resolve_model_config,
    analyze,
)
from turboquant.integrations.llamacpp_adapter import (
    compare_kv_methods,
    project_gguf_plus_tq_memory,
)


class TestTurboQuantKVManager:
    def test_init(self):
        mgr = TurboQuantKVManager(bits=4, head_dim=64, num_layers=4)
        assert mgr.bits == 4
        assert mgr.head_dim == 64
        assert mgr.num_layers == 4
        assert len(mgr._key_compressors) == 4
        assert len(mgr._value_compressors) == 4

    def test_compress_decompress_layer(self):
        mgr = TurboQuantKVManager(bits=4, head_dim=64, num_layers=4)
        key = torch.randn(1, 4, 16, 64, dtype=torch.float16)
        value = torch.randn(1, 4, 16, 64, dtype=torch.float16)

        comp_k, comp_v = mgr.compress_layer(0, key, value)
        recon_k, recon_v = mgr.decompress_layer(0, comp_k, comp_v)

        assert recon_k.shape == key.shape
        assert recon_v.shape == value.shape

    def test_compress_decompress_all(self):
        num_layers = 4
        mgr = TurboQuantKVManager(bits=4, head_dim=64, num_layers=num_layers)

        kv_cache = [
            (
                torch.randn(1, 4, 16, 64, dtype=torch.float16),
                torch.randn(1, 4, 16, 64, dtype=torch.float16),
            )
            for _ in range(num_layers)
        ]

        compressed = mgr.compress_all(kv_cache)
        assert len(compressed) == num_layers

        decompressed = mgr.decompress_all(compressed)
        assert len(decompressed) == num_layers
        for k, v in decompressed:
            assert k.shape == (1, 4, 16, 64)
            assert v.shape == (1, 4, 16, 64)

    def test_quality(self):
        mgr = TurboQuantKVManager(bits=4, head_dim=128, num_layers=2)
        key = torch.randn(1, 4, 32, 128, dtype=torch.float16)
        value = torch.randn(1, 4, 32, 128, dtype=torch.float16)

        comp_k, comp_v = mgr.compress_layer(0, key, value)
        recon_k, _ = mgr.decompress_layer(0, comp_k, comp_v)

        cos = torch.nn.functional.cosine_similarity(
            key.float().reshape(-1, 128),
            recon_k.float().reshape(-1, 128),
            dim=-1,
        ).mean()
        assert cos > 0.99

    def test_memory_savings(self):
        mgr = TurboQuantKVManager(bits=4, head_dim=128, num_layers=32)
        savings = mgr.memory_savings(num_kv_heads=4, context_length=32768)

        assert savings["original_mb"] > savings["compressed_mb"]
        assert savings["ratio"] > 1.0
        assert savings["savings_pct"] > 50

    def test_to_device(self):
        mgr = TurboQuantKVManager(bits=4, head_dim=64, num_layers=2)
        mgr.to(torch.device("cpu"))
        # Just verify it doesn't crash


class TestResolveModelConfig:
    def test_explicit_values(self):
        cfg = resolve_model_config("custom", num_layers=32, num_kv_heads=8, head_dim=128)
        assert cfg["num_layers"] == 32
        assert cfg["num_kv_heads"] == 8
        assert cfg["head_dim"] == 128
        assert cfg["source"] == "explicit"

    def test_known_model_qwen(self):
        cfg = resolve_model_config("Qwen/Qwen2.5-7B-Instruct")
        assert cfg["num_layers"] > 0
        assert cfg["num_kv_heads"] > 0
        assert cfg["head_dim"] > 0

    def test_known_model_llama(self):
        cfg = resolve_model_config("meta-llama/Llama-3.1-8B")
        assert cfg["num_layers"] == 32
        assert cfg["num_kv_heads"] == 8
        assert cfg["head_dim"] == 128

    def test_known_model_mistral(self):
        cfg = resolve_model_config("mistralai/Mistral-7B-v0.1")
        assert cfg["num_layers"] == 32

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve config"):
            resolve_model_config("totally-unknown-model-xyz")

    def test_partial_override(self):
        cfg = resolve_model_config("meta-llama/Llama-3.1-8B", num_layers=64)
        assert cfg["num_layers"] == 64  # overridden
        assert cfg["num_kv_heads"] == 8  # from known table


class TestAnalyze:
    def test_analyze_with_explicit_config(self):
        report = analyze(
            "test-model",
            gpu_memory_gb=24,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            context_lengths=[2048, 4096],
            batch_sizes=[1, 4],
            run_speed_benchmark=False,
            print_report=False,
        )
        assert "model_config" in report
        assert "quality" in report
        assert "memory_table" in report
        assert "feasibility" in report
        assert report["tq_compression_ratio"] > 1.0

    def test_analyze_quality_results(self):
        report = analyze(
            "test-model",
            gpu_memory_gb=24,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            context_lengths=[2048],
            batch_sizes=[1],
            run_speed_benchmark=False,
            print_report=False,
        )
        for bits in [2, 3, 4]:
            assert bits in report["quality"]
            assert report["quality"][bits]["cosine_mean"] > 0.9
            assert report["quality"][bits]["compression_ratio"] > 1.0

    def test_analyze_feasibility(self):
        report = analyze(
            "test-model",
            gpu_memory_gb=80,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            context_lengths=[2048],
            batch_sizes=[1],
            run_speed_benchmark=False,
            print_report=False,
        )
        f = report["feasibility"][1]
        assert f["max_context_tq"] > f["max_context_fp16"]
        assert f["improvement_x"] > 1.0

    def test_analyze_memory_table(self):
        report = analyze(
            "test-model",
            gpu_memory_gb=24,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            context_lengths=[2048, 4096],
            batch_sizes=[1, 4],
            run_speed_benchmark=False,
            print_report=False,
        )
        assert len(report["memory_table"]) == 4  # 2 contexts × 2 batches
        for row in report["memory_table"]:
            assert row["tq_kv_mb"] < row["fp16_kv_mb"]
            assert row["saved_pct"] > 0

    def test_analyze_with_known_model(self):
        report = analyze(
            "meta-llama/Llama-3.1-8B",
            gpu_memory_gb=24,
            context_lengths=[4096],
            batch_sizes=[1],
            run_speed_benchmark=False,
            print_report=False,
        )
        assert report["model_config"]["num_layers"] == 32
        assert report["model_params_b"] > 0

    def test_analyze_with_speed(self):
        report = analyze(
            "test-model",
            gpu_memory_gb=24,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            context_lengths=[2048],
            batch_sizes=[1],
            run_speed_benchmark=True,
            print_report=False,
        )
        assert report["speed"]["per_token_all_layers_ms"] > 0
        assert report["speed"]["per_token_per_layer_ms"] > 0


class TestLlamaCppAdapter:
    def test_compare_kv_methods(self):
        results = compare_kv_methods(
            num_layers=4, num_kv_heads=2, head_dim=64, seq_len=32
        )

        assert "standard_q4" in results
        assert "turboquant" in results

        for method in ["standard_q4", "turboquant"]:
            assert "cosine_mean" in results[method]
            assert "ip_corr_mean" in results[method]
            assert results[method]["cosine_mean"] > 0.9
            assert results[method]["compression_ratio"] > 1.0

    def test_project_gguf_plus_tq_memory(self):
        rows = project_gguf_plus_tq_memory(
            model_params_b=2.0,
            num_layers=28,
            num_kv_heads=4,
            head_dim=128,
            context_lengths=[4096, 32768],
        )

        assert len(rows) == 2
        for row in rows:
            assert row["total_tq_mb"] < row["total_fp16_mb"]
            assert row["saved_pct"] > 0


class TestOllamaAdapterImport:
    """Just verify the module can be imported (doesn't require Ollama running)."""

    def test_import(self):
        from turboquant.integrations.ollama_adapter import (
            list_models,
            get_model_info,
            generate,
            project_kv_memory,
        )
        assert callable(list_models)
        assert callable(get_model_info)
        assert callable(generate)
        assert callable(project_kv_memory)

    def test_project_kv_memory(self):
        from turboquant.integrations.ollama_adapter import project_kv_memory

        rows = project_kv_memory(
            num_layers=28,
            num_kv_heads=4,
            head_dim=128,
            context_lengths=[4096, 32768],
        )
        assert len(rows) == 2
        for row in rows:
            assert row["fp16_kv_mb"] > row["tq_kv_mb"]
            assert row["saved_pct"] > 0


class TestTransformersAdapterImport:
    """Verify the module imports (doesn't require transformers installed)."""

    def test_import(self):
        from turboquant.integrations.transformers_adapter import (
            extract_kv_cache,
            compress_kv_cache,
            decompress_kv_cache,
            benchmark_kv_compression,
            get_model_kv_config,
        )
        assert callable(extract_kv_cache)
        assert callable(compress_kv_cache)
        assert callable(decompress_kv_cache)
        assert callable(benchmark_kv_compression)
        assert callable(get_model_kv_config)
