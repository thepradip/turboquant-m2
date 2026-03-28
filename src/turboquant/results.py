"""
Experiment result saving — auto-save after each run.

Usage:
    from turboquant import save_experiment, list_experiments, load_experiment

    result = compress_cache(cache, model=model, bits=4)
    save_experiment(
        model_name="Qwen3.5-2B",
        compress_result=result,
        context_tokens=8000,
        gen_tps=35.2,
        ttft_ms=3000,
    )
"""

import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_results_dir() -> Path:
    """Get (and create) the results/ directory next to the package root."""
    # Walk up from this file to find the project root (where pyproject.toml or .git lives)
    d = Path(__file__).resolve().parent
    for _ in range(5):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            results = d / "results"
            results.mkdir(exist_ok=True)
            return results
        d = d.parent
    # Fallback: results/ next to src/
    results = Path(__file__).resolve().parent.parent.parent / "results"
    results.mkdir(exist_ok=True)
    return results


def _get_hardware() -> Dict[str, str]:
    """Auto-detect hardware info (Apple Silicon chip, RAM, OS)."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
    }
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=2,
            ).strip()
            info["chip"] = chip
        except Exception:
            info["chip"] = platform.processor() or "unknown"
        try:
            ram_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, timeout=2,
            ).strip())
            info["ram_gb"] = round(ram_bytes / (1024 ** 3))
        except Exception:
            pass
    return info


def _short_model_name(model_name: str) -> str:
    """Shorten model name for filenames: 'mlx-community/Qwen3.5-2B-4bit' → 'qwen35-2b-4bit'."""
    name = model_name.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    return name.lower()


def save_experiment(
    model_name: str,
    compress_result: Optional[Dict] = None,
    model: Any = None,
    context_tokens: Optional[int] = None,
    gen_tokens: Optional[int] = None,
    gen_tps: Optional[float] = None,
    ttft_ms: Optional[float] = None,
    comp_ms: Optional[float] = None,
    response: Optional[str] = None,
    passed: Optional[bool] = None,
    notes: Optional[str] = None,
    bits: int = 4,
    **extra,
) -> str:
    """
    Save experiment result to results/ directory.

    Args:
        model_name: Model identifier (e.g. "Qwen3.5-2B").
        compress_result: Dict returned by compress_cache().
        model: MLX model object (for auto-detecting config).
        context_tokens: Number of input tokens.
        gen_tokens: Number of generated tokens.
        gen_tps: Generation throughput (tokens/sec).
        ttft_ms: Time to first token (ms).
        comp_ms: Compression time (ms). Overrides compress_result if set.
        response: Model output text.
        passed: Whether the test passed.
        notes: Free-text notes.
        bits: Quantization bits used.
        **extra: Any additional key-value pairs to save.

    Returns:
        Path to the saved JSON file.
    """
    now = datetime.now()
    record = {
        "timestamp": now.isoformat(),
        "model_name": model_name,
        "bits": bits,
    }

    # Auto-detect model config
    if model is not None:
        try:
            from . import get_model_config
            record["model_config"] = get_model_config(model)
        except Exception:
            pass

    # Hardware info
    record["hardware"] = _get_hardware()

    # Merge compress_cache() result
    if compress_result:
        record.update(compress_result)

    # User-provided metrics (override compress_result if set)
    if context_tokens is not None:
        record["context_tokens"] = context_tokens
    if gen_tokens is not None:
        record["gen_tokens"] = gen_tokens
    if gen_tps is not None:
        record["gen_tps"] = gen_tps
    if ttft_ms is not None:
        record["ttft_ms"] = ttft_ms
    if comp_ms is not None:
        record["compress_ms"] = comp_ms
    if response is not None:
        record["response"] = response
    if passed is not None:
        record["passed"] = passed
    if notes is not None:
        record["notes"] = notes

    # Extra metrics
    record.update(extra)

    # Build filename
    ctx_part = f"_{context_tokens}tok" if context_tokens else ""
    ts = now.strftime("%Y%m%d_%H%M%S")
    short = _short_model_name(model_name)
    filename = f"{short}{ctx_part}_{ts}.json"

    results_dir = _get_results_dir()
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return str(filepath)


def list_experiments(model_filter: Optional[str] = None) -> List[Dict]:
    """
    List all saved experiment results.

    Args:
        model_filter: Optional substring to filter by model name.

    Returns:
        List of dicts with 'filename', 'model_name', 'timestamp', 'context_tokens',
        'cosine', 'passed' for each saved experiment, sorted newest first.
    """
    results_dir = _get_results_dir()
    entries = []

    for f in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        name = data.get("model_name", "")
        if model_filter and model_filter.lower() not in name.lower():
            continue

        entries.append({
            "filename": f.name,
            "model_name": name,
            "timestamp": data.get("timestamp", ""),
            "context_tokens": data.get("context_tokens"),
            "cosine": data.get("cosine"),
            "compress_ms": data.get("compress_ms"),
            "saved_mb": data.get("saved_mb"),
            "gen_tps": data.get("gen_tps"),
            "passed": data.get("passed"),
        })

    return entries


def load_experiment(filename: str) -> Dict:
    """
    Load a specific experiment result.

    Args:
        filename: Filename (not full path) of the result JSON.

    Returns:
        Full experiment dict.
    """
    results_dir = _get_results_dir()
    filepath = results_dir / filename

    with open(filepath) as f:
        return json.load(f)
