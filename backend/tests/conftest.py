"""Shared test fixtures. Tests never read the real .env, never touch the network, and never need the ONNX model."""
import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    # model_dir points at an empty folder: tests must behave the same whether or not the real model is installed locally
    return Settings(db_path=tmp_path / "events.sqlite", model_dir=tmp_path / "no_model", groq_api_key="test-groq-key", gemini_api_key="test-gemini-key")


def pytest_report_header(config):
    """Print the CPU and onnxruntime version at the top of every run, so a CI log always says what hardware produced the numbers
    (int8 scores differ slightly between CPUs: docs/decisions.md Day 4.9)."""
    import platform

    import onnxruntime
    model, flags = platform.processor() or platform.machine(), ""
    try:
        from pathlib import Path
        info = Path("/proc/cpuinfo").read_text().splitlines()
        model = next((ln.split(":", 1)[1].strip() for ln in info if ln.startswith("model name")), model)
        raw = next((ln.split(":", 1)[1].split() for ln in info if ln.startswith("flags")), [])
        flags = " ".join(f for f in raw if f.startswith(("avx", "amx", "sse4", "fma")) or "vnni" in f)
    except OSError:
        pass
    return f"cpu: {model} | isa flags: {flags or 'n/a'} | onnxruntime {onnxruntime.__version__}"
