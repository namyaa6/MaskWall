"""Shared test fixtures. Tests never read the real .env, never touch the network, and never need the ONNX model."""
import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    # model_dir points at an empty folder: tests must behave the same whether or not the real model is installed locally
    return Settings(db_path=tmp_path / "events.sqlite", model_dir=tmp_path / "no_model", groq_api_key="test-groq-key", gemini_api_key="test-gemini-key")
