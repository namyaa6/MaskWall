"""Shared test fixtures. Tests never read the real .env, never touch the network, and never need the ONNX model."""
import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=tmp_path / "events.sqlite", groq_api_key="test-groq-key", gemini_api_key="test-gemini-key")
