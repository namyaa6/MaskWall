import pytest
from fastapi.testclient import TestClient

from app.config import ConfigError, Settings
from app.main import create_app


def env(**kw):
    return {k: str(v) for k, v in kw.items()}


def test_defaults_server_model_is_minilm():
    s = Settings.from_env(env={}, dotenv_path=None)
    assert s.server_model == "minilm"
    assert s.scan_mode == "sliding" and s.max_chars == 8000 and s.llm_order == ("groq", "gemini")
    assert s.model_path.name == "minilm"


def test_server_model_switch_selects_directory():
    s = Settings.from_env(env=env(MASKWALL_SERVER_MODEL="DeBERTa"), dotenv_path=None)  # case-insensitive
    assert s.server_model == "deberta" and s.model_path.name == "deberta"


@pytest.mark.parametrize("bad", [
    dict(MASKWALL_SERVER_MODEL="gpt"), dict(MASKWALL_SCAN_MODE="full"), dict(MASKWALL_LLM_ORDER="groq,openai"),
    dict(MASKWALL_MAX_WINDOWS="0"), dict(MASKWALL_MAX_CHARS="abc"), dict(MASKWALL_BLOCK_THRESHOLD="1.5"), dict(MASKWALL_LLM_ORDER=","),
])
def test_invalid_config_raises(bad):
    with pytest.raises(ConfigError):
        Settings.from_env(env=env(**bad), dotenv_path=None)


def test_env_file_is_fallback_and_environment_wins(tmp_path):
    f = tmp_path / ".env"
    f.write_text("GROQ_API_KEY=from-file\nGEMINI_API_KEY=gem-file\nMASKWALL_SCAN_MODE=head_tail\n")
    s = Settings.from_env(env=env(GROQ_API_KEY="from-env"), dotenv_path=f)
    assert s.groq_api_key == "from-env"          # real environment variable wins
    assert s.gemini_api_key == "gem-file"        # file fills the gap
    assert s.scan_mode == "head_tail"


def test_missing_env_file_is_fine(tmp_path):
    assert Settings.from_env(env={}, dotenv_path=tmp_path / "nope.env").groq_api_key is None


def test_secrets_never_appear_in_repr():
    s = Settings(groq_api_key="SECRET-GROQ-123", gemini_api_key="SECRET-GEM-456")
    text = repr(s) + str(s)
    assert "SECRET-GROQ-123" not in text and "SECRET-GEM-456" not in text


def test_health(settings):
    with TestClient(create_app(settings)) as c:
        r = c.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["server_model"] == "minilm" and body["model_loaded"] is False
    assert body["llm_providers"] == ["groq", "gemini"]     # names only, never keys
    assert "key" not in str(body).lower().replace("llm_providers", "")


def test_health_lists_only_configured_providers(tmp_path):
    s = Settings(db_path=tmp_path / "e.sqlite", groq_api_key=None, gemini_api_key="x")
    with TestClient(create_app(s)) as c:
        assert c.get("/health").json()["llm_providers"] == ["gemini"]
