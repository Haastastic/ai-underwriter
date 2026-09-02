"""Backend settings + .env loading."""

import os

from app.backend.config import load_env_file, settings_from_env


def test_load_env_file_missing_returns_false(tmp_path):
    assert load_env_file(tmp_path / "absent.env") is False


def test_load_env_file_populates_environ(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AIU_TEST_MARKER=from_dotenv\nANTHROPIC_API_KEY=sk-test-xyz\n")
    try:
        assert load_env_file(env) is True
        assert os.environ["AIU_TEST_MARKER"] == "from_dotenv"
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-xyz"
    finally:  # load_dotenv writes straight to os.environ; clean up ourselves
        os.environ.pop("AIU_TEST_MARKER", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_real_env_var_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AIU_MODEL_VERSION=from_dotenv\n")
    monkeypatch.setenv("AIU_MODEL_VERSION", "from_real_env")

    load_env_file(env)

    assert os.environ["AIU_MODEL_VERSION"] == "from_real_env"


def test_settings_from_env_applies_overrides(monkeypatch):
    monkeypatch.setenv("AIU_MODEL_VERSION", "v7")
    monkeypatch.setenv("AIU_DENY_AT_OR_ABOVE", "0.42")
    monkeypatch.setenv("AIU_MAX_REASONS", "2")

    settings = settings_from_env()

    assert settings.model_version == "v7"
    assert settings.deny_at_or_above == 0.42
    assert settings.max_reasons == 2


def test_settings_from_env_defaults(monkeypatch):
    for key in (
        "AIU_MODELS_ROOT",
        "AIU_MODEL_VERSION",
        "AIU_DB_PATH",
        "AIU_APPROVE_BELOW",
        "AIU_DENY_AT_OR_ABOVE",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = settings_from_env()

    assert settings.model_version == "v1"
    assert settings.approve_below == 0.08
    assert settings.deny_at_or_above == 0.30
