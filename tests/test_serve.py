import sys
import types

import pytest

import serve
import talk
from conftest import ROOT

CONFIG = ["--config", str(ROOT / "configs/assistant.yml")]


@pytest.fixture(autouse=True)
def no_real_env_file(monkeypatch, tmp_path):
    """Tests must never read the developer's real .env (it holds the API key and the access code)."""
    monkeypatch.setattr(talk, "ENV_FILE", tmp_path / ".env")
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ASSISTANT_ACCESS_CODE"):
        monkeypatch.delenv(name, raising=False)


def test_missing_api_key_exits_before_loading_any_model():
    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        serve.main(CONFIG)


def test_missing_access_code_exits_before_loading_any_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(SystemExit, match="ASSISTANT_ACCESS_CODE is not set"):
        serve.main(CONFIG)


def test_key_and_code_from_the_env_file_reach_model_loading(monkeypatch):
    for name in ("GEMINI_API_KEY", "ASSISTANT_ACCESS_CODE"):
        monkeypatch.setenv(name, "")  # registered with monkeypatch, so the loaded values are undone
        monkeypatch.delenv(name)
    talk.ENV_FILE.write_text("GEMINI_API_KEY=test-key\nASSISTANT_ACCESS_CODE=gizli\n", encoding="utf-8")

    class Reached(Exception):
        pass

    def fake_tts(**kwargs):
        raise Reached

    monkeypatch.setitem(sys.modules, "assistant.tts", types.SimpleNamespace(TTSService=fake_tts))
    with pytest.raises(Reached):
        serve.main(CONFIG)
