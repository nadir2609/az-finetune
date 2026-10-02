import sys
import types

import pytest

import talk
from conftest import ROOT


@pytest.fixture(autouse=True)
def no_real_env_file(monkeypatch, tmp_path):
    """Tests must never read the developer's real .env (it holds the API key)."""
    monkeypatch.setattr(talk, "ENV_FILE", tmp_path / ".env")


def test_missing_api_key_exits_before_loading_any_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        talk.main(["--text", "--config", str(ROOT / "configs/assistant.yml")])


def test_broken_config_is_reported_before_the_key_check(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    bad = tmp_path / "assistant.yml"
    bad.write_text("tts: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing section"):
        talk.main(["--config", str(bad)])


def test_list_devices_needs_no_config_or_key(monkeypatch):
    called = []
    monkeypatch.setattr(talk, "list_devices", lambda: called.append(1))
    talk.main(["--list-devices", "--config", "does-not-exist.yml"])
    assert called == [1]


def test_stdin_lines_stops_at_end_of_input(monkeypatch):
    lines = iter(["Salam", "Sağ ol"])

    def fake_input(prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    assert list(talk.stdin_lines()) == ["Salam", "Sağ ol"]


def test_env_file_values_are_loaded_without_overriding_the_shell(tmp_path):
    path = tmp_path / ".env"
    path.write_text("\n".join(["# comment", "", "GEMINI_API_KEY = 'from-file'", 'export OTHER="x y"', "PLAIN=1",
                               "not a pair"]), encoding="utf-8")
    environ = {"PLAIN": "from-shell"}
    talk.load_env_file(path, environ)
    assert environ == {"PLAIN": "from-shell", "GEMINI_API_KEY": "from-file", "OTHER": "x y"}


def test_env_file_with_a_utf8_bom_is_read(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=abc\n", encoding="utf-8-sig")  # what Notepad and PowerShell 5.1 write
    environ = {}
    talk.load_env_file(path, environ)
    assert environ == {"GEMINI_API_KEY": "abc"}


def test_missing_env_file_is_fine(tmp_path):
    environ = {}
    talk.load_env_file(tmp_path / "nope.env", environ)
    assert environ == {}


def test_key_from_the_env_file_passes_the_key_check(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")  # registered with monkeypatch, so the loaded value is undone
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    talk.ENV_FILE.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")

    class Reached(Exception):
        pass

    def fake_tts(**kwargs):
        raise Reached

    monkeypatch.setitem(sys.modules, "assistant.tts", types.SimpleNamespace(TTSService=fake_tts))
    with pytest.raises(Reached):  # got past the key check and on to loading the models
        talk.main(["--text", "--config", str(ROOT / "configs/assistant.yml")])
