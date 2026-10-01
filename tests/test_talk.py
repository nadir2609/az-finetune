import pytest

import talk
from conftest import ROOT


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
