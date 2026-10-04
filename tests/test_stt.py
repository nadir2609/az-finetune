import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from assistant.stt import WhisperSTT, clean_transcript, ct2_model

PROMPT = "Salam, necəsiniz? Bu gün sizə necə kömək edə bilərəm?"


def seg(text, no_speech_prob=0.1):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)


def test_segments_are_joined_and_stripped():
    assert clean_transcript([seg(" Salam."), seg(" Hava necədir? ")]) == "Salam. Hava necədir?"


def test_segments_marked_as_non_speech_are_dropped():
    assert clean_transcript([seg(" Salam.", 0.2), seg(" Altyazı.", 0.9)]) == "Salam."


def test_no_segments_is_empty():
    assert clean_transcript(iter([])) == ""


def test_text_without_letters_is_empty():
    assert clean_transcript([seg(" ... "), seg("?!")]) == ""


def test_an_echo_of_the_initial_prompt_is_empty():
    assert clean_transcript([seg(" salam necəsiniz"), seg("Bu gün sizə necə kömək edə bilərəm")], PROMPT) == ""
    assert clean_transcript([seg(" Salam, necəsiniz?")], PROMPT) == "Salam, necəsiniz?"


def test_builtin_names_and_local_directories_are_loaded_as_given(tmp_path):
    assert ct2_model("large-v3-turbo", tmp_path) == "large-v3-turbo"
    assert ct2_model(f"{tmp_path.as_posix()}/", tmp_path / "stt") == f"{tmp_path.as_posix()}/"


def test_a_converted_repo_is_reused(tmp_path):
    converted = tmp_path / "LocalDoc--azerbaijani-whisper-turbo"
    converted.mkdir()
    (converted / "model.bin").write_bytes(b"")
    assert ct2_model("LocalDoc/azerbaijani-whisper-turbo", tmp_path) == str(converted)


def test_concurrent_transcriptions_take_turns(monkeypatch):
    active, peak = [0], [0]

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            def segments():  # faster-whisper decodes lazily, while the segments are read
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                time.sleep(0.05)
                active[0] -= 1
                yield seg(" Salam.")
            return segments(), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    stt = WhisperSTT(device="cpu")
    results = []
    threads = [threading.Thread(target=lambda: results.append(stt.transcribe(np.zeros(16000, dtype=np.float32))))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["Salam."] * 4 and peak[0] == 1
