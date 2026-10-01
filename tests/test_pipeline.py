import numpy as np
import pytest

from assistant.chunker import SentenceChunker
from assistant.llm import LLMError
from assistant.pipeline import SECTIONS, Pipeline, load_config
from assistant.vad import VadConfig
from conftest import ROOT

FALLBACK = np.full(7, 99, dtype=np.float32)


class FakeLLM:
    """stream_reply yields the scripted items; an exception item is raised at that point."""

    def __init__(self, *script):
        self.script, self.calls = script, []

    def stream_reply(self, user_text):
        self.calls.append(user_text)
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            yield item


class FakeTTS:
    """The nth synthesized chunk is 4 samples of the value n (1-based)."""

    def __init__(self, fail_on=None):
        self.texts, self.fail_on = [], fail_on

    def synthesize(self, text):
        if text == self.fail_on:
            raise RuntimeError("boom")
        self.texts.append(text)
        return np.full(4, len(self.texts), dtype=np.float32)


class FakePlayer:
    def __init__(self, pipeline_ref, fail=False):
        self.played, self.states, self.ref, self.fail = [], [], pipeline_ref, fail

    def play(self, wav):
        if self.fail:
            raise OSError("device unplugged")
        self.states.append(self.ref[0].state)
        self.played.append(wav)

    def speech(self):
        """Values of the non-silent buffers played, in order."""
        return [int(w[0]) for w in self.played if w.any()]


class FakeMic:
    def __init__(self):
        self.drained = self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def read(self, timeout=None):
        return np.zeros(512, dtype=np.float32)

    def drain(self):
        self.drained += 1


class FakeSegmenter:
    def __init__(self):
        self.resets = 0

    def feed(self, block):
        return np.zeros(1024, dtype=np.float32)

    def reset(self):
        self.resets += 1


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def make(llm, tts=None, heard="Salam", player_fails=False):
    ref, logs, sleeps = [], [], []
    tts = tts or FakeTTS()
    player = FakePlayer(ref, fail=player_fails)
    p = Pipeline(llm, tts, player, mic=FakeMic(), segmenter=FakeSegmenter(), stt=FakeSTT(heard),
                 chunker={"min_chars": 1, "max_chars": 180}, gap_ms=150, tail_ms=300,
                 fallback_wav=FALLBACK, log=logs.append, sleep=sleeps.append)
    ref.append(p)
    return p, tts, player, logs, sleeps


def test_chunks_are_spoken_in_order_with_gaps_between_them():
    p, tts, player, _, _ = make(FakeLLM("Birinci cümlə. İkin", "ci cümlə. ", "Üçüncü."))
    assert p.step() == "Birinci cümlə. İkinci cümlə. Üçüncü."
    assert tts.texts == ["birinci cümlə.", "ikinci cümlə.", "üçüncü."]
    assert player.speech() == [1, 2, 3]
    assert [len(w) for w in player.played] == [4, 3600, 4, 3600, 4]  # 150 ms of silence at 24 kHz
    assert set(player.states) == {"SPEAKING"}


def test_after_speaking_the_mic_is_drained_and_the_segmenter_reset():
    p, _, _, _, sleeps = make(FakeLLM("Bəli."))
    p.step()
    assert sleeps == [0.3]
    assert (p.mic.drained, p.segmenter.resets) == (1, 1)


def test_empty_transcript_skips_the_llm():
    llm = FakeLLM("Bəli.")
    p, _, player, _, _ = make(llm, heard="")
    assert p.step() is None
    assert llm.calls == [] and player.played == []


def test_stt_failure_is_treated_as_nothing_heard():
    llm = FakeLLM("Bəli.")
    p, _, _, logs, _ = make(llm, heard=RuntimeError("cuda"))
    assert p.step() is None
    assert llm.calls == [] and any("[stt]" in line for line in logs)


def test_llm_failure_before_any_text_plays_the_fallback_phrase():
    p, tts, player, logs, _ = make(FakeLLM(LLMError("quota")))
    assert p.step() == ""
    assert tts.texts == [] and player.speech() == [99]
    assert any("[llm] LLMError: quota" in line for line in logs)


def test_llm_failure_mid_stream_speaks_the_finished_chunks_only():
    p, tts, player, _, _ = make(FakeLLM("Birinci cümlə. İkinci yarım", LLMError("503")))
    p.step()
    assert tts.texts == ["birinci cümlə."] and player.speech() == [1]


def test_a_chunk_that_fails_in_tts_is_skipped():
    p, tts, player, logs, _ = make(FakeLLM("Birinci. İkinci. Üçüncü."), tts=FakeTTS(fail_on="ikinci."))
    p.step()
    assert tts.texts == ["birinci.", "üçüncü."] and player.speech() == [1, 2]
    assert any("[tts] skipped 'İkinci.'" in line for line in logs)


def test_a_chunk_with_nothing_speakable_never_reaches_tts():
    p, tts, _, _, _ = make(FakeLLM("😀. Salam dostum."))
    p.step()
    assert tts.texts == ["salam dostum."]


def test_playback_failure_ends_the_turn_instead_of_hanging():
    p, _, _, logs, _ = make(FakeLLM("Birinci. İkinci."), player_fails=True)
    assert p.step() == "Birinci. İkinci."
    assert p.mic.drained == 1 and any("[audio] OSError" in line for line in logs)


def test_text_mode_ignores_blank_lines_and_needs_no_microphone():
    llm = FakeLLM("Bəli.")
    ref = []
    player = FakePlayer(ref)
    p = Pipeline(llm, FakeTTS(), player, log=lambda line: None)
    ref.append(p)
    p.run_text(["", "   ", "Salam"])
    assert llm.calls == ["Salam"] and player.speech() == [1]


def test_latency_line_reports_each_stage():
    p, _, _, logs, _ = make(FakeLLM("Bəli."))
    p.step()
    line = next(x for x in logs if x.startswith("[latency]"))
    assert [part.split("=")[0] for part in line.split()[1:]] == ["stt", "llm_first_chunk", "tts_first", "first_audio"]


def test_run_starts_and_stops_the_microphone():
    p, _, _, _, _ = make(FakeLLM(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        p.run()
    assert (p.mic.started, p.mic.stopped) == (1, 1)


def test_load_config_names_missing_sections(tmp_path):
    path = tmp_path / "assistant.yml"
    path.write_text("tts: {}\nstt: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="llm, vad, chunker, playback"):
        load_config(path)


def test_shipped_config_matches_the_constructors():
    cfg = load_config(ROOT / "configs/assistant.yml")
    assert set(cfg) == set(SECTIONS)
    VadConfig(**cfg["vad"])
    SentenceChunker(**cfg["chunker"])
    assert set(cfg["llm"]) == {"model", "system_prompt", "max_turns", "max_output_tokens", "thinking_budget"}
    assert set(cfg["stt"]) == {"model", "device", "compute_type", "initial_prompt"}
    assert set(cfg["playback"]) == {"gap_ms", "tail_ms"}
    assert set(cfg["tts"]) == {"checkpoint"}
