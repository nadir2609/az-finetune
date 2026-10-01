"""Turn loop: listen -> transcribe -> stream the LLM reply through TTS to the speakers (half-duplex)."""
import queue
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from assistant.audio_io import OUT_RATE
from assistant.chunker import SentenceChunker
from assistant.llm import LLMError
from assistant.text_norm import to_spoken

FALLBACK_TEXT = "Bağışlayın, hazırda cavab verə bilmirəm. Zəhmət olmasa, bir az sonra yenidən cəhd edin."
SECTIONS = ("tts", "stt", "llm", "vad", "chunker", "playback")
_TIMINGS = ("stt", "llm_first_chunk", "tts_first", "first_audio")
_END = object()


def load_config(path):
    """configs/assistant.yml as a dict; every section must be present."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [s for s in SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"{path}: missing section(s): {', '.join(missing)}")
    return cfg


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000)


class Pipeline:
    """mic, segmenter and stt are only needed for run()/step(); run_text() and reply() work without them."""

    def __init__(self, llm, tts, player, mic=None, segmenter=None, stt=None, chunker=None,
                 gap_ms=150, tail_ms=300, fallback_wav=None, log=print, sleep=time.sleep):
        self.llm, self.tts, self.player = llm, tts, player
        self.mic, self.segmenter, self.stt = mic, segmenter, stt
        self.chunker_cfg = chunker or {}
        self.gap = np.zeros(int(OUT_RATE * gap_ms / 1000), dtype=np.float32)
        self.tail_s = tail_ms / 1000
        self.fallback_wav, self.log, self.sleep = fallback_wav, log, sleep
        self.state = "LISTENING"

    def run(self):
        self.mic.start()
        try:
            while True:
                self.step()
        finally:
            self.mic.stop()

    def run_text(self, lines):
        """Typed turns instead of the microphone (debugging: no mic, VAD or STT involved)."""
        for line in lines:
            self.reply(line)

    def step(self):
        """One microphone turn. Returns the reply text, or None if nothing intelligible was heard."""
        self.state = "LISTENING"
        utterance = self._listen()
        t0 = time.perf_counter()
        self.state = "THINKING"
        try:
            text = self.stt.transcribe(utterance)
        except Exception as e:
            self.log(f"[stt] {type(e).__name__}: {e}")
            text = ""
        if not text:
            return None
        try:
            return self.reply(text, t0, {"stt": _ms(t0)})
        finally:  # half-duplex: forget everything the mic picked up while the assistant was speaking
            self.sleep(self.tail_s)
            self.mic.drain()
            self.segmenter.reset()

    def _listen(self):
        while True:
            block = self.mic.read(timeout=0.1)
            if block is None:
                continue
            utterance = self.segmenter.feed(block)
            if utterance is not None:
                return utterance

    def reply(self, user_text, t0=None, marks=None):
        """Speak the LLM's reply to user_text; returns its text ("" if the LLM failed before any)."""
        if not user_text.strip():
            return None
        t0 = time.perf_counter() if t0 is None else t0
        marks = {} if marks is None else marks
        self.state = "THINKING"
        self.log(f"Siz: {user_text}")
        text_q, audio_q = queue.Queue(), queue.Queue()
        workers = [threading.Thread(target=self._tts_worker, args=(text_q, audio_q, marks, t0), daemon=True),
                   threading.Thread(target=self._play_worker, args=(audio_q, marks, t0), daemon=True)]
        for w in workers:
            w.start()
        chunker, said, queued = SentenceChunker(**self.chunker_cfg), [], 0
        try:
            for delta in self.llm.stream_reply(user_text):
                said.append(delta)
                for chunk in chunker.feed(delta):
                    marks.setdefault("llm_first_chunk", _ms(t0))
                    text_q.put(chunk)
                    queued += 1
            for chunk in chunker.flush():
                marks.setdefault("llm_first_chunk", _ms(t0))
                text_q.put(chunk)
                queued += 1
        except LLMError as e:  # chunks already queued are still spoken
            self.log(f"[llm] {type(e).__name__}: {e}")
            if not queued and self.fallback_wav is not None:
                text_q.put(self.fallback_wav)
        finally:
            text_q.put(_END)
            for w in workers:
                w.join()
        text = "".join(said).strip()
        self.log(f"Köməkçi: {text}")
        self.log("[latency] " + " ".join(f"{k}={marks[k]}ms" for k in _TIMINGS if k in marks))
        return text

    def _tts_worker(self, text_q, audio_q, marks, t0):
        while (item := text_q.get()) is not _END:
            if isinstance(item, np.ndarray):  # pre-synthesized audio (the fallback phrase)
                audio_q.put(item)
                continue
            try:
                spoken = to_spoken(item)
                if spoken:
                    wav = self.tts.synthesize(spoken)
                    marks.setdefault("tts_first", _ms(t0))
                    audio_q.put(wav)
            except Exception as e:
                self.log(f"[tts] skipped {item!r}: {type(e).__name__}: {e}")
        audio_q.put(_END)

    def _play_worker(self, audio_q, marks, t0):
        first = True
        while (wav := audio_q.get()) is not _END:
            try:
                if first:
                    marks["first_audio"] = _ms(t0)
                    self.state = "SPEAKING"
                else:
                    self.player.play(self.gap)
                first = False
                self.player.play(wav)
            except Exception as e:
                self.log(f"[audio] {type(e).__name__}: {e}")
