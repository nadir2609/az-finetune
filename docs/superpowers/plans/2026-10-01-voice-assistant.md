# Azerbaijani Voice Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A terminal program on this Windows machine that listens to Azerbaijani speech, gets a reply from Gemini, and speaks it with the project's fine-tuned StyleTTS2 voice, sentence by sentence.

**Architecture:** A new `assistant/` package of small single-purpose modules (VAD segmenter, STT, LLM client, sentence chunker, text normalizer, TTS service, audio I/O) wired by a half-duplex `Pipeline` that uses two worker threads and two queues, so TTS and playback overlap with the LLM stream. Existing scripts are not modified; `assistant/tts.py` reuses `Synth` from `scripts/synthesize.py`. Every part is injected into `Pipeline`, so unit tests use fakes and need no GPU, microphone or network.

**Tech Stack:** Python 3.10 venv, PyTorch 2.11.0+cu128 (installed), faster-whisper (`large-v3-turbo`), silero-vad, sounddevice, google-genai (Gemini free plan), pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-voice-assistant-design.md`

## Global Constraints

- Branch: `feat/voice-assistant`. Commit after each task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The working tree has unrelated uncommitted changes (`configs/calibration.json`, `configs/config_ft_az.yml`, `configs/inference.json`, `data/*`, `metadata.csv`, `run_tests.py`, `scripts/batch_synth.py`, `test_samples/`, `audio_test.zip`). Never stage, revert or edit them: always `git add <the exact paths the task names>`, never `git add -A` or `git add .`.
- Python: always `.venv\Scripts\python.exe` (3.10). Never the system Python 3.14.
- Do not modify anything under `scripts/` except creating `scripts/talk.py`; do not touch `StyleTTS2/`, training configs, or the TTS model.
- New dependencies go only in `requirements-assistant.txt`. After installing, torch must still be `2.11.0+cu128`.
- The Gemini key is read only from the `GEMINI_API_KEY` environment variable. Never write it to a file, a config, a log line or a commit.
- All files written by our code use UTF-8 explicitly (`encoding="utf-8"`). Do not put non-ASCII characters in shell command lines (PowerShell mangles `ə`); put Azerbaijani text in files.
- Unit tests must not need a GPU, a microphone, speakers or the network.
- Audio formats: microphone 16 kHz mono float32 in 512-sample blocks; TTS output and playback 24 kHz mono float32.
- Half-duplex only. Out of scope: barge-in, echo cancellation, browser or telephony clients, wake word, tools or retrieval, STT fine-tuning.
- Do not run the assistant while a training run is using the GPU.

## Review Focus

Inputs the spec implies but does not spell out; each has a test in the task that owns the code.

- Gemini returns a reply with no text at all (safety block, token limit): the user hears the fallback phrase and the history is left unchanged (Task 4 `test_reply_without_any_text_is_an_error_and_leaves_history_unchanged`, Task 8 `test_llm_failure_before_any_text_plays_the_fallback_phrase`).
- Whisper turns noise into punctuation or an echo of its prompt: treated as nothing heard, no LLM call (Task 5 `test_text_without_letters_is_empty`, `test_an_echo_of_the_initial_prompt_is_empty`).
- The LLM writes a long run with no sentence punctuation, or one unbroken token: it is still cut into chunks of at most `max_chars` (Task 2 `test_long_text_without_commas_is_split_at_a_space`, `test_unbroken_text_longer_than_the_limit_is_hard_cut`).
- A chunk with nothing speakable (emoji only): skipped without calling the TTS (Task 8 `test_a_chunk_with_nothing_speakable_never_reaches_tts`).
- The speaker device fails mid-reply: the turn ends and listening resumes instead of hanging (Task 8 `test_playback_failure_ends_the_turn_instead_of_hanging`).

## Refinements to the spec made in this plan

- The LLM stream is consumed on the pipeline's own thread; only TTS and playback are worker threads. Behaviour is the same as the spec's three workers.
- The fallback phrase is also played when the LLM fails after producing some text but before any complete chunk, so the user never gets silence.
- After a mid-stream LLM failure, the user turn and the partial reply stay in the history; only a failure with no text removes the user turn.
- `stt.py` also treats an echo of the Whisper prompt as silence.
- Extra unit tests beyond the spec's list: `test_stt.py`, `test_audio_io.py`, `test_tts.py`, `test_talk.py`.

## File map

| File | Responsibility |
|---|---|
| `requirements-assistant.txt` | the assistant's extra dependencies |
| `assistant/__init__.py` | package root; puts `scripts/` on `sys.path` |
| `assistant/text_norm.py` | `to_spoken`, `number_to_words`: LLM text to G2P-safe Azerbaijani |
| `assistant/chunker.py` | `SentenceChunker`: streamed deltas to speakable chunks |
| `assistant/vad.py` | `Segmenter`, `VadConfig`, `silero_segmenter`: mic blocks to utterances |
| `assistant/llm.py` | `GeminiLLM`, `LLMError`, `LLMRateLimited`: streaming chat with history |
| `assistant/stt.py` | `WhisperSTT`, `clean_transcript`: utterance to text |
| `assistant/audio_io.py` | `MicStream`, `Player`, `list_devices`, `OUT_RATE` |
| `assistant/tts.py` | `TTSService`, `resolve_checkpoint`: load-once StyleTTS2 voice |
| `assistant/pipeline.py` | `Pipeline`, `load_config`, `FALLBACK_TEXT`: the turn loop |
| `configs/assistant.yml` | all settings |
| `scripts/talk.py` | command-line entry point |
| `tests/conftest.py` | modified: also puts the repo root on `sys.path` |
| `tests/test_*.py` | one test file per module above |
| `README.md` | modified: new "Voice assistant" section |

All test commands are run from the repo root in PowerShell.

---

### Task 1: Package scaffold, dependencies and the text normalizer

**Files:**
- Create: `requirements-assistant.txt`, `assistant/__init__.py`, `assistant/text_norm.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_text_norm.py`

**Interfaces:**
- Consumes: `az_text.normalize_text(text) -> str` and `az_g2p.text_to_ipa(text) -> str` from `scripts/` (existing).
- Produces: `assistant.ROOT: Path` (repo root); `assistant.text_norm.number_to_words(n: int) -> str` (raises `ValueError` outside 0..999 999 999 999); `assistant.text_norm.to_spoken(text: str) -> str` (never raises; returns `""` when nothing speakable is left).

- [ ] **Step 1: Create `requirements-assistant.txt`**

```text
faster-whisper
silero-vad
sounddevice
google-genai<3
```

- [ ] **Step 2: Install and verify torch was not replaced**

Run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-assistant.txt
.\.venv\Scripts\python.exe -c "import torch, faster_whisper, silero_vad, sounddevice; from google import genai; print(torch.__version__, torch.cuda.is_available())"
```
Expected: `2.11.0+cu128 True`. If the torch version changed, stop and report it; do not continue.

- [ ] **Step 3: Replace `tests/conftest.py` so tests can import the `assistant` package**

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
```

- [ ] **Step 4: Create `assistant/__init__.py`**

```python
"""Azerbaijani voice-to-voice assistant: mic -> VAD -> STT -> LLM -> StyleTTS2 -> speakers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))  # az_text, synthesize (same convention as tests/conftest.py)
```

- [ ] **Step 5: Write the failing tests in `tests/test_text_norm.py`**

```python
import pytest

from assistant.text_norm import number_to_words, to_spoken


@pytest.mark.parametrize("n, words", [
    (0, "sıfır"), (7, "yeddi"), (10, "on"), (19, "on doqquz"), (40, "qırx"), (100, "yüz"),
    (150, "yüz əlli"), (999, "doqquz yüz doxsan doqquz"), (1000, "min"), (1500, "min beş yüz"),
    (2026, "iki min iyirmi altı"), (21000, "iyirmi bir min"), (100000, "yüz min"),
    (1_000_000, "bir milyon"), (2_000_300, "iki milyon üç yüz"), (3_000_000_000, "üç milyard"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_number_to_words_rejects_out_of_range():
    with pytest.raises(ValueError):
        number_to_words(10 ** 12)
    with pytest.raises(ValueError):
        number_to_words(-1)


@pytest.mark.parametrize("text, spoken", [
    ("Qiymət 150 manatdır.", "qiymət yüz əlli manatdır."),
    ("Saat 18:30-da görüşək.", "saat on səkkiz otuzda görüşək."),
    ("Saat 9:05", "saat doqquz sıfır beş"),
    ("3,5 faiz", "üç vergül beş faiz"),
    ("3.5 faiz", "üç vergül beş faiz"),
    ("Endirim 50%-dir.", "endirim əlli faizdir."),
    ("20 AZN", "iyirmi manat"),
    ("$5", "beş dollar"),
    ("100₼", "yüz manat"),
    ("7 €", "yeddi avro"),
    ("5-ci mərtəbə", "beşinci mərtəbə"),
    ("3-cü", "üçüncü"),
    ("6-cı", "altıncı"),
    ("10-cu", "onuncu"),
    ("2-ci", "ikinci"),
    ("30-da", "otuzda"),
    ("25-də", "iyirmi beşdə"),
    ("1 500 nəfər", "min beş yüz nəfər"),
    ("1.500 nəfər", "min beş yüz nəfər"),
    ("1234567890123", "bir iki üç dörd beş altı yeddi səkkiz doqquz sıfır bir iki üç"),
    ("5kq", "beş kq"),
])
def test_numbers_become_words(text, spoken):
    assert to_spoken(text) == spoken


@pytest.mark.parametrize("text, spoken", [
    ("**Salam!** Bu [sayt](https://example.az) yaxşıdır 😀", "salam! bu sayt yaxşıdır"),
    ("Bu (yəni o) yaxşıdır.", "bu, yəni o, yaxşıdır."),
    ("- birinci\n- ikinci", "birinci, ikinci"),
    ("Salam — necəsən?", "salam, necəsən?"),
    ("Bəli!!!", "bəli!"),
    ("WiFi café", "vifi cafe"),
    ("İSTİ ILIQ", "isti ılıq"),
])
def test_markup_and_letters(text, spoken):
    assert to_spoken(text) == spoken


@pytest.mark.parametrize("text", ["😀👍", "...", "", "   ", "***", "— —"])
def test_nothing_speakable_gives_empty_string(text):
    assert to_spoken(text) == ""


@pytest.mark.parametrize("text", [
    "Tarix: 01.10.2026", "+994 50 123 45 67", "E-mail: test@example.com", "x² + y² = z²",
    "Привет, как дела?", "C++ və C# dilləri", "100%", "a/b\\c", "“Sitat” — dedi.", "1.2.3.4",
    "5-ci, 6-cı və 10-cu", "№ 5", "3 × 4 = 12", "naïve façade", "₼", "18:30:45", "١٢٣", "sosial-iqtisadi",
    "-5 dərəcə", "5-", "10-15 dəqiqə", "ǝ Ә", "a\tb\r\nc", "½ ¾", "𝟓",
])
def test_output_never_breaks_the_g2p(text):
    from az_g2p import text_to_ipa
    from az_text import normalize_text
    text_to_ipa(normalize_text(to_spoken(text)))  # must not raise
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_text_norm.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'assistant.text_norm'`.

- [ ] **Step 7: Create `assistant/text_norm.py`**

```python
"""Turn LLM output into text the Azerbaijani G2P accepts: numbers as words, no markdown, no foreign symbols."""
import re
import unicodedata

from az_text import normalize_text

_ONES = ["sıfır", "bir", "iki", "üç", "dörd", "beş", "altı", "yeddi", "səkkiz", "doqquz"]
_TENS = ["", "on", "iyirmi", "otuz", "qırx", "əlli", "altmış", "yetmiş", "səksən", "doxsan"]
_SCALES = [(10 ** 9, "milyard"), (10 ** 6, "milyon"), (10 ** 3, "min")]
MAX_NUMBER = 999_999_999_999

_LETTERS = set("abcçdeəfgğhxıijkqlmnoöprsştuüvyz")
_KEEP = _LETTERS | set(" ,.!?;:-")
_PUNCT = ",.!?;:"
_LETTER = r"[^\W\d_]"
_SUFFIX = rf"(?:-(?={_LETTER}))?"  # the hyphen of "30-da": dropped, so the suffix joins the number word
_NUM = r"\d+(?:[.,]\d+)?"
_UNITS = [("%", "faiz"), ("₼", "manat"), (rf"(?<!{_LETTER})AZN\b", "manat"), (r"\$", "dollar"), ("€", "avro")]
_HIGH = {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "ə": "i", "i": "i", "ö": "ü", "ü": "ü"}


def _below_thousand(n):
    words = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += ([_ONES[hundreds]] if hundreds > 1 else []) + ["yüz"]
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if ones:
        words.append(_ONES[ones])
    return words


def number_to_words(n):
    """Azerbaijani cardinal for 0 <= n <= MAX_NUMBER."""
    if not 0 <= n <= MAX_NUMBER:
        raise ValueError(f"number out of range: {n}")
    if n == 0:
        return _ONES[0]
    words = []
    for value, name in _SCALES:
        count, n = divmod(n, value)
        if count:
            if not (value == 1000 and count == 1):  # 1000 is "min", not "bir min"
                words += _below_thousand(count)
            words.append(name)
    return " ".join(words + _below_thousand(n))


def _say(digits):
    """Words for a digit string; digit by digit if it has a leading zero or is above MAX_NUMBER."""
    if (len(digits) > 1 and digits[0] == "0") or int(digits) > MAX_NUMBER:
        return " ".join(_ONES[int(d)] for d in digits)
    return number_to_words(int(digits))


def _ordinal(words):
    vowel = _HIGH[[c for c in words if c in _HIGH][-1]]
    return words + ("nc" if words[-1] in _HIGH else vowel + "nc") + vowel


def _numbers(text):
    for symbol, word in _UNITS:
        text = re.sub(rf"({_NUM})\s*(?:{symbol}){_SUFFIX}", rf"\1 {word}", text, flags=re.I)
        text = re.sub(rf"(?:{symbol})\s*({_NUM})", rf"\1 {word}", text, flags=re.I)
        text = re.sub(symbol, f" {word} ", text, flags=re.I)
    text = re.sub(r"(?<![\d.,])\d{1,3}(?:[ .]\d{3})+(?!\d)", lambda m: re.sub(r"[ .]", "", m[0]), text)
    text = re.sub(rf"\b(\d{{1,2}}):(\d{{2}})\b{_SUFFIX}", lambda m: f"{_say(m[1])} {_say(m[2])}", text)
    text = re.sub(rf"(?<![\d.,])(\d+)[.,](\d+)(?![.,]?\d){_SUFFIX}",
                  lambda m: f"{_say(m[1])} vergül {_say(m[2])}", text)
    text = re.sub(rf"(\d+)-c[iıuü](?!{_LETTER})", lambda m: _ordinal(_say(m[1])), text, flags=re.I)
    text = re.sub(rf"(\d+)-(?={_LETTER})", lambda m: _say(m[1]), text)
    return re.sub(r"\d+", lambda m: f" {_say(m[0])} ", text)


def _markup(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"[*_#`~>|\\]", "", text)
    text = re.sub(r"[()\[\]{}]|\s*\n\s*", " , ", text)
    return re.sub(r"['’‘ʼ]", "", text)


def _letters(text):
    out = []
    for ch in text.replace("w", "v"):
        if ch not in _KEEP:
            ch = unicodedata.normalize("NFD", ch)[0]  # é -> e
        if ch in _KEEP:
            out.append(ch)
    return "".join(out)


def _strongest(match):
    run = re.sub(r"\s", "", match[0])
    return next((c for c in run if c in ".!?"), run[0])


def to_spoken(text):
    """Text that az_text.phonemize_texts accepts without raising; "" if nothing speakable is left."""
    text = _letters(normalize_text(_numbers(_markup(text))))
    text = re.sub(rf"(?<!{_LETTER})-|-(?!{_LETTER})", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(rf"\s+([{_PUNCT}])", r"\1", text)
    text = re.sub(rf"[{_PUNCT}](?:\s*[{_PUNCT}])+", _strongest, text)
    text = text.strip().lstrip(_PUNCT + " ")
    return text if re.search(_LETTER, text) else ""
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_text_norm.py -q`
Expected: `76 passed`.

- [ ] **Step 9: Run the whole suite to verify nothing else broke**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: `190 passed` (114 existing + 76 new).

- [ ] **Step 10: Commit**

```powershell
git add requirements-assistant.txt assistant/__init__.py assistant/text_norm.py tests/conftest.py tests/test_text_norm.py docs/superpowers/plans/2026-10-01-voice-assistant.md
git commit -m "feat(assistant): package scaffold and Azerbaijani text normalizer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Sentence chunker

**Files:**
- Create: `assistant/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `assistant.chunker.SentenceChunker(min_chars=25, max_chars=180)` with `feed(delta: str) -> list[str]` and `flush() -> list[str]`. Chunks are stripped, keep their final punctuation, and are never longer than `max_chars`.

- [ ] **Step 1: Write the failing tests in `tests/test_chunker.py`**

```python
from assistant.chunker import SentenceChunker


def feed_all(chunker, deltas):
    out = []
    for d in deltas:
        out += chunker.feed(d)
    return out


def test_sentence_is_emitted_once_its_end_is_followed_by_whitespace():
    c = SentenceChunker(min_chars=10, max_chars=180)
    assert c.feed("Bu gün hava çox gözəldir.") == []  # the stream could still continue with "5"
    assert c.feed(" Sabah") == ["Bu gün hava çox gözəldir."]
    assert c.flush() == ["Sabah"]


def test_boundary_split_across_deltas():
    c = SentenceChunker(min_chars=10, max_chars=180)
    assert feed_all(c, ["Bu gün hava", " çox gözəl", "dir", "! ", "Sabah da."]) == ["Bu gün hava çox gözəldir!"]
    assert c.flush() == ["Sabah da."]


def test_decimal_point_is_not_a_boundary():
    c = SentenceChunker(min_chars=5, max_chars=180)
    assert c.feed("Qiymət 3.5 manatdır. Sonra") == ["Qiymət 3.5 manatdır."]


def test_short_sentence_is_merged_with_the_next():
    c = SentenceChunker(min_chars=25, max_chars=180)
    assert c.feed("Bəli. Sizə kömək edə bilərəm. ") == ["Bəli. Sizə kömək edə bilərəm."]


def test_question_mark_is_kept():
    c = SentenceChunker(min_chars=5, max_chars=180)
    assert c.feed("Siz gələcəksinizmi? ") == ["Siz gələcəksinizmi?"]


def test_long_sentence_is_split_at_the_last_comma_before_the_limit():
    c = SentenceChunker(min_chars=5, max_chars=40)
    text = "Birinci hissə burada bitir, ikinci hissə isə burada davam edir. "
    assert c.feed(text) == ["Birinci hissə burada bitir,", "ikinci hissə isə burada davam edir."]


def test_long_text_without_commas_is_split_at_a_space():
    c = SentenceChunker(min_chars=5, max_chars=20)
    chunks = c.feed("bir iki üç dörd beş altı yeddi səkkiz doqquz on") + c.flush()
    assert all(len(x) <= 20 for x in chunks)
    assert " ".join(chunks) == "bir iki üç dörd beş altı yeddi səkkiz doqquz on"


def test_unbroken_text_longer_than_the_limit_is_hard_cut():
    c = SentenceChunker(min_chars=5, max_chars=10)
    chunks = c.feed("a" * 25) + c.flush()
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]


def test_flush_returns_short_remainder_and_empties_the_buffer():
    c = SentenceChunker()
    assert c.feed("Bəli.") == []
    assert c.flush() == ["Bəli."]
    assert c.flush() == []


def test_lines_without_punctuation_come_out_at_flush():
    c = SentenceChunker(min_chars=5, max_chars=180)
    assert c.feed("birinci sətir\nikinci sətir") == []
    assert c.flush() == ["birinci sətir\nikinci sətir"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_chunker.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.chunker'`.

- [ ] **Step 3: Create `assistant/chunker.py`**

```python
"""Cut a streamed LLM reply into chunks sized for the TTS model."""
import re

_END = re.compile(r"[.!?]+(?=\s)")


class SentenceChunker:
    def __init__(self, min_chars=25, max_chars=180):
        self.min_chars, self.max_chars = min_chars, max_chars
        self._buf = ""

    def feed(self, delta):
        """Add streamed text; returns the chunks that are now complete."""
        self._buf += delta
        out = []
        while True:
            end = self._boundary()
            if end is None:
                if len(self._buf) <= self.max_chars:
                    return out
                end = self._cut(self._buf)  # no sentence end in sight: cut at a comma or space
            piece, self._buf = self._buf[:end].strip(), self._buf[end:]
            out += self._split_long(piece)

    def flush(self):
        """End of stream: whatever is left, however short."""
        piece, self._buf = self._buf.strip(), ""
        return self._split_long(piece)

    def _boundary(self):
        for m in _END.finditer(self._buf):
            if len(self._buf[:m.end()].strip()) >= self.min_chars:
                return m.end()
        return None

    def _cut(self, text):
        window = text[:self.max_chars]
        i = max(window.rfind(","), window.rfind(";"))
        if i > 0:
            return i + 1
        i = window.rfind(" ")
        return i if i > 0 else self.max_chars

    def _split_long(self, piece):
        out = []
        while len(piece) > self.max_chars:
            cut = self._cut(piece)
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        return out + ([piece] if piece else [])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_chunker.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```powershell
git add assistant/chunker.py tests/test_chunker.py
git commit -m "feat(assistant): sentence chunker for streamed LLM text" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: VAD segmenter

**Files:**
- Create: `assistant/vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `assistant.vad.SAMPLE_RATE = 16000`, `BLOCK = 512`; `VadConfig(start_threshold, end_threshold, end_silence_ms, min_speech_ms, preroll_ms, max_utterance_s)` dataclass; `Segmenter(speech_prob, cfg=None, reset_model=None)` with `feed(block: np.ndarray) -> np.ndarray | None` and `reset()`; `silero_segmenter(cfg=None) -> Segmenter` (loads the Silero model; not unit-tested).

- [ ] **Step 1: Write the failing tests in `tests/test_vad.py`**

```python
import numpy as np

from assistant.vad import BLOCK, Segmenter, VadConfig

# 3 silent blocks end an utterance, 2 speech blocks are the minimum, 2 blocks of pre-roll
CFG = VadConfig(start_threshold=0.5, end_threshold=0.35, end_silence_ms=96, min_speech_ms=64,
                preroll_ms=64, max_utterance_s=30)


def run(probs, cfg=CFG):
    """Feed one block per probability (block n is filled with the value n); returns feed()'s results."""
    it = iter(probs)
    seg = Segmenter(lambda block: next(it), cfg)
    return [seg.feed(np.full(BLOCK, n, dtype=np.float32)) for n in range(len(probs))]


def block_ids(utterance):
    return utterance[::BLOCK].astype(int).tolist()


def test_utterance_includes_preroll_and_ends_after_the_silence_window():
    out = run([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert [o is not None for o in out] == [False] * 8 + [True, False]
    assert block_ids(out[8]) == [2, 3, 4, 5, 6, 7, 8]  # block 2 is pre-roll, block 3 triggered


def test_a_dip_between_the_thresholds_does_not_end_the_utterance():
    out = run([0.9, 0.4, 0.4, 0.4, 0.4, 0.9, 0.1, 0.1, 0.1])
    assert [o is not None for o in out] == [False] * 8 + [True]


def test_short_burst_is_discarded():
    out = run([0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert all(o is None for o in out)


def test_listening_resumes_after_an_utterance():
    out = run([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
    assert block_ids(out[4]) == [0, 1, 2, 3, 4]
    assert block_ids(out[10]) == [5, 6, 7, 8, 9, 10]


def test_utterance_is_cut_at_the_maximum_length():
    cfg = VadConfig(end_silence_ms=96, min_speech_ms=64, preroll_ms=64, max_utterance_s=0.16)  # 5 blocks
    out = run([0.9] * 8, cfg)
    assert [o is not None for o in out] == [False] * 4 + [True] + [False] * 3
    assert len(out[4]) == 5 * BLOCK


def test_reset_drops_a_partial_utterance_and_resets_the_model():
    calls = []
    it = iter([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    seg = Segmenter(lambda block: next(it), CFG, reset_model=lambda: calls.append(1))
    seg.feed(np.zeros(BLOCK, dtype=np.float32))
    seg.feed(np.zeros(BLOCK, dtype=np.float32))
    seg.reset()
    assert calls == [1]
    assert all(seg.feed(np.zeros(BLOCK, dtype=np.float32)) is None for _ in range(4))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_vad.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.vad'`.

- [ ] **Step 3: Create `assistant/vad.py`**

```python
"""Cut the microphone stream into utterances using per-block speech probabilities."""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 512  # samples per block: the size Silero VAD expects at 16 kHz (32 ms)
BLOCK_MS = 1000 * BLOCK / SAMPLE_RATE


@dataclass
class VadConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    end_silence_ms: int = 600
    min_speech_ms: int = 250
    preroll_ms: int = 300
    max_utterance_s: float = 30


class Segmenter:
    """feed() 512-sample float32 blocks; returns a whole utterance when the speaker stops."""

    def __init__(self, speech_prob, cfg=None, reset_model=None):
        self.cfg = cfg or VadConfig()
        self._prob, self._reset_model = speech_prob, reset_model
        self._end_blocks = math.ceil(self.cfg.end_silence_ms / BLOCK_MS)
        self._min_blocks = math.ceil(self.cfg.min_speech_ms / BLOCK_MS)
        self._max_blocks = int(self.cfg.max_utterance_s * 1000 / BLOCK_MS)
        self._pre = deque(maxlen=math.ceil(self.cfg.preroll_ms / BLOCK_MS))
        self._clear()

    def _clear(self):
        self._pre.clear()
        self._blocks, self._speech, self._silence = None, 0, 0

    def reset(self):
        self._clear()
        if self._reset_model:
            self._reset_model()

    def feed(self, block):
        p = self._prob(block)
        if self._blocks is None:
            self._pre.append(block)
            if p >= self.cfg.start_threshold:
                self._blocks, self._speech = list(self._pre), 1
            return None
        self._blocks.append(block)
        if p >= self.cfg.end_threshold:
            self._speech, self._silence = self._speech + 1, 0
        else:
            self._silence += 1
        if self._silence < self._end_blocks and len(self._blocks) < self._max_blocks:
            return None
        blocks, speech = self._blocks, self._speech
        self._clear()
        return np.concatenate(blocks) if speech >= self._min_blocks else None


def silero_segmenter(cfg=None):
    """Segmenter backed by the Silero VAD model (CPU)."""
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad()

    def speech_prob(block):
        with torch.no_grad():
            return model(torch.from_numpy(block), SAMPLE_RATE).item()

    return Segmenter(speech_prob, cfg, reset_model=model.reset_states)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_vad.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Check the real Silero model loads and scores a block**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); import numpy as np; from assistant.vad import silero_segmenter, BLOCK; s = silero_segmenter(); print(s.feed(np.zeros(BLOCK, dtype=np.float32)), round(s._prob(np.zeros(BLOCK, dtype=np.float32)), 3)); s.reset()"
```
Expected: `None` and a probability below `0.1` (silence), no exception.

- [ ] **Step 6: Commit**

```powershell
git add assistant/vad.py tests/test_vad.py
git commit -m "feat(assistant): VAD utterance segmenter" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Gemini client

**Files:**
- Create: `assistant/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `google.genai.Client().models.generate_content_stream(model=, contents=, config=)`; `contents` and `config` are passed as plain dicts, which the SDK accepts. Each streamed chunk has `.text` (`str` or `None`). API errors carry an integer `.code`.
- Produces: `assistant.llm.LLMError`, `LLMRateLimited(LLMError)`; `GeminiLLM(model, system_prompt, max_turns=10, max_output_tokens=300, thinking_budget=0, client=None)` with `stream_reply(user_text: str) -> Iterator[str]` and `history: list[tuple[str, str]]`.

- [ ] **Step 1: Write the failing tests in `tests/test_llm.py`**

```python
from types import SimpleNamespace

import pytest

from assistant.llm import GeminiLLM, LLMError, LLMRateLimited


class FakeClient:
    """Stands in for genai.Client: each call takes the next scripted reply, a list whose items are
    text deltas (None = a chunk without text) or an exception to raise at that point."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.models = self

    def generate_content_stream(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        for item in self.replies.pop(0):
            if isinstance(item, Exception):
                raise item
            yield SimpleNamespace(text=item)


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def make(client, **kw):
    return GeminiLLM("gemini-test", "Sistem.", client=client, **kw)


def test_stream_yields_deltas_and_records_both_turns():
    llm = make(FakeClient(["Salam", None, "! Necəsən?"]))
    assert list(llm.stream_reply("Salam")) == ["Salam", "! Necəsən?"]
    assert llm.history == [("user", "Salam"), ("model", "Salam! Necəsən?")]


def test_request_carries_model_system_prompt_history_and_thinking_budget():
    client = FakeClient(["Bir."], ["İki."])
    llm = make(client, max_output_tokens=50)
    list(llm.stream_reply("a"))
    list(llm.stream_reply("b"))
    call = client.calls[1]
    assert call["model"] == "gemini-test"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "a"}]},
                                {"role": "model", "parts": [{"text": "Bir."}]},
                                {"role": "user", "parts": [{"text": "b"}]}]
    assert call["config"] == {"system_instruction": "Sistem.", "max_output_tokens": 50,
                              "thinking_config": {"thinking_budget": 0}}


def test_thinking_budget_none_omits_the_setting():
    client = FakeClient(["Bir."])
    list(make(client, thinking_budget=None).stream_reply("a"))
    assert "thinking_config" not in client.calls[0]["config"]


def test_history_is_trimmed_to_max_turns():
    llm = make(FakeClient(["1."], ["2."], ["3."]), max_turns=2)
    for q in "abc":
        list(llm.stream_reply(q))
    assert llm.history == [("user", "b"), ("model", "2."), ("user", "c"), ("model", "3.")]


def test_failure_before_any_text_rolls_the_user_turn_back():
    llm = make(FakeClient([ApiError(500)], ["Bəli."]))
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert llm.history == []
    assert list(llm.stream_reply("b")) == ["Bəli."]


def test_http_429_is_rate_limited():
    llm = make(FakeClient([ApiError(429)]))
    with pytest.raises(LLMRateLimited):
        list(llm.stream_reply("a"))


def test_failure_mid_stream_keeps_what_was_said():
    llm = make(FakeClient(["Birinci cümlə. ", ApiError(503)]))
    got = []
    with pytest.raises(LLMError):
        for delta in llm.stream_reply("a"):
            got.append(delta)
    assert got == ["Birinci cümlə. "]
    assert llm.history == [("user", "a"), ("model", "Birinci cümlə. ")]


def test_reply_without_any_text_is_an_error_and_leaves_history_unchanged():
    llm = make(FakeClient([None, ""]))
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert llm.history == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_llm.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.llm'`.

- [ ] **Step 3: Create `assistant/llm.py`**

```python
"""Streaming Gemini chat client that keeps the conversation history."""


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    pass


class GeminiLLM:
    def __init__(self, model, system_prompt, max_turns=10, max_output_tokens=300, thinking_budget=0, client=None):
        if client is None:
            from google import genai
            client = genai.Client()  # reads GEMINI_API_KEY
        self.client, self.model, self.max_turns = client, model, max_turns
        self.config = {"system_instruction": system_prompt, "max_output_tokens": max_output_tokens}
        if thinking_budget is not None:  # null in the config: for models that reject the setting
            self.config["thinking_config"] = {"thinking_budget": thinking_budget}
        self.history = []  # [(role, text)], role is "user" or "model"

    def stream_reply(self, user_text):
        """Yield the reply's text deltas. Raises LLMError (LLMRateLimited on HTTP 429); a reply with
        no text at all is an LLMError too, and leaves the history as it was."""
        self.history.append(("user", user_text))
        said = []
        try:
            contents = [{"role": role, "parts": [{"text": text}]} for role, text in self.history]
            for chunk in self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=self.config):
                if chunk.text:  # None for chunks without text parts
                    said.append(chunk.text)
                    yield chunk.text
            if not said:
                raise LLMError("empty reply")
        except LLMError:
            raise
        except Exception as e:
            kind = LLMRateLimited if getattr(e, "code", None) == 429 else LLMError
            raise kind(str(e)) from e
        finally:
            if said:
                self.history.append(("model", "".join(said)))
                self.history = self.history[-2 * self.max_turns:]
            else:
                self.history.pop()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_llm.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```powershell
git add assistant/llm.py tests/test_llm.py
git commit -m "feat(assistant): streaming Gemini client with history" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Speech-to-text

**Files:**
- Create: `assistant/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Consumes: `faster_whisper.WhisperModel(model, device=, compute_type=)` and its `transcribe(audio, language=, beam_size=, vad_filter=, condition_on_previous_text=, initial_prompt=) -> (segments, info)`; each segment has `.text` and `.no_speech_prob`.
- Produces: `assistant.stt.clean_transcript(segments, initial_prompt=None, max_no_speech=0.6) -> str`; `WhisperSTT(model="large-v3-turbo", device="cuda", compute_type="float16", initial_prompt=None)` with `transcribe(audio: np.ndarray) -> str` (16 kHz mono float32 in, `""` when nothing was said).

- [ ] **Step 1: Write the failing tests in `tests/test_stt.py`**

```python
from types import SimpleNamespace

from assistant.stt import clean_transcript

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_stt.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.stt'`.

- [ ] **Step 3: Create `assistant/stt.py`**

```python
"""Azerbaijani speech-to-text with faster-whisper."""
import os
import re
from pathlib import Path

import numpy as np


def _key(text):
    return re.sub(r"[\W_]+", "", text).casefold()


def clean_transcript(segments, initial_prompt=None, max_no_speech=0.6):
    """Join Whisper segments into one line. "" when nothing was really said: segments Whisper marks as
    non-speech are dropped, and so is a result with no letters or one that just echoes the prompt
    (what Whisper tends to produce on noise)."""
    text = " ".join(s.text.strip() for s in segments if s.no_speech_prob <= max_no_speech).strip()
    if not _key(text) or (initial_prompt and _key(text) == _key(initial_prompt)):
        return ""
    return text


def _add_torch_dlls():
    """CTranslate2 needs the CUDA 12 cuBLAS/cuDNN DLLs on Windows; torch ships them in torch/lib."""
    import torch
    lib = Path(torch.__file__).parent / "lib"
    if os.name == "nt" and lib.is_dir():
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"


class WhisperSTT:
    def __init__(self, model="large-v3-turbo", device="cuda", compute_type="float16", initial_prompt=None):
        from faster_whisper import WhisperModel
        _add_torch_dlls()
        self.initial_prompt = initial_prompt
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
            self.transcribe(np.zeros(16000, dtype=np.float32))  # GPU libraries are first used here
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[stt] {device} unavailable ({type(e).__name__}: {e}); falling back to CPU int8")
            self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        """16 kHz mono float32 array -> Azerbaijani text ("" if no speech)."""
        segments, _ = self.model.transcribe(audio, language="az", beam_size=5, vad_filter=False,
                                            condition_on_previous_text=False,
                                            initial_prompt=self.initial_prompt)
        return clean_transcript(segments, self.initial_prompt)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_stt.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Check Whisper on the GPU with a real Azerbaijani clip**

The first run downloads the model (about 1.6 GB). Run:
```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -c "import sys, time; sys.path.insert(0, '.'); import librosa; from assistant.stt import WhisperSTT; s = WhisperSTT(); a, _ = librosa.load('data/wavs/000063.wav', sr=16000); t = time.time(); print(s.transcribe(a)); print(round(time.time() - t, 2), 's on', s.model.model.device)"
.\.venv\Scripts\python.exe -c "import csv; print(next(r['text'] for r in csv.DictReader(open('metadata.csv', encoding='utf-8', newline='')) if r['filename'].startswith('000063')))"
```
Expected: the first command prints an Azerbaijani sentence close to the reference text printed by the second, then a time under 1 s `on cuda`.

If it prints `falling back to CPU int8`, or the process dies with a message about a missing `cudnn` or `cublas` DLL (CTranslate2 can abort without a Python exception), GPU STT is unavailable: note the exact message, and in Task 8 set `stt.device: cpu` and `stt.compute_type: int8` in `configs/assistant.yml`. Report this to the user at the end; do not try to install CUDA libraries.

- [ ] **Step 6: Commit**

```powershell
git add assistant/stt.py tests/test_stt.py
git commit -m "feat(assistant): faster-whisper Azerbaijani STT" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Audio input and output

**Files:**
- Create: `assistant/audio_io.py`
- Test: `tests/test_audio_io.py`

**Interfaces:**
- Consumes: `assistant.vad.BLOCK`, `assistant.vad.SAMPLE_RATE` (Task 3); `sounddevice.InputStream`, `sounddevice.OutputStream`, `sounddevice.query_devices`.
- Produces: `assistant.audio_io.OUT_RATE = 24000`; `MicStream(device=None)` with `start()`, `stop()`, `close()`, `read(timeout=None) -> np.ndarray | None`, `drain()`; `Player(device=None)` with `play(wav: np.ndarray)` and `close()`; `list_devices()`.

- [ ] **Step 1: Write the failing tests in `tests/test_audio_io.py`**

```python
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from assistant.audio_io import MicStream, Player


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs, self.written, self.events = kwargs, [], []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")

    def write(self, data):
        self.written.append(data)


@pytest.fixture(autouse=True)
def fake_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeStream, OutputStream=FakeStream))


def test_mic_opens_16k_mono_512_sample_blocks():
    mic = MicStream(device=3)
    kw = mic._stream.kwargs
    assert (kw["samplerate"], kw["blocksize"], kw["channels"], kw["dtype"], kw["device"]) == \
           (16000, 512, 1, "float32", 3)


def test_mic_read_returns_callback_blocks_in_order_as_mono_copies():
    mic = MicStream()
    callback = mic._stream.kwargs["callback"]
    buf = np.ones((512, 1), dtype=np.float32)
    callback(buf, 512, None, None)
    buf[:] = 2  # PortAudio reuses the buffer: the queued block must be a copy
    callback(buf, 512, None, None)
    first, second = mic.read(timeout=0), mic.read(timeout=0)
    assert first.shape == (512,) and first[0] == 1 and second[0] == 2


def test_mic_read_times_out_with_none_and_drain_empties_the_queue():
    mic = MicStream()
    assert mic.read(timeout=0.01) is None
    for _ in range(3):
        mic._stream.kwargs["callback"](np.zeros((512, 1), dtype=np.float32), 512, None, None)
    mic.drain()
    assert mic.read(timeout=0.01) is None


def test_player_opens_24k_mono_and_writes_clipped_float32_columns():
    player = Player(device=5)
    stream = player._stream
    assert (stream.kwargs["samplerate"], stream.kwargs["channels"], stream.kwargs["device"]) == (24000, 1, 5)
    assert stream.events == ["start"]
    player.play(np.array([0.5, 2.0, -3.0], dtype=np.float64))
    out = stream.written[0]
    assert out.dtype == np.float32 and out.shape == (3, 1)
    assert out[:, 0].tolist() == [0.5, 1.0, -1.0]
    player.close()
    assert stream.events == ["start", "stop", "close"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_audio_io.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.audio_io'`.

- [ ] **Step 3: Create `assistant/audio_io.py`**

```python
"""Microphone capture and speaker playback (sounddevice / PortAudio)."""
import queue

import numpy as np

from assistant.vad import BLOCK, SAMPLE_RATE

OUT_RATE = 24000  # StyleTTS2's output sample rate


class MicStream:
    """16 kHz mono capture; the audio callback only queues blocks, read() hands them out."""

    def __init__(self, device=None):
        import sounddevice as sd
        self._q = queue.Queue()
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1, dtype="float32",
                                      device=device, callback=self._on_audio)

    def _on_audio(self, indata, frames, time, status):
        self._q.put(indata[:, 0].copy())

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()

    def close(self):
        self._stream.close()

    def read(self, timeout=None):
        """Next 512-sample float32 block, or None if none arrived within `timeout` seconds."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        """Discard everything captured so far."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


class Player:
    def __init__(self, device=None):
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="float32", device=device)
        self._stream.start()

    def play(self, wav):
        """Blocks until the 24 kHz mono buffer has been handed to the sound card."""
        self._stream.write(np.clip(wav, -1.0, 1.0).astype(np.float32).reshape(-1, 1))

    def close(self):
        self._stream.stop()
        self._stream.close()


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_audio_io.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Check the real devices open at the required sample rates**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import sys, time; sys.path.insert(0, '.'); from assistant.audio_io import MicStream, Player, list_devices; list_devices(); m = MicStream(); m.start(); time.sleep(0.5); b = m.read(timeout=1); m.stop(); m.close(); print('mic block', None if b is None else b.shape); p = Player(); p.close(); print('player ok')"
```
Expected: the device table, then `mic block (512,)` and `player ok`. If a stream fails to open with an invalid sample rate error, the default device does not resample: note it for the README troubleshooting text in Task 9 (the user picks another device with `--input-device` / `--output-device`). If the machine has no microphone, `mic block None` or an open error is acceptable here; report it.

- [ ] **Step 6: Commit**

```powershell
git add assistant/audio_io.py tests/test_audio_io.py
git commit -m "feat(assistant): microphone capture and speaker playback" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: TTS service

**Files:**
- Create: `assistant/tts.py`
- Test: `tests/test_tts.py`

**Interfaces:**
- Consumes: `assistant.ROOT` (Task 1); from `scripts/synthesize.py` (existing): `load_inference_defaults(path) -> dict` (keys `alpha`, `beta`, `steps`, `embedding_scale`, `speed`, `ref`, and `checkpoint` when tuned), `default_config(checkpoint) -> Path`, `Synth(config_path, checkpoint, device)` with `.g2p`, `.compute_styles(paths)`, and `__call__(ipa, ref_s, alpha, beta, steps, embedding_scale, speed) -> np.ndarray` at 24 kHz; `az_text.phonemize_texts(texts, g2p=) -> list[str]`.
- Produces: `assistant.tts.resolve_checkpoint(checkpoint, defaults) -> Path`; `TTSService(checkpoint=None, inference_path=ROOT / "configs/inference.json", device="cuda")` with `synthesize(text: str) -> np.ndarray` (24 kHz mono float32).

- [ ] **Step 1: Write the failing tests in `tests/test_tts.py`**

```python
import pytest

from assistant import ROOT
from assistant.tts import resolve_checkpoint


def test_explicit_checkpoint_wins_and_is_resolved_against_the_repo_root():
    assert resolve_checkpoint("configs/az_lexicon.tsv", {"checkpoint": "other.pth"}) == \
           (ROOT / "configs/az_lexicon.tsv").resolve()


def test_falls_back_to_the_checkpoint_inference_json_was_tuned_on():
    assert resolve_checkpoint(None, {"checkpoint": "configs\\az_lexicon.tsv"}) == \
           (ROOT / "configs/az_lexicon.tsv").resolve()


def test_absolute_path_is_kept(tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    assert resolve_checkpoint(str(ckpt), {}) == ckpt.resolve()


def test_no_checkpoint_anywhere_is_a_clear_error():
    with pytest.raises(ValueError, match="tts.checkpoint"):
        resolve_checkpoint(None, {})


def test_missing_file_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="nope.pth"):
        resolve_checkpoint("runs/nope.pth", {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_tts.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.tts'`.

- [ ] **Step 3: Create `assistant/tts.py`**

```python
"""The project's fine-tuned StyleTTS2 voice, loaded once and reused for every chunk."""
import threading

import numpy as np

from assistant import ROOT
from az_text import phonemize_texts


def resolve_checkpoint(checkpoint, defaults):
    """`checkpoint` (relative to the repo root, or absolute), else the one configs/inference.json was tuned on."""
    checkpoint = checkpoint or defaults.get("checkpoint")
    if not checkpoint:
        raise ValueError("no TTS checkpoint: set tts.checkpoint in configs/assistant.yml")
    path = (ROOT / checkpoint).resolve()
    if not path.exists():
        raise FileNotFoundError(f"TTS checkpoint not found: {path}")
    return path


class TTSService:
    def __init__(self, checkpoint=None, inference_path=ROOT / "configs/inference.json", device="cuda"):
        import synthesize as S  # imports torch and StyleTTS2
        d = S.load_inference_defaults(inference_path)
        checkpoint = resolve_checkpoint(checkpoint, d)
        self._synth = S.Synth(S.default_config(checkpoint), checkpoint, device)
        if d["ref"]:
            refs = [ROOT / r for r in d["ref"]]  # inference.json paths are relative to the repo root
        else:
            first = (ROOT / "data/val_list.txt").read_text(encoding="utf-8").splitlines()[0].split("|")[0]
            refs = [ROOT / "data/wavs" / first]
        self._ref = self._synth.compute_styles(refs)
        self._args = (d["alpha"], d["beta"], d["steps"], d["embedding_scale"], d["speed"])
        self._lock = threading.Lock()
        self.synthesize("salam, mən hazıram.")  # warm-up: the first call is about 1 s slower

    def synthesize(self, text):
        """Speakable text (see text_norm.to_spoken) -> 24 kHz mono float32 audio."""
        ipa = phonemize_texts([text], g2p=self._synth.g2p)[0]
        with self._lock:
            wav = self._synth(ipa, self._ref, *self._args)
        return np.asarray(wav, dtype=np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_tts.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Hear the real voice read normalized text**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import sys, time; sys.path.insert(0, '.'); from assistant.tts import TTSService; from assistant.audio_io import Player; from assistant.text_norm import to_spoken; t = TTSService(); s = time.time(); w = t.synthesize(to_spoken('Saat 18:30, 150 manat.')); print(round(time.time() - s, 2), 's for', round(len(w) / 24000, 2), 's of audio', w.dtype); Player().play(w); time.sleep(1)"
```
Expected: after about 20 s of loading, a line like `0.4 s for 3.1 s of audio float32`, and the voice says "saat on səkkiz otuz, yüz əlli manat" through the speakers. Synthesis time above 1 s means the warm-up did not run or the GPU is busy; investigate before continuing.

- [ ] **Step 6: Commit**

```powershell
git add assistant/tts.py tests/test_tts.py
git commit -m "feat(assistant): load-once StyleTTS2 service" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Pipeline and configuration

**Files:**
- Create: `assistant/pipeline.py`, `configs/assistant.yml`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `assistant.audio_io.OUT_RATE` (Task 6); `assistant.chunker.SentenceChunker` (Task 2); `assistant.llm.LLMError` (Task 4); `assistant.text_norm.to_spoken` (Task 1); `assistant.vad.VadConfig` (Task 3, in the test only). At run time the pipeline calls, on the objects passed in: `llm.stream_reply(text)`, `tts.synthesize(text)`, `player.play(wav)`, `mic.start()/stop()/read(timeout=)/drain()`, `segmenter.feed(block)/reset()`, `stt.transcribe(audio)`.
- Produces: `assistant.pipeline.FALLBACK_TEXT: str`; `SECTIONS`; `load_config(path) -> dict` (raises `ValueError` naming missing sections); `Pipeline(llm, tts, player, mic=None, segmenter=None, stt=None, chunker=None, gap_ms=150, tail_ms=300, fallback_wav=None, log=print, sleep=time.sleep)` with `run()`, `run_text(lines)`, `step() -> str | None`, `reply(user_text, t0=None, marks=None) -> str | None`, and `state` in `"LISTENING" | "THINKING" | "SPEAKING"`.

- [ ] **Step 1: Create `configs/assistant.yml`**

If Task 5 found that GPU STT is unavailable, write `device: cpu` and `compute_type: int8` in the `stt` section instead of the values below.

```yaml
# Voice assistant settings (scripts/talk.py). The Gemini key is read from the GEMINI_API_KEY environment variable.
tts:
  checkpoint: null            # null = the checkpoint named in configs/inference.json
stt:
  model: large-v3-turbo
  device: cuda                # cpu also works (compute_type int8), about 1-2 s slower per turn
  compute_type: float16
  initial_prompt: "Salam, necəsiniz? Bu gün sizə necə kömək edə bilərəm?"
llm:
  model: gemini-2.5-flash
  max_turns: 10               # exchanges kept as conversation history
  max_output_tokens: 300
  thinking_budget: 0          # 0 = no thinking (fastest); null = do not send the setting
  system_prompt: |
    Sən Azərbaycan dilində danışan səsli köməkçisən. Yalnız Azərbaycan dilində,
    bir-üç qısa cümlə ilə, danışıq üslubunda cavab ver. Siyahı, markdown, emoji
    və ixtisarlardan istifadə etmə. Rəqəmləri sözlə yaz.
vad:
  start_threshold: 0.5
  end_threshold: 0.35
  end_silence_ms: 600
  min_speech_ms: 250
  preroll_ms: 300
  max_utterance_s: 30
chunker:
  min_chars: 25
  max_chars: 180
playback:
  gap_ms: 150
  tail_ms: 300
```

- [ ] **Step 2: Write the failing tests in `tests/test_pipeline.py`**

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -q`
Expected: `ModuleNotFoundError: No module named 'assistant.pipeline'`.

- [ ] **Step 4: Create `assistant/pipeline.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -q`
Expected: `14 passed`, in well under 5 s. A test that hangs means a worker thread never received its end marker; fix the pipeline, not the test.

- [ ] **Step 6: Commit**

```powershell
git add assistant/pipeline.py configs/assistant.yml tests/test_pipeline.py
git commit -m "feat(assistant): half-duplex turn pipeline and config" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Command-line entry point, README, end-to-end check

**Files:**
- Create: `scripts/talk.py`
- Modify: `README.md` (add a row to the Layout table and a new section before `## Troubleshooting`)
- Test: `tests/test_talk.py`

**Interfaces:**
- Consumes: `assistant.audio_io.MicStream`, `Player`, `list_devices` (Task 6); `assistant.llm.GeminiLLM` (Task 4); `assistant.pipeline.FALLBACK_TEXT`, `Pipeline`, `load_config` (Task 8); `assistant.text_norm.to_spoken` (Task 1); `assistant.tts.TTSService` (Task 7); `assistant.stt.WhisperSTT` (Task 5); `assistant.vad.VadConfig`, `silero_segmenter` (Task 3).
- Produces: `scripts/talk.py` with `main(argv=None)` and `stdin_lines()`.

- [ ] **Step 1: Write the failing tests in `tests/test_talk.py`**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_talk.py -q`
Expected: `ModuleNotFoundError: No module named 'talk'`.

- [ ] **Step 3: Create `scripts/talk.py`**

```python
"""Talk to the Azerbaijani voice assistant: microphone -> Whisper -> Gemini -> StyleTTS2 -> speakers."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.audio_io import MicStream, Player, list_devices  # noqa: E402
from assistant.llm import GeminiLLM  # noqa: E402
from assistant.pipeline import FALLBACK_TEXT, Pipeline, load_config  # noqa: E402
from assistant.text_norm import to_spoken  # noqa: E402

NO_KEY = ("GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey, then run:\n"
          "    $env:GEMINI_API_KEY = '<your key>'")


def stdin_lines():
    while True:
        try:
            yield input("> ")
        except EOFError:
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/assistant.yml"))
    ap.add_argument("--text", action="store_true", help="type instead of speaking (no microphone or STT)")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
    a = ap.parse_args(argv)

    if a.list_devices:
        list_devices()
        return
    cfg = load_config(a.config)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(NO_KEY)

    from assistant.tts import TTSService  # torch and StyleTTS2: only after the cheap checks passed
    print("[talk] loading the TTS model ...")
    tts = TTSService(cfg["tts"]["checkpoint"])
    fallback = tts.synthesize(to_spoken(FALLBACK_TEXT))
    llm = GeminiLLM(**cfg["llm"])
    mic = segmenter = stt = None
    if not a.text:
        from assistant.stt import WhisperSTT
        from assistant.vad import VadConfig, silero_segmenter
        print("[talk] loading the STT model ...")
        stt = WhisperSTT(**cfg["stt"])
        segmenter = silero_segmenter(VadConfig(**cfg["vad"]))
        mic = MicStream(a.input_device)
    player = Player(a.output_device)
    pipeline = Pipeline(llm, tts, player, mic=mic, segmenter=segmenter, stt=stt, chunker=cfg["chunker"],
                        gap_ms=cfg["playback"]["gap_ms"], tail_ms=cfg["playback"]["tail_ms"],
                        fallback_wav=fallback)
    print("[talk] ready: " + ("type a message" if a.text else "speak") + " (Ctrl+C to quit)")
    try:
        if a.text:
            pipeline.run_text(stdin_lines())
        else:
            pipeline.run()
    except KeyboardInterrupt:
        print("\n[talk] bye")
    finally:
        player.close()
        if mic:
            mic.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Azerbaijani letters on a cp1252 Windows console
    main()
```

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_talk.py -q`
Expected: `4 passed`.

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: `246 passed` (114 existing + 132 new).

- [ ] **Step 5: Document it in `README.md`**

Add this row to the Layout table, after the `scripts/synthesize.py` row:

```markdown
| `assistant/`, `scripts/talk.py`, `configs/assistant.yml` | Voice assistant: microphone → Whisper → Gemini → this model's voice (see *Voice assistant*). |
```

Insert this section immediately before the `## Troubleshooting` heading:

````markdown
## Voice assistant

Speak Azerbaijani into the microphone, get a spoken answer in this model's voice:
microphone → Silero VAD → faster-whisper (`large-v3-turbo`) → Gemini → StyleTTS2 → speakers.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-assistant.txt   # once
$env:GEMINI_API_KEY = '<your key>'                                        # https://aistudio.google.com/apikey
.\.venv\Scripts\python.exe scripts\talk.py            # speak; Ctrl+C to quit
.\.venv\Scripts\python.exe scripts\talk.py --text     # type instead of speaking (no microphone or STT)
.\.venv\Scripts\python.exe scripts\talk.py --list-devices
```

Startup takes about 30 s (TTS and Whisper models; the first run also downloads Whisper,
about 1.6 GB). The voice, checkpoint and style settings are the ones in
`configs/inference.json`; everything else is in `configs/assistant.yml`.

How it behaves:

- **Half-duplex.** The microphone is ignored while the assistant speaks; you cannot
  interrupt a reply. Use speakers or headphones.
- **Sentence streaming.** The reply is cut into sentences as Gemini streams it, and each
  one is synthesized (about 0.4 s) and played while the next is still arriving. Each turn
  prints a `[latency]` line: `stt`, `llm_first_chunk`, `tts_first`, `first_audio`, in ms
  from the moment the end of speech was detected (0.6 s after you stop talking).
- **Text normalization.** `assistant/text_norm.py` rewrites what the G2P cannot read:
  `150` → "yüz əlli", `18:30` → "on səkkiz otuz", `50%` → "əlli faiz", `5-ci` → "beşinci",
  markdown and emoji are dropped. A mispronounced word is fixed in `configs/az_lexicon.tsv`.

Settings worth knowing in `configs/assistant.yml`:

| Key | Effect |
|---|---|
| `llm.model` | Gemini model name; change it if the free-plan lineup changes. |
| `llm.system_prompt` | The assistant's persona and answer style. |
| `llm.thinking_budget` | `0` disables thinking (fastest); `null` for models that reject the setting. |
| `vad.end_silence_ms` | Silence that ends your turn. Lower is snappier but cuts you off at pauses. |
| `vad.start_threshold` | Raise it if background noise starts turns by itself. |
| `stt.device` / `stt.compute_type` | `cpu` / `int8` if GPU Whisper does not work (about 1–2 s slower). |

Limitations: Whisper's Azerbaijani is weaker than its major languages and sometimes drifts
into Turkish, so recognition is the usual cause of an odd answer (check the `Siz:` line).
The Gemini free plan has per-minute and per-day request limits (the assistant then says it
cannot answer right now), and Google may use free-plan prompts to improve its models, so
do not say anything sensitive. Do not run the assistant during a training run: both need
the GPU.

If a sound device refuses to open, pick another one from `--list-devices` with
`--input-device N` / `--output-device N`.
````

- [ ] **Step 6: End-to-end check in text mode**

This step needs the user's Gemini key. Check with `.\.venv\Scripts\python.exe -c "import os; print(bool(os.environ.get('GEMINI_API_KEY')))"`. If it prints `False`, do not ask for the key in chat and do not write it anywhere: finish Step 8, then tell the user that Steps 6 and 7 are theirs to run after setting `$env:GEMINI_API_KEY`.

Run `.\.venv\Scripts\python.exe scripts\talk.py --text`, type a greeting and a question whose answer contains a number (for example the price of something or what year it is), then press Ctrl+C.
Expected: each reply is printed after `Köməkçi:`, spoken in the fine-tuned voice with numbers read as words, followed by a `[latency]` line. No traceback. If Gemini rejects `thinking_budget` for the configured model, set `llm.thinking_budget: null` in `configs/assistant.yml`; if the model name is not available on the free plan, change `llm.model` to one that is.

- [ ] **Step 7: End-to-end check with the microphone**

Run `.\.venv\Scripts\python.exe scripts\talk.py` and hold a conversation of at least three turns in Azerbaijani, including staying silent for ten seconds and speaking while the assistant is talking.
Expected: `Siz:` shows what was said; replies are spoken; silence triggers nothing; speech during a reply is ignored and does not start a new turn afterwards. Record the `[latency]` lines: `first_audio` should be around 1.5–2 s (the 0.6 s end-of-speech wait is not included in it). Report the measured values to the user.

- [ ] **Step 8: Commit**

```powershell
git add scripts/talk.py tests/test_talk.py README.md
git commit -m "feat(assistant): talk.py entry point and README section" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
