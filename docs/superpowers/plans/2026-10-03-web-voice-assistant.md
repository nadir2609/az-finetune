# Web Voice Assistant (Demo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A web page that lets up to 5 people talk hands-free with the Azerbaijani assistant, served from this PC's GPU, with the reply starting less than 3 s after the visitor stops speaking.

**Architecture:** One FastAPI process loads Whisper and the StyleTTS2 voice once. Each WebSocket connection is a session that runs the existing `Pipeline` in its own thread, with a socket-backed microphone and speaker in place of the sound card. A static page captures the microphone at 16 kHz, streams it up, and plays the 24 kHz reply audio it receives.

**Tech Stack:** Python 3.10 venv, PyTorch 2.11.0+cu128 (installed), FastAPI + uvicorn (new), existing `assistant/` package, plain JavaScript with the Web Audio API and an AudioWorklet, pytest with FastAPI's `TestClient`.

**Spec:** `docs/superpowers/specs/2026-10-03-web-voice-assistant-design.md`

## Global Constraints

- Branch: `feat/web-assistant`. Commit after each task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push unless the user asks.
- Stage only the exact paths a task names; never `git add -A` or `git add .`.
- Python: always `.venv\Scripts\python.exe`. After installing dependencies, torch must still be `2.11.0+cu128`.
- New dependencies go only in `requirements-assistant.txt`.
- The server binds to `127.0.0.1` only. `GEMINI_API_KEY` and `ASSISTANT_ACCESS_CODE` are read from the environment or `.env`; never write them to a file, a log line, a page or a commit.
- Unit tests must not need a GPU, a microphone, speakers or the network, and must not call Gemini (the free plan allows 20 requests a day).
- Audio formats: up 16 kHz mono 16-bit little-endian, re-blocked to 512-sample float32 blocks; down 24 kHz mono 16-bit little-endian.
- Close codes: 4401 wrong access code, 4429 demo full, 4408 idle timeout.
- Text shown to visitors on the page is in Azerbaijani.
- All files written by our code use UTF-8 explicitly. Do not put non-ASCII characters in shell command lines.
- Half-duplex only. Out of scope: accounts, saved history, barge-in, streaming STT, cloud deployment, native apps, WebRTC.
- Do not run the server while a training run is using the GPU.

## Review Focus

Inputs the spec implies but does not spell out; each has a test in the task that owns the code.

1. **Audio frames of an unexpected size** (odd byte counts, several blocks in one message): they are re-blocked without losing sample alignment. Task 2.
2. **Reply samples outside [-1, 1] or NaN** from the TTS: they are clipped / zeroed instead of wrapping around into loud noise. Task 2.
3. **A text message from the client** where audio is expected: ignored, the session keeps working. Task 5.
4. **An access code with non-ASCII letters** (an Azerbaijani word): compared correctly instead of raising. Task 5.
5. **A visitor who connects and never speaks**: closed with 4408 and the slot is freed for someone else. Task 5.

## File Structure

| Path | Responsibility |
|---|---|
| `assistant/web_io.py` (new) | `SocketMic`, `SocketPlayer`, `Disconnected`, `IdleTimeout`: the sound card replaced by a connection. No FastAPI imports. |
| `assistant/server.py` (new) | `create_app(...)`: routes, access check, session limit, the bridge between the event loop and a session's thread. |
| `assistant/web/index.html`, `app.js`, `mic-worklet.js` (new) | The page. |
| `assistant/pipeline.py` (modify) | `on_event` callback; stop the turn when the player fails; `server` config section. |
| `assistant/stt.py` (modify) | Lock around transcription. |
| `scripts/serve.py` (new) | Entry point: checks, model loading, uvicorn. |
| `scripts/measure_latency.py` (new) | Streams a WAV to a running server and reports the reply delay. |
| `scripts/setup_inference.py` (modify) | `--gpu`. |
| `configs/assistant.yml`, `requirements-assistant.txt`, `README.md` (modify) | Settings, dependencies, instructions. |

---

### Task 1: Dependencies and the `server` config section

**Files:**
- Modify: `requirements-assistant.txt`
- Modify: `configs/assistant.yml` (append)
- Modify: `assistant/pipeline.py` (`SECTIONS`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `load_config(path)["server"]` with keys `host`, `port`, `max_sessions`, `idle_timeout_s`.

- [ ] **Step 1: Update the failing tests**

In `tests/test_pipeline.py`, in `test_load_config_names_missing_sections` change the expected message, and in `test_shipped_config_matches_the_constructors` add one assertion:

```python
    with pytest.raises(ValueError, match="llm, vad, chunker, playback, server"):
        load_config(path)
```

```python
    assert set(cfg["server"]) == {"host", "port", "max_sessions", "idle_timeout_s"}
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q -k "load_config or shipped_config"`
Expected: 2 failed (the message lacks `server`; `KeyError: 'server'`).

- [ ] **Step 3: Implement**

`assistant/pipeline.py`:

```python
SECTIONS = ("tts", "stt", "llm", "vad", "chunker", "playback", "server")
```

Append to `configs/assistant.yml`:

```yaml
server:                       # web demo (scripts/serve.py)
  host: 127.0.0.1             # the tunnel connects locally; nothing is exposed on the LAN
  port: 8000
  max_sessions: 5             # visitors connected at the same time
  idle_timeout_s: 300         # a connection that sends no audio for this long is closed
```

Append to `requirements-assistant.txt`:

```
fastapi
uvicorn[standard]
```

- [ ] **Step 4: Install and verify**

Run: `.venv\Scripts\python.exe -m pip install -r requirements-assistant.txt`
Run: `.venv\Scripts\python.exe -c "import torch, fastapi, uvicorn; print(torch.__version__)"`
Expected: `2.11.0+cu128`
Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: all pass (278 before this plan).

- [ ] **Step 5: Commit**

```bash
git add requirements-assistant.txt configs/assistant.yml assistant/pipeline.py tests/test_pipeline.py
git commit -m "feat(assistant): server config section and web dependencies"
```

---

### Task 2: `assistant/web_io.py`

**Files:**
- Create: `assistant/web_io.py`
- Test: `tests/test_web_io.py`

**Interfaces:**
- Consumes: `assistant.vad.BLOCK` (512).
- Produces:
  - `class Disconnected(Exception)`, `class IdleTimeout(Disconnected)`
  - `SocketMic(idle_timeout_s=300, clock=time.monotonic)` with `push(data: bytes)`, `read(timeout=None) -> np.ndarray | None`, `drain()`, `start()`, `stop()`, `close()`
  - `SocketPlayer(send: Callable[[bytes], None])` with `play(wav: np.ndarray)`

- [ ] **Step 1: Write the failing tests**

`tests/test_web_io.py`:

```python
import numpy as np
import pytest

from assistant.web_io import Disconnected, IdleTimeout, SocketMic, SocketPlayer


def pcm(samples):
    return np.asarray(samples, dtype="<i2").tobytes()


def test_one_block_of_16_bit_audio_becomes_512_float_samples():
    mic = SocketMic()
    mic.push(pcm([16384] * 512))
    block = mic.read(timeout=0)
    assert block.dtype == np.float32 and block.shape == (512,)
    assert block[0] == pytest.approx(0.5)
    assert mic.read(timeout=0) is None


def test_frames_of_any_size_are_reblocked_without_losing_alignment():
    mic = SocketMic()
    data = pcm(range(1024))  # two blocks, sent as odd-sized pieces
    for piece in (data[:1], data[1:700], data[700:1500], data[1500:]):
        mic.push(piece)
    first, second = mic.read(timeout=0), mic.read(timeout=0)
    assert first[1] == pytest.approx(1 / 32768) and second[0] == pytest.approx(512 / 32768)
    assert mic.read(timeout=0) is None


def test_several_blocks_in_one_message():
    mic = SocketMic()
    mic.push(pcm([0] * 512 * 3))
    assert [mic.read(timeout=0) is not None for _ in range(4)] == [True, True, True, False]


def test_drain_discards_queued_blocks():
    mic = SocketMic()
    mic.push(pcm([0] * 1024))
    mic.drain()
    assert mic.read(timeout=0) is None


def test_read_after_close_raises_disconnected():
    mic = SocketMic()
    mic.close()
    with pytest.raises(Disconnected):
        mic.read(timeout=0)


def test_no_audio_for_longer_than_the_idle_timeout_raises_idle_timeout():
    now = [0.0]
    mic = SocketMic(idle_timeout_s=300, clock=lambda: now[0])
    now[0] = 299.0
    assert mic.read(timeout=0) is None
    mic.push(pcm([0] * 512))  # audio restarts the idle clock
    mic.drain()
    now[0] = 598.0
    assert mic.read(timeout=0) is None
    now[0] = 600.0
    with pytest.raises(IdleTimeout):
        mic.read(timeout=0)


def test_start_and_stop_exist_for_the_pipeline():
    mic = SocketMic()
    mic.start()
    mic.stop()


def test_player_sends_16_bit_little_endian_audio():
    sent = []
    SocketPlayer(sent.append).play(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert np.frombuffer(sent[0], dtype="<i2").tolist() == [0, 16383, -16383]


def test_player_clips_out_of_range_samples_and_zeroes_nan():
    sent = []
    SocketPlayer(sent.append).play(np.array([2.0, -3.0, np.nan], dtype=np.float32))
    assert np.frombuffer(sent[0], dtype="<i2").tolist() == [32767, -32767, 0]


def test_player_lets_a_send_failure_through():
    def send(data):
        raise Disconnected("closed")

    with pytest.raises(Disconnected):
        SocketPlayer(send).play(np.zeros(4, dtype=np.float32))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_io.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'assistant.web_io'`.

- [ ] **Step 3: Implement**

`assistant/web_io.py`:

```python
"""Socket-backed stand-ins for MicStream and Player: a browser connection instead of the sound card."""
import queue
import time

import numpy as np

from assistant.vad import BLOCK

_BLOCK_BYTES = 2 * BLOCK  # 16-bit samples


class Disconnected(Exception):
    """The visitor is gone."""


class IdleTimeout(Disconnected):
    """The visitor sent no audio for too long."""


class SocketMic:
    """push() what the browser sends (16 kHz mono 16-bit little-endian, any message size);
    read() hands out 512-sample float32 blocks like MicStream."""

    def __init__(self, idle_timeout_s=300, clock=time.monotonic):
        self._q, self._buf = queue.Queue(), b""
        self._idle, self._clock = idle_timeout_s, clock
        self._last, self._closed = clock(), False

    def push(self, data):
        self._last = self._clock()
        self._buf += data
        while len(self._buf) >= _BLOCK_BYTES:
            block, self._buf = self._buf[:_BLOCK_BYTES], self._buf[_BLOCK_BYTES:]
            self._q.put(np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self._closed = True

    def read(self, timeout=None):
        """Next block, or None if none arrived within `timeout` seconds."""
        if self._closed:
            raise Disconnected("connection closed")
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            if self._clock() - self._last > self._idle:
                raise IdleTimeout(f"no audio for {self._idle} s") from None
            return None

    def drain(self):
        """Discard the blocks queued so far (a partial block is kept, so samples stay aligned)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


class SocketPlayer:
    """play() sends 24 kHz mono 16-bit little-endian audio through `send`, which raises Disconnected
    once the connection is closed. It does not wait for the audio to be heard."""

    def __init__(self, send):
        self._send = send

    def play(self, wav):
        samples = np.clip(np.nan_to_num(np.asarray(wav, dtype=np.float32)), -1.0, 1.0)
        self._send((samples * 32767).astype("<i2").tobytes())
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_io.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add assistant/web_io.py tests/test_web_io.py
git commit -m "feat(assistant): socket-backed microphone and speaker"
```

---

### Task 3: Pipeline events, and stop the turn when the player fails

**Files:**
- Modify: `assistant/pipeline.py` (`Pipeline.__init__`, `reply`, `_play`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `Pipeline(..., on_event=None)`; `on_event(kind: str, data)` is called in the pipeline's thread with `("user_text", str)`, then at the end of a reply `("reply_text", str)`, `("latency", dict[str, int])`, `("reply_end", None)`. An exception raised by `on_event` leaves `reply()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_events_report_the_turn_in_order():
    p, _, _, _, _ = make(FakeLLM("Bəli."))
    events = []
    p.on_event = lambda kind, data: events.append((kind, data))
    p.step()
    assert [kind for kind, _ in events] == ["user_text", "reply_text", "latency", "reply_end"]
    assert events[0][1] == "Salam" and events[1][1] == "Bəli." and events[3][1] is None
    assert set(events[2][1]) == {"stt", "llm_first_chunk", "tts_first", "first_audio"}


def test_a_failing_player_stops_the_rest_of_the_reply():
    import threading
    p, _, _, logs, _ = make(FakeLLM("Bəli."), player_fails=True)
    stop = threading.Event()
    assert p._play(np.ones(4, dtype=np.float32), stop) is False
    assert stop.is_set() and any("[audio] OSError" in line for line in logs)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q -k "events_report or failing_player"`
Expected: 2 failed (no events recorded; `stop.is_set()` is False).

- [ ] **Step 3: Implement**

In `Pipeline.__init__`, add the parameter after `clock` and store it:

```python
                 gap_ms=150, tail_ms=300, fallback_wav=None, log=print, sleep=time.sleep, rtf=None,
                 clock=time.perf_counter, on_event=None):
        self.rtf, self.clock, self.on_event = rtf, clock, on_event
```

Extend the class docstring with one sentence:

```python
    `on_event(kind, data)` is told about each turn: "user_text", "reply_text", "latency", "reply_end"."""
```

Add the helper below `run_text`:

```python
    def _emit(self, kind, data=None):
        if self.on_event:
            self.on_event(kind, data)
```

In `reply`, after `self.log(f"Siz: {user_text}")`:

```python
        self._emit("user_text", user_text)
```

and replace the end of `reply` (from `text = "".join(said).strip()`) with:

```python
        text = "".join(said).strip()
        timings = {k: marks[k] for k in _TIMINGS if k in marks}
        self.log(f"Köməkçi: {text}")
        self.log("[latency] " + " ".join(f"{k}={v}ms" for k, v in timings.items()))
        self._emit("reply_text", text)
        self._emit("latency", timings)
        self._emit("reply_end")
        return text
```

In `_play`, the `except` branch becomes:

```python
        except Exception as e:
            stop.set()  # nobody can hear the rest of this reply: do not synthesize it
            self.log(f"[audio] {type(e).__name__}: {e}")
            return False
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_talk.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/pipeline.py tests/test_pipeline.py
git commit -m "feat(assistant): turn events; stop synthesizing when playback fails"
```

---

### Task 4: One transcription at a time

**Files:**
- Modify: `assistant/stt.py` (`WhisperSTT`)
- Test: `tests/test_stt.py`

**Interfaces:**
- Produces: `WhisperSTT.transcribe(audio)` is safe to call from several threads; calls run one after another.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stt.py` (add `import sys`, `import threading`, `import time`, `import numpy as np` at the top, and import `WhisperSTT` next to `clean_transcript`):

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stt.py -q -k concurrent`
Expected: FAIL, `assert ... and 4 == 1` (or another peak above 1).

- [ ] **Step 3: Implement**

In `assistant/stt.py` add `import threading` and change `WhisperSTT`:

```python
    def __init__(self, model="large-v3-turbo", device="cuda", compute_type="float16", initial_prompt=None):
        from faster_whisper import WhisperModel
        _add_torch_dlls()
        self.initial_prompt = initial_prompt
        self._lock = threading.Lock()  # one model shared by every web session
```

(the rest of `__init__` is unchanged)

```python
    def transcribe(self, audio):
        """16 kHz mono float32 array -> Azerbaijani text ("" if no speech)."""
        with self._lock:  # the segments are decoded while they are read, so that stays inside
            segments, _ = self.model.transcribe(audio, language="az", beam_size=5, vad_filter=False,
                                                condition_on_previous_text=False,
                                                initial_prompt=self.initial_prompt)
            return clean_transcript(segments, self.initial_prompt)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stt.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add assistant/stt.py tests/test_stt.py
git commit -m "feat(assistant): share one Whisper model between threads"
```

---

### Task 5: The WebSocket server

**Files:**
- Create: `assistant/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Pipeline(..., on_event=...)` (Task 3); `SocketMic`, `SocketPlayer`, `Disconnected`, `IdleTimeout` (Task 2); `cfg["server"]["max_sessions"]`, `cfg["server"]["idle_timeout_s"]`, `cfg["chunker"]`, `cfg["playback"]["gap_ms"]`.
- Produces:
  - `create_app(stt, tts, fallback_wav, cfg, access_code, make_llm, make_segmenter, rtf=None, log=print) -> FastAPI`
  - `make_llm()` and `make_segmenter()` are called once per connection.
  - `app.state.sessions`: the set of connected sockets.
  - Constants `BAD_CODE = 4401`, `FULL = 4429`, `IDLE = 4408`.
  - `WS /ws?code=...`; messages as in the spec's *Protocol* table.

- [ ] **Step 1: Write the failing tests**

`tests/test_server.py`:

```python
import json
import time
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from assistant.server import BAD_CODE, FULL, IDLE, create_app

BLOCK = bytes(1024)  # 512 samples of 16-bit silence
CFG = {"server": {"max_sessions": 2, "idle_timeout_s": 300},
       "chunker": {"min_chars": 1, "max_chars": 180}, "playback": {"gap_ms": 150}}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def stream_reply(self, user_text):
        self.calls.append(user_text)
        yield "Bəli, hazıram."


class FakeTTS:
    def synthesize(self, text):
        return np.full(4, 0.5, dtype=np.float32)


class FakeSTT:
    def transcribe(self, audio):
        return "Salam"


class FakeSegmenter:
    """Every block is a whole utterance."""

    def feed(self, block):
        return block

    def reset(self):
        pass


def make_app(code="secret", **server):
    llms = []

    def make_llm():
        llms.append(FakeLLM())
        return llms[-1]

    cfg = {**CFG, "server": {**CFG["server"], **server}}
    app = create_app(FakeSTT(), FakeTTS(), np.zeros(4, dtype=np.float32), cfg, code, make_llm, FakeSegmenter,
                     log=lambda line: None)
    return app, llms


def connect(app, code="secret"):
    return TestClient(app).websocket_connect("/ws?code=" + quote(code))


def turn(ws):
    """Send one block of audio; the reply's messages up to reply_end as (kind, payload)."""
    ws.send_bytes(BLOCK)
    out = []
    while True:
        message = ws.receive()
        if message.get("bytes") is not None:
            out.append(("audio", message["bytes"]))
            continue
        data = json.loads(message["text"])
        out.append((data["type"], data))
        if data["type"] == "reply_end":
            return out


def close_code(ws):
    with pytest.raises(WebSocketDisconnect) as e:
        ws.receive_text()
    return e.value.code


def wait_until(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


def test_a_turn_returns_text_audio_and_timings_in_order():
    app, _ = make_app()
    with connect(app) as ws:
        out = turn(ws)
    assert [kind for kind, _ in out] == ["user_text", "audio", "reply_text", "latency", "reply_end"]
    assert out[0][1]["text"] == "Salam" and out[2][1]["text"] == "Bəli, hazıram."
    assert np.frombuffer(out[1][1], dtype="<i2").tolist() == [16383] * 4
    assert "first_audio" in out[3][1]


def test_wrong_or_missing_access_code_is_refused():
    app, llms = make_app()
    with connect(app, "wrong") as ws:
        assert close_code(ws) == BAD_CODE
    with TestClient(app).websocket_connect("/ws") as ws:
        assert close_code(ws) == BAD_CODE
    assert llms == []


def test_access_code_with_non_ascii_letters():
    app, _ = make_app(code="gizli-söz")
    with connect(app, "gizli-soz") as ws:
        assert close_code(ws) == BAD_CODE
    with connect(app, "gizli-söz") as ws:
        assert turn(ws)[0][0] == "user_text"


def test_connection_beyond_the_limit_is_refused():
    app, _ = make_app(max_sessions=1)
    with connect(app) as first:
        turn(first)
        with connect(app) as second:
            assert close_code(second) == FULL


def test_each_session_has_its_own_conversation():
    app, llms = make_app()
    with connect(app) as a, connect(app) as b:
        turn(a)
        turn(b)
    assert [llm.calls for llm in llms] == [["Salam"], ["Salam"]]


def test_closing_the_page_frees_the_session():
    app, _ = make_app(max_sessions=1)
    with connect(app) as ws:
        turn(ws)
    assert wait_until(lambda: not app.state.sessions)
    with connect(app) as ws:
        assert turn(ws)[-1][0] == "reply_end"


def test_a_text_message_from_the_client_is_ignored():
    app, _ = make_app()
    with connect(app) as ws:
        ws.send_text("hello")
        assert turn(ws)[0][0] == "user_text"


def test_a_visitor_who_never_speaks_is_closed_and_the_slot_freed():
    app, _ = make_app(max_sessions=1, idle_timeout_s=0.2)
    with connect(app) as ws:
        assert close_code(ws) == IDLE
    assert wait_until(lambda: not app.state.sessions)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'assistant.server'`.

- [ ] **Step 3: Implement**

`assistant/server.py`:

```python
"""Web demo: one WebSocket per visitor, each running the Pipeline against the shared STT and TTS models."""
import asyncio
import hmac
import threading

from fastapi import FastAPI, WebSocket

from assistant.pipeline import Pipeline
from assistant.web_io import Disconnected, IdleTimeout, SocketMic, SocketPlayer

BAD_CODE, FULL, IDLE = 4401, 4429, 4408  # WebSocket close codes the page explains to the visitor
_SEND_TIMEOUT_S = 10


def create_app(stt, tts, fallback_wav, cfg, access_code, make_llm, make_segmenter, rtf=None, log=print):
    """`stt` and `tts` are shared; make_llm() and make_segmenter() are called once per connection."""
    app = FastAPI()
    sessions = app.state.sessions = set()  # only touched on the event loop

    @app.websocket("/ws")
    async def ws(socket: WebSocket, code: str = ""):
        await socket.accept()  # a close code only reaches the browser on an accepted connection
        if not hmac.compare_digest(code.encode("utf-8"), access_code.encode("utf-8")):
            return await socket.close(BAD_CODE)
        if len(sessions) >= cfg["server"]["max_sessions"]:
            return await socket.close(FULL)

        loop, closed, done, result = asyncio.get_running_loop(), threading.Event(), asyncio.Event(), {}

        def send(make_coro):
            """Called in the session's thread; blocks until the message is handed to the socket."""
            if closed.is_set():
                raise Disconnected("connection closed")
            try:
                asyncio.run_coroutine_threadsafe(make_coro(), loop).result(_SEND_TIMEOUT_S)
            except Exception as e:
                closed.set()
                raise Disconnected(f"{type(e).__name__}: {e}") from e

        def on_event(kind, data):
            message = {"type": kind}
            if kind == "latency":
                message.update(data)
            elif data is not None:
                message["text"] = data
            send(lambda: socket.send_json(message))

        mic = SocketMic(cfg["server"]["idle_timeout_s"])
        player = SocketPlayer(lambda data: send(lambda: socket.send_bytes(data)))
        pipeline = Pipeline(make_llm(), tts, player, mic=mic, segmenter=make_segmenter(), stt=stt,
                            chunker=cfg["chunker"], gap_ms=cfg["playback"]["gap_ms"],
                            tail_ms=0,  # the page keeps its microphone muted while it plays the reply
                            fallback_wav=fallback_wav, log=log, rtf=rtf, on_event=on_event)

        def work():
            try:
                pipeline.run()
            except IdleTimeout:
                result["close"] = IDLE
            except Disconnected:
                pass
            except Exception as e:  # one session's crash must not take the others down
                log(f"[session] {type(e).__name__}: {e}")
                result["close"] = 1011
            finally:
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:  # the server is shutting down
                    pass

        async def receive():
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes"):
                    mic.push(message["bytes"])

        sessions.add(socket)
        threading.Thread(target=work, daemon=True).start()
        receiver, finished = asyncio.ensure_future(receive()), asyncio.ensure_future(done.wait())
        try:
            await asyncio.wait({receiver, finished}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            closed.set()
            mic.close()
            sessions.discard(socket)
            receiver.cancel()
            finished.cancel()
            if "close" in result:
                try:
                    await socket.close(result["close"])
                except RuntimeError:  # already closed by the visitor
                    pass

    return app
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -q`
Expected: 8 passed.

- [ ] **Step 5: Run the whole suite and commit**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: all pass.

```bash
git add assistant/server.py tests/test_server.py
git commit -m "feat(assistant): WebSocket server with one pipeline per visitor"
```

---

### Task 6: The web page

**Files:**
- Create: `assistant/web/index.html`, `assistant/web/app.js`, `assistant/web/mic-worklet.js`
- Modify: `assistant/server.py` (serve the page)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `WS /ws?code=...`, close codes 4401 / 4429 / 4408, the JSON messages `user_text`, `reply_text`, `latency`, `reply_end`.
- Produces: `GET /` (the page), `GET /static/app.js`, `GET /static/mic-worklet.js`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
def test_the_page_and_its_scripts_are_served():
    app, _ = make_app()
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200 and "/static/app.js" in page.text
    assert "AudioWorkletNode" in client.get("/static/app.js").text
    assert "registerProcessor" in client.get("/static/mic-worklet.js").text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -q -k page_and_its_scripts`
Expected: FAIL, status 404.

- [ ] **Step 3: Write the page**

`assistant/web/index.html`:

```html
<!doctype html>
<html lang="az">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Səsli köməkçi</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 34rem; margin: 0 auto; padding: 1rem; }
  h1 { font-size: 1.3rem; }
  form { display: flex; gap: .5rem; }
  input, button { font: inherit; padding: .6rem .8rem; border-radius: .5rem; border: 1px solid #8886; }
  input { flex: 1; min-width: 0; }
  button { background: #2563eb; color: #fff; border-color: #2563eb; }
  button:disabled { opacity: .5; }
  #status { margin: 1rem 0; font-weight: 600; min-height: 1.5em; }
  #log p { margin: .4rem 0; }
  #log .who { opacity: .6; margin-right: .4rem; }
  #latency { font-size: .8rem; opacity: .6; }
</style>
</head>
<body>
<h1>Səsli köməkçi</h1>
<form id="form">
  <input id="code" type="password" placeholder="Giriş kodu" autocomplete="off" required>
  <button id="start">Başla</button>
</form>
<div id="status" aria-live="polite"></div>
<div id="log"></div>
<div id="latency"></div>
<script src="/static/app.js"></script>
</body>
</html>
```

`assistant/web/mic-worklet.js`:

```js
// Brings the microphone down to 16 kHz (averaging the input samples that fall into each output sample)
// and posts 512-sample blocks of 16-bit audio, the size the server's speech detector expects.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // input samples per output sample (1 if the context is already 16 kHz)
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.block = new Int16Array(512);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i];
      this.count += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const sample = Math.max(-1, Math.min(1, this.sum / this.count));
        this.sum = 0;
        this.count = 0;
        this.block[this.filled++] = sample * 32767;
        if (this.filled === this.block.length) {
          this.port.postMessage(this.block.buffer.slice(0));
          this.filled = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic", MicProcessor);
```

`assistant/web/app.js`:

```js
// Page of the voice assistant: microphone -> WebSocket -> server, reply audio -> speaker.
const OUT_RATE = 24000;   // the server's reply audio
const TAIL_S = 0.3;       // keep the microphone muted this long after the reply ends
const CLOSED = {
  4401: "Giriş kodu yanlışdır.",
  4429: "Demo hazırda doludur. Bir az sonra yenidən cəhd edin.",
  4408: "Uzun müddət səs gəlmədiyi üçün bağlantı bağlandı.",
};
const $ = (id) => document.getElementById(id);

let socket = null, stream = null, micCtx = null, playCtx = null;
let playUntil = 0;        // playCtx time at which the queued reply audio ends
let thinking = false;

function setStatus(text) { $("status").textContent = text; }

function addLine(who, text) {
  const p = document.createElement("p"), label = document.createElement("span");
  label.className = "who";
  label.textContent = who + ":";
  p.append(label, text);
  $("log").append(p);
  p.scrollIntoView({ block: "end" });
}

// Half-duplex: while the reply plays (and shortly after), nothing from the microphone is sent,
// so the assistant never hears its own voice from the speaker.
function speaking() { return playCtx.currentTime < playUntil + TAIL_S; }

function play(data) {
  const pcm = new Int16Array(data);
  if (!pcm.length) return;
  const buffer = playCtx.createBuffer(1, pcm.length, OUT_RATE), channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const source = playCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(playCtx.destination);
  playUntil = Math.max(playUntil, playCtx.currentTime + 0.05);
  source.start(playUntil);
  playUntil += buffer.duration;
  thinking = false;
}

function stop(message) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (micCtx) micCtx.close();
  if (playCtx) playCtx.close();
  stream = micCtx = playCtx = socket = null;
  $("start").disabled = false;
  setStatus(message);
}

async function start() {
  $("start").disabled = true;
  playCtx = new AudioContext();  // created in the tap handler: phones only allow audio after a tap
  playCtx.resume();
  try {
    stream = await navigator.mediaDevices.getUserMedia(
      { audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  } catch (e) {
    return stop("Mikrofona icazə verilmədi. Brauzerin tənzimləmələrində icazə verin.");
  }
  try {
    micCtx = new AudioContext({ sampleRate: 16000 });  // let the browser resample where it can
  } catch (e) {
    micCtx = new AudioContext();
  }
  await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
  const mic = new AudioWorkletNode(micCtx, "mic");
  micCtx.createMediaStreamSource(stream).connect(mic);

  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/ws?code=${encodeURIComponent($("code").value)}`);
  socket.binaryType = "arraybuffer";
  mic.port.onmessage = (e) => {
    if (socket && socket.readyState === WebSocket.OPEN && !speaking()) socket.send(e.data);
  };
  socket.onopen = () => setStatus("Dinləyirəm…");
  socket.onmessage = (e) => {
    if (typeof e.data !== "string") return play(e.data);
    const message = JSON.parse(e.data);
    if (message.type === "user_text") {
      addLine("Siz", message.text);
      thinking = true;
    } else if (message.type === "reply_text") {
      addLine("Köməkçi", message.text);
    } else if (message.type === "latency" && message.first_audio !== undefined) {
      $("latency").textContent = `Cavab ${(message.first_audio / 1000).toFixed(1)} saniyəyə başladı`;
    } else if (message.type === "reply_end") {
      thinking = false;
    }
  };
  socket.onclose = (e) => stop(CLOSED[e.code] || "Bağlantı kəsildi. Yenidən başlayın.");
}

setInterval(() => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  setStatus(playCtx.currentTime < playUntil ? "Danışıram…" : thinking ? "Düşünürəm…" : "Dinləyirəm…");
}, 100);

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  start();
});
```

- [ ] **Step 4: Serve it**

In `assistant/server.py` add the imports and, inside `create_app` right after `sessions = ...`, the routes:

```python
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).parent / "web"
```

```python
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -q`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add assistant/web/index.html assistant/web/app.js assistant/web/mic-worklet.js assistant/server.py tests/test_server.py
git commit -m "feat(assistant): web page for the voice assistant"
```

---

### Task 7: `scripts/serve.py`

**Files:**
- Create: `scripts/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `talk.load_env_file`, `talk.ENV_FILE`, `talk.NO_KEY`; `create_app` (Task 5); `load_config`, `FALLBACK_TEXT`; `TTSService`, `WhisperSTT`, `GeminiLLM`, `silero_segmenter`, `VadConfig`.
- Produces: `serve.main(argv=None)`; `serve.NO_CODE` message.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_serve.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'serve'`.

- [ ] **Step 3: Implement**

`scripts/serve.py`:

```python
"""Serve the Azerbaijani voice assistant as a web page: browser -> Whisper -> Gemini -> StyleTTS2 -> browser."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import talk  # noqa: E402  (same folder: the .env loader and the missing-key message)
from assistant.audio_io import OUT_RATE  # noqa: E402
from assistant.llm import GeminiLLM  # noqa: E402
from assistant.pipeline import FALLBACK_TEXT, load_config  # noqa: E402
from assistant.text_norm import to_spoken  # noqa: E402

NO_CODE = ("ASSISTANT_ACCESS_CODE is not set. Choose a code visitors must type on the page and put this line\n"
           "in the .env file in the repo root:\n"
           "    ASSISTANT_ACCESS_CODE=<your code>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/assistant.yml"))
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    talk.load_env_file(talk.ENV_FILE)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(talk.NO_KEY)
    code = os.environ.get("ASSISTANT_ACCESS_CODE")
    if not code:
        sys.exit(NO_CODE)

    from assistant.tts import TTSService  # torch and StyleTTS2: only after the cheap checks passed
    print("[serve] loading the TTS model ...")
    tts = TTSService(**cfg["tts"])
    start = time.perf_counter()
    fallback = tts.synthesize(to_spoken(FALLBACK_TEXT))
    rtf = (time.perf_counter() - start) / (len(fallback) / OUT_RATE)  # TTS speed on this machine

    import uvicorn
    from assistant.server import create_app
    from assistant.stt import WhisperSTT
    from assistant.vad import VadConfig, silero_segmenter
    print("[serve] loading the STT model ...")
    stt = WhisperSTT(**cfg["stt"])
    app = create_app(stt, tts, fallback, cfg, code, make_llm=lambda: GeminiLLM(**cfg["llm"]),
                     make_segmenter=lambda: silero_segmenter(VadConfig(**cfg["vad"])), rtf=rtf)
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    print(f"[serve] open http://{host}:{port} (Ctrl+C to quit)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Azerbaijani letters on a cp1252 Windows console
    main()
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_serve.py tests/test_talk.py -q`
Expected: all pass.

- [ ] **Step 5: Start it for real**

Add `ASSISTANT_ACCESS_CODE=<a code of your choice>` to `.env` (ask the user for the value; do not invent or print one).
Run: `.venv\Scripts\python.exe scripts\serve.py`
Expected: no `cuda unavailable` line, then `[serve] open http://127.0.0.1:8000`. Open that address in a browser on this PC, enter the code, press *Başla*, allow the microphone, and say a sentence. Expected: the status goes *Dinləyirəm → Düşünürəm → Danışıram*, the transcript shows both lines, and the reply is heard. Stop the server with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add scripts/serve.py tests/test_serve.py
git commit -m "feat: serve.py, the web demo entry point"
```

---

### Task 8: `setup_inference.py --gpu`

**Files:**
- Modify: `scripts/setup_inference.py`
- Test: `tests/test_setup_inference.py`

**Interfaces:**
- Produces: `torch_plan(installed: str | None, gpu: bool, platform: str) -> tuple[bool, str | None] | None` — `(uninstall_first, index_url)` to get the wanted PyTorch build, or `None` if it is already installed; `main(argv=None)` accepting `--gpu`.

- [ ] **Step 1: Write the failing tests**

`tests/test_setup_inference.py`:

```python
from setup_inference import CPU_INDEX, GPU_INDEX, torch_plan


def test_fresh_environment_gets_the_cpu_build():
    assert torch_plan(None, gpu=False, platform="win32") == (False, CPU_INDEX)


def test_macos_installs_from_the_default_index():
    assert torch_plan(None, gpu=False, platform="darwin") == (False, None)


def test_an_installed_torch_is_left_alone_without_the_gpu_flag():
    assert torch_plan("2.11.0+cpu", gpu=False, platform="win32") is None
    assert torch_plan("2.11.0+cu128", gpu=False, platform="win32") is None


def test_gpu_flag_installs_the_cuda_build():
    assert torch_plan(None, gpu=True, platform="win32") == (False, GPU_INDEX)


def test_gpu_flag_replaces_a_cpu_build():
    assert torch_plan("2.14.1+cpu", gpu=True, platform="win32") == (True, GPU_INDEX)


def test_gpu_flag_keeps_an_installed_cuda_build():
    assert torch_plan("2.11.0+cu128", gpu=True, platform="win32") is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_setup_inference.py -q`
Expected: `ImportError: cannot import name 'CPU_INDEX'`.

- [ ] **Step 3: Implement**

In `scripts/setup_inference.py`: replace `import importlib.util` with `import argparse` and `import importlib.metadata`, update the docstring's install sentence to

```
Installs torch (the CPU build, or the CUDA 12.8 build with --gpu) and the requirements, clones StyleTTS2 at the
pinned commit and applies patches/styletts2.patch. Safe to re-run.
```

add below `MIN_WEIGHTS_BYTES`:

```python
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
GPU_INDEX = "https://download.pytorch.org/whl/cu128"  # same build as setup_env.ps1
```

add below `is_lfs_pointer`:

```python
def torch_plan(installed, gpu, platform):
    """(uninstall_first, index_url) to get the wanted torch build, or None if it is already installed.
    `installed` is the installed torch version, or None."""
    if gpu:
        if installed and "+cu" in installed:
            return None
        return installed is not None, GPU_INDEX
    if installed:
        return None
    return False, None if platform == "darwin" else CPU_INDEX


def installed_torch():
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
```

and change the start of `main`:

```python
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true",
                    help="install the CUDA 12.8 build of PyTorch (NVIDIA GPU), replacing a CPU build")
    a = ap.parse_args(argv)

    pip = (sys.executable, "-m", "pip", "install")
    run(*pip, "--upgrade", "pip")
    plan = torch_plan(installed_torch(), a.gpu, sys.platform)
    if plan:
        uninstall, index = plan
        if uninstall:
            run(sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio")
        run(*pip, "torch", "torchaudio", *(("--index-url", index) if index else ()))
```

(the lines from `run(*pip, "-r", "requirements.txt", ...)` on are unchanged)

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_setup_inference.py -q`
Expected: 6 passed.
Run: `.venv\Scripts\python.exe scripts\setup_inference.py --gpu`
Expected: no torch reinstall (the CUDA build is already there), ends with `Ready. Voice model: ...`.
Run: `.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.11.0+cu128 True`

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_inference.py tests/test_setup_inference.py
git commit -m "feat: setup_inference.py --gpu installs the CUDA build of PyTorch"
```

---

### Task 9: Latency check, README, and the measurement

**Files:**
- Create: `scripts/measure_latency.py`
- Modify: `README.md` (new *Web demo* section before *Troubleshooting*; one row in the *Layout* table)

**Interfaces:**
- Consumes: a running `scripts/serve.py`; `talk.load_env_file`; the `websockets` package (already installed as a dependency of `google-genai` and `uvicorn[standard]`).

This script needs a running server, a GPU and Gemini, so it has no unit test; its output is the plan's acceptance evidence.

- [ ] **Step 1: Write the script**

`scripts/measure_latency.py`:

```python
"""Measure the web demo's reply delay: stream a recorded question to a running server the way the page does.

    python scripts/serve.py                                   (in another terminal)
    python scripts/measure_latency.py --wav out/01.wav
    python scripts/measure_latency.py --wav out/01.wav --sessions 5

The WAV should end where the speech ends: the delay is counted from its last sample.
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import talk  # noqa: E402

RATE, BLOCK = 16000, 512
BLOCK_S = BLOCK / RATE
SILENCE = bytes(2 * BLOCK)
GIVE_UP_S = 30


def load_pcm(path):
    """Any WAV -> 16 kHz mono 16-bit little-endian bytes."""
    import librosa
    import numpy as np
    audio, _ = librosa.load(path, sr=RATE, mono=True)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def run_session(url, pcm, results, index):
    from websockets.sync.client import connect
    seen = {}
    with connect(url, max_size=None) as ws:
        def receive():
            try:
                for message in ws:
                    if isinstance(message, bytes):
                        seen.setdefault("audio", time.perf_counter())
                    elif '"latency"' in message:
                        seen["server"] = message
                    elif '"reply_end"' in message:
                        return
            except Exception as e:
                seen["error"] = f"{type(e).__name__}: {e}"

        receiver = threading.Thread(target=receive, daemon=True)
        receiver.start()
        for i in range(0, len(pcm), 2 * BLOCK):  # in real time, like a microphone
            ws.send(pcm[i:i + 2 * BLOCK])
            time.sleep(BLOCK_S)
        end = time.perf_counter()
        while receiver.is_alive() and time.perf_counter() - end < GIVE_UP_S:
            if "audio" not in seen:  # the page stops sending once the reply starts playing
                ws.send(SILENCE)
            time.sleep(BLOCK_S)
    results[index] = (seen.get("audio", math.nan) - end, seen.get("server"), seen.get("error"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="a recorded Azerbaijani question")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--sessions", type=int, default=1, help="visitors speaking at the same moment")
    a = ap.parse_args(argv)

    talk.load_env_file(talk.ENV_FILE)
    code = os.environ.get("ASSISTANT_ACCESS_CODE", "")
    pcm, results = load_pcm(a.wav), [None] * a.sessions
    threads = [threading.Thread(target=run_session, args=(f"{a.url}?code={quote(code)}", pcm, results, i))
               for i in range(a.sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, (delay, server, error) in enumerate(results, 1):
        print(f"session {i}: first audio {delay:.2f} s after the end of speech | server: {server} | error: {error}")
    worst = max(delay for delay, _, _ in results)
    print(f"worst: {worst:.2f} s ({'OK' if worst < 3 else 'over the 3 s target'})")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    main()
```

- [ ] **Step 2: Measure**

Make a question to send (the project's own voice stands in for a visitor), using a text file so no non-ASCII reaches the shell: there is already `out\01.wav` ("Salam, bu gün hava çox gözəldir.") from the quick start; if it is missing, run the README's `synthesize.py` example first.

Terminal 1: `.venv\Scripts\python.exe scripts\serve.py`
Terminal 2: `.venv\Scripts\python.exe scripts\measure_latency.py --wav out\01.wav`
Expected: `worst: X.XX s (OK)` with X.XX under 3. Record the session line and the server's `latency` values.

Then: `.venv\Scripts\python.exe scripts\measure_latency.py --wav out\01.wav --sessions 5`
Expected: all five sessions answered, worst under 3 s. This spends 5 Gemini requests; with the single run above it is 6 of the free plan's 20 a day, so do not repeat it needlessly.

If the single-session figure is above 2.7 s, set `vad.end_silence_ms: 400` in `configs/assistant.yml`, restart the server, measure once more, and record both figures. If a figure is still above 3 s, stop and report the numbers to the user instead of tuning further.

- [ ] **Step 3: README**

Add this row to the *Layout* table, after the `assistant/`, `scripts/talk.py` row:

```markdown
| `scripts/serve.py`, `assistant/server.py`, `assistant/web/` | Web demo: the same assistant as a web page for phones and laptops (see *Web demo*). |
```

Add this section before `## Troubleshooting`, replacing the two measured figures with the ones from Step 2:

````markdown
## Web demo

The same assistant as a web page: visitors open a link on a phone or laptop, type the access code,
and talk hands-free. The models run on this machine's GPU; the page only streams audio.

```powershell
.\.venv\Scripts\python.exe scripts\setup_inference.py --gpu   # once: the CUDA build of PyTorch (NVIDIA GPU)
# in .env, next to GEMINI_API_KEY:  ASSISTANT_ACCESS_CODE=<a code visitors must type>
.\.venv\Scripts\python.exe scripts\serve.py                   # then open http://127.0.0.1:8000
```

To let other people in, give the local server a public https address with a tunnel (browsers only
allow the microphone over https):

```powershell
winget install Cloudflare.cloudflared      # once
cloudflared tunnel --url http://127.0.0.1:8000
```

It prints a `https://....trycloudflare.com` link; send it together with the access code. The link
changes every time the tunnel is restarted, and the demo is only up while this PC and both commands
are running.

How it behaves:

- **Up to 5 visitors at once** (`server.max_sessions`); one more is told the demo is full. Each has
  a separate conversation. A connection that sends no audio for 5 minutes is closed.
- **Half-duplex**, as in `talk.py`: the page mutes its microphone while the reply plays.
- **Delay:** measured <X.X> s from the end of speech to the first audio with one visitor and <Y.Y> s
  with five speaking at the same moment. Check it yourself with the server running:
  `.\.venv\Scripts\python.exe scripts\measure_latency.py --wav out\01.wav` (add `--sessions 5`).
- **Gemini plan:** the free plan's daily request limit is used up quickly by several visitors; use a
  paid key for a demo. The key and the access code stay on this machine and are never sent to the page.

Do not run the server during a training run: both need the GPU.
````

- [ ] **Step 4: Check on a phone**

With `serve.py` and the tunnel running, open the tunnel's link on a phone (mobile data, not the same Wi-Fi), enter the code, and hold a three-turn conversation on the phone's speaker. Expected: each reply starts within about 3 s, and the assistant does not answer its own voice. Also enter a wrong code once (expected: *Giriş kodu yanlışdır.*) and refuse the microphone once (expected: the *Mikrofona icazə verilmədi* message). This step needs the user; if they are not available, report it as not done.

- [ ] **Step 5: Run everything and commit**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: all pass.

```bash
git add scripts/measure_latency.py README.md
git commit -m "docs: web demo instructions and the latency check"
```

If `vad.end_silence_ms` was changed in Step 2, add `configs/assistant.yml` to the same commit and say so in its message.
