# Web voice assistant (demo release) — design

Date: 2026-10-03

## Goal

People open a web page on a phone or laptop and talk hands-free with the
Azerbaijani assistant, which answers in this project's StyleTTS2 voice. The reply
must start within 2–3 s of the user finishing their sentence.

Decisions made with the user:

- **Scale:** a demo, 1–5 people talking at the same time.
- **Client:** a web page in the browser; nothing to install.
- **Hosting:** this PC (RTX 5060, 8 GB) behind a tunnel that gives a public https link.
- **Turn-taking:** hands-free and half-duplex, as in `scripts/talk.py`: the server
  detects the end of speech; the user cannot interrupt a reply.
- **Transport:** raw audio over one WebSocket, all logic on the server
  (chosen over WebRTC and over end-of-speech detection in the browser).

Success means: a visitor with the link and the access code speaks a question and
hears the first audio of the answer less than 3 s after they stop talking,
measured by `scripts/measure_latency.py` on this PC, with up to 5 visitors connected.

## What exists today

`assistant/` implements the same conversation for the local microphone and
speakers (see `2026-10-01-voice-assistant-design.md`):

- `Pipeline(llm, tts, player, mic, segmenter, stt, ...)` runs the turn loop in the
  calling thread. It needs a `mic` with `start()`, `stop()`, `read(timeout)`,
  `drain()` and a `player` with `play(wav)`; everything else is already
  independent of the sound card.
- `TTSService.synthesize` is guarded by a lock, so one loaded voice can be shared
  by several threads. `WhisperSTT.transcribe` has no lock.
- `GeminiLLM` holds one conversation's history.
- `Segmenter` holds one audio stream's state (Silero VAD is stateful).

Measured on this PC on 2026-10-03, per turn:

| Stage | CPU PyTorch | GPU PyTorch (cu128) |
|---|---|---|
| End-of-speech silence (`vad.end_silence_ms`) | 0.6 s | 0.6 s |
| Whisper `large-v3-turbo`, 4 s question | about 6 s | 0.25 s |
| Gemini, first sentence (`llm_first_chunk`) | 0.7–1.2 s | not re-measured (network) |
| TTS, first sentence | 1.1–2.3 s | 0.24 s |

The models only meet the target on a GPU, which is why they run on a server and
not on the visitor's device.

## Architecture

```
browser page                                   this PC
------------                                   -------
microphone -> worklet (16 kHz, 512 samples) --WebSocket--> SocketMic -> Segmenter -> WhisperSTT
                                                                                        |
speaker <- playback queue <----------WebSocket---- SocketPlayer <- TTSService <- GeminiLLM
status line, transcript <----------- JSON events ------------- Pipeline
```

- One process, started by `scripts/serve.py`: a FastAPI app served by uvicorn.
- `WhisperSTT` and `TTSService` are loaded once at startup and shared.
- Each WebSocket connection is a **session**: its own `GeminiLLM`, `Segmenter`,
  `SocketMic`, `SocketPlayer` and `Pipeline`, with `Pipeline.run()` in its own
  thread. The asyncio side only moves bytes between the socket and the session.
- The tunnel (Cloudflare quick tunnel) forwards a public https address to
  `127.0.0.1:8000`. It is run by hand and documented in the README.

## Components

### `assistant/web_io.py` (new)

- `Disconnected(Exception)` — the visitor is gone or idle.
- `SocketMic(idle_timeout_s)` — stands in for `MicStream`.
  - `push(data: bytes)` (called from the event loop): 16-bit little-endian mono
    16 kHz audio of any length; re-blocks it into 512-sample float32 blocks.
  - `read(timeout)` returns the next block or `None`; raises `Disconnected` after
    `close()` or when no audio has arrived for `idle_timeout_s`.
  - `start()`, `stop()` do nothing; `drain()` discards queued blocks; `close()`.
- `SocketPlayer(send)` — stands in for `Player`.
  - `play(wav)` converts 24 kHz float32 to 16-bit little-endian and calls
    `send(bytes)`, which blocks until the frame is handed to the socket and raises
    `Disconnected` once the connection is closed. It does not wait for the audio
    to be heard.

### `assistant/server.py` (new)

- `create_app(stt, tts, fallback_wav, cfg, access_code, make_llm, make_segmenter)`
  returns the FastAPI app; the factories are injected so tests use fakes.
- `GET /` and `/static/*` serve `assistant/web/`.
- `WS /ws?code=<access code>`:
  - wrong or missing code: close with code 4401 before accepting any audio
    (compared with `hmac.compare_digest`);
  - `server.max_sessions` already connected: close with code 4429;
  - otherwise create a session, start its thread, forward binary messages to
    `SocketMic.push`, and on disconnect call `mic.close()` and mark the player
    closed. The thread ends on `Disconnected`.
- Session events are sent as JSON text messages (see *Protocol*).

### `assistant/pipeline.py` (changed)

- New optional `on_event(kind, data)` callback, called with `("user_text", text)`
  when a turn is recognised, `("reply_text", text)` and `("latency", marks)` at the
  end of a reply, and `("reply_end", None)` after the last audio was sent.
  `scripts/talk.py` does not pass it.
- `_play` sets the turn's stop flag when the player raises, so a closed connection
  stops the TTS worker instead of synthesizing the rest of the reply.
- `load_config` requires the new `server` section.

### `assistant/stt.py` (changed)

- `transcribe` takes a lock around the model call and the reading of its
  segments, so sessions use the one Whisper model in turn.

### `assistant/web/` (new)

- `index.html` — access-code field, Start button, status line
  (listening / thinking / speaking), and the transcript of the conversation.
- `app.js` — opens the WebSocket, starts the microphone with `echoCancellation`
  and `noiseSuppression`, queues and plays incoming audio back to back at 24 kHz
  with the Web Audio API, shows events, and explains errors in plain words
  (microphone refused, wrong code, demo full, connection lost).
  **Half-duplex:** it sends no microphone audio from the first reply frame until
  0.3 s after its playback queue is empty, so the assistant does not hear itself.
- `mic-worklet.js` — an `AudioWorkletProcessor` that resamples the microphone
  from the browser's rate to 16 kHz and posts 512-sample 16-bit blocks.

### `scripts/serve.py` (new)

- Loads `configs/assistant.yml` and `.env`, exits with a clear message if
  `GEMINI_API_KEY` or `ASSISTANT_ACCESS_CODE` is missing, loads the TTS and STT
  models, measures the TTS speed as `talk.py` does, and runs uvicorn on
  `server.host`:`server.port`.

### `scripts/measure_latency.py` (new)

- Connects to a running server, streams a WAV file of a spoken question in real
  time followed by silence, and prints the time from the end of the speech to the
  first audio frame, plus the server's `latency` event. Used for the success
  criterion and to check 5 simultaneous sessions (`--sessions 5`).

### Setup and configuration (changed)

- `requirements-assistant.txt`: add `fastapi` and `uvicorn[standard]`.
- `scripts/setup_inference.py --gpu`: install PyTorch from the `cu128` index
  instead of the CPU index, replacing a CPU build if one is installed.
- `configs/assistant.yml`, new section:

```yaml
server:
  host: 127.0.0.1        # the tunnel connects locally; nothing is exposed on the LAN
  port: 8000
  max_sessions: 5
  idle_timeout_s: 300
```

- `README.md`: a *Web demo* section (GPU setup, `serve.py`, the access code,
  starting the tunnel, the latency check).

## Protocol

| Direction | Message | Meaning |
|---|---|---|
| up | binary | 16 kHz mono 16-bit little-endian audio (32 KB/s) |
| down | binary | 24 kHz mono 16-bit little-endian audio of the reply (48 KB/s) |
| down | `{"type": "user_text", "text": ...}` | what was recognised; the page shows *thinking* |
| down | `{"type": "reply_text", "text": ...}` | the full reply |
| down | `{"type": "reply_end"}` | no more audio for this turn |
| down | `{"type": "latency", "stt": ms, "llm_first_chunk": ms, "tts_first": ms, "first_audio": ms}` | server-side timings |

Close codes: 4401 wrong access code, 4429 demo full, 4408 idle timeout.

## Latency and capacity budget

| Stage | Budget |
|---|---|
| End-of-speech silence | 0.6 s |
| Upload of the last blocks | 0.1 s |
| Whisper | 0.25 s |
| Gemini first sentence | 1.0 s (measured 0.7–1.2 s) |
| TTS first sentence | 0.25 s |
| Download and playback start | 0.1 s |
| **Total** | **about 2.3 s** |

- If the measured total is above 2.7 s, lower `vad.end_silence_ms` to 400.
- The GPU is shared: a session that finishes speaking while another is being
  transcribed or synthesized waits for it, at most a few tenths of a second each.
  With 5 sessions the worst case stays under 3 s only if Gemini is near 1 s.
- GPU memory: the local assistant already runs both models together on this 8 GB
  card, and they are loaded once however many sessions connect; a session adds
  only a Silero VAD model, on the CPU.

## Error handling

- Gemini error or quota exhausted: the fallback phrase is spoken (existing behaviour).
- Speech recognition error or nothing intelligible: the turn is dropped silently.
- Visitor disconnects mid-reply: the send raises `Disconnected`, the turn's stop
  flag is set, the session thread ends and the session is freed.
- No recognised speech for `idle_timeout_s`: the session is closed with code 4408
  (an open page streams silence continuously, so audio alone does not count as activity).
- A crash in one session's thread is logged and closes only that connection.

## Testing

- `tests/test_web_io.py`: re-blocking of odd-sized pushes, `read` timeout, `drain`,
  `Disconnected` after `close()` and after the idle timeout, 16-bit conversion and
  clipping in `SocketPlayer`.
- `tests/test_server.py` (FastAPI `TestClient`, fake STT/TTS/LLM/segmenter): wrong
  code is refused, the sixth connection is refused, a full turn returns
  `user_text`, audio frames, `reply_text`, `latency`, `reply_end` in order, two
  sessions keep separate histories, a disconnect mid-reply frees the session.
- `tests/test_pipeline.py`: `on_event` calls; a failing player stops the TTS worker.
- `tests/test_stt.py`: concurrent `transcribe` calls do not overlap.
- By hand: the page on a phone and a laptop through the tunnel (the JavaScript has
  no automated tests), and `measure_latency.py` with 1 and 5 sessions.

## Known limitations

- The tunnel's link changes on every restart, and the demo is only up while this
  PC is on and not training.
- The Gemini free plan allows 20 requests a day on `gemini-2.5-flash`; a paid key
  is needed for a demo.
- Whisper's Azerbaijani recognition errors are unchanged.
- Uncompressed audio; a poor mobile connection will stutter.
- One shared access code, no per-user limits.

## Out of scope

Accounts, saved chat history, interrupting the assistant mid-reply, transcribing
while the user is still speaking, cloud deployment, native apps, WebRTC.
