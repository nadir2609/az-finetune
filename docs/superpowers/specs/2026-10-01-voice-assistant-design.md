# Azerbaijani voice-to-voice assistant — design

Date: 2026-10-01

## Goal

A conversational assistant that runs on this Windows machine: the user speaks
Azerbaijani into the microphone, an LLM answers in Azerbaijani, and the answer is
spoken through the speakers by the project's fine-tuned StyleTTS2 voice.

Decisions made with the user:

- **Interface:** local microphone and speakers, started from the terminal.
- **LLM:** Gemini API on the free plan.
- **Turn-taking:** half-duplex. The microphone is ignored while the assistant
  speaks; the user cannot interrupt a reply.

Success means: a spoken question gets a spoken, correctly pronounced Azerbaijani
answer, the first audio starts about 2–2.5 s after the user stops talking, and
no LLM output (digits, markdown, foreign letters) can crash the TTS.

## What exists today

- `scripts/synthesize.py` — `Synth(config, checkpoint)` loads StyleTTS2 once.
  `compute_styles(refs)` returns the reference style. `synth(ipa, ref_s, alpha,
  beta, steps, embedding_scale, speed)` returns one utterance as a 24 kHz mono
  float numpy array. There is no streaming inside the model.
- `scripts/az_text.py` — `phonemize_texts(texts, g2p="rules")` normalizes and
  converts text to IPA. `scripts/az_g2p.py` raises `ValueError` on any character
  outside the Azerbaijani alphabet and `,.!?;:` (digits, `w`, brackets, …).
- `configs/inference.json` — tuned `alpha`, `beta`, `embedding_scale`, `speed`,
  `steps`, the 16 reference clips, and the checkpoint they were tuned on
  (`runs/az_ft_fresh2/epoch_2nd_00013.pth`).

Measured on the RTX 5060 (8 GB) with that checkpoint:

| Item | Result |
|---|---|
| Model load | 13.9 s (once) |
| Style from 16 reference clips | 4.5 s (once) |
| First synthesis call (cold) | 1.27 s |
| Synthesis per sentence (warm) | 0.39–0.47 s for 3–7 s of audio |
| Peak VRAM | 1.22 GB allocated, 1.36 GB reserved |

The model was trained on clips of 2.0–10.7 s, so chunks sent to it should stay
roughly in that range.

## Architecture

Plain Python threads and queues. Every library involved (faster-whisper,
PyTorch, sounddevice, google-genai streaming) is blocking, and there is no
network server, so asyncio or a voice framework would add weight without
benefit.

```
mic 16 kHz ─► Segmenter (VAD) ─► STT ─► Gemini (streaming) ─► SentenceChunker
                                                                     │ text_q
speakers 24 kHz ◄── Player ◄── audio_q ◄── TTSService ◄── to_spoken (normalizer)
```

New package `assistant/` at the repo root. Existing scripts are not modified;
`assistant/tts.py` puts `scripts/` on `sys.path` and imports `synthesize`, the
same way `scripts/batch_synth.py` does.

```
assistant/
  __init__.py
  audio_io.py     MicStream, Player
  vad.py          Segmenter
  stt.py          WhisperSTT
  llm.py          GeminiLLM
  chunker.py      SentenceChunker
  text_norm.py    to_spoken, number_to_words
  tts.py          TTSService
  pipeline.py     Pipeline, load_config
scripts/talk.py             entry point
configs/assistant.yml       settings
requirements-assistant.txt  faster-whisper, silero-vad, sounddevice, google-genai
tests/test_text_norm.py, test_chunker.py, test_vad.py, test_pipeline.py, test_llm.py
```

The new dependencies go in `requirements-assistant.txt`, not `requirements.txt`,
so the training environment setup is unchanged. `tests/conftest.py` gains the
repo root on `sys.path` so `import assistant` works.

## Components

### `audio_io.py`

- `MicStream(device=None)` — a `sounddevice.InputStream` at 16 kHz, mono,
  float32, block size 512 samples (32 ms). The audio callback only puts blocks
  on a queue. `read(timeout)` returns the next block or `None`. `drain()`
  discards everything queued.
- `Player(device=None)` — `play(wav)` writes a 24 kHz float32 array to a
  `sounddevice.OutputStream` and blocks until it has been played.
- `list_devices()` — prints input and output devices for `--list-devices`.

### `vad.py`

`Segmenter(speech_prob, cfg)` turns 512-sample blocks into utterances.
`speech_prob` is a callable `block -> float`; in production it is the Silero VAD
model (`load_silero_vad()`, called as `model(tensor, 16000).item()`), in tests a
scripted function. The state machine is in this module rather than Silero's
`VADIterator` so it can be tested without the model.

`feed(block) -> np.ndarray | None`:

- Idle: keep a 300 ms pre-roll ring buffer. Speech starts when the probability
  is ≥ `start_threshold` (0.5); the pre-roll becomes the start of the utterance.
- In speech: append blocks. The utterance ends when the probability has stayed
  below `end_threshold` (0.35) for `end_silence_ms` (600).
- An utterance with less than `min_speech_ms` (250) of speech blocks is
  discarded (coughs, clicks).
- An utterance reaching `max_utterance_s` (30) is ended and returned.

`reset()` clears buffers and calls the model's `reset_states()`.

### `stt.py`

`WhisperSTT(model="large-v3-turbo", device="cuda", compute_type="float16")`
wraps `faster_whisper.WhisperModel`.

`transcribe(audio) -> str` calls `model.transcribe(audio, language="az",
beam_size=5, condition_on_previous_text=False, vad_filter=False,
initial_prompt=<short Azerbaijani sentence from config>)`, drops segments with
`no_speech_prob > 0.6`, and returns the joined, stripped text.

faster-whisper on a Windows GPU needs the CUDA 12 cuBLAS and cuDNN DLLs. Before
creating the model, `stt.py` adds the `torch/lib` directory of the installed
torch to the DLL search path (`os.add_dll_directory`). If creating the model on
`cuda` still fails, it logs the error and falls back to `device="cpu",
compute_type="int8"`.

### `llm.py`

`GeminiLLM(model, system_prompt, max_turns=10, max_output_tokens=300)` uses the
`google-genai` SDK. The key is read from the `GEMINI_API_KEY` environment
variable; it is never stored in a config file.

`stream_reply(user_text) -> Iterator[str]` yields text deltas from
`client.models.generate_content_stream(model=..., contents=history, config=...)`
with `system_instruction` set and thinking disabled
(`ThinkingConfig(thinking_budget=0)`) to cut latency.

The class owns the history as a list of user and model turns:

- The user turn is appended before the call; the model turn (the text actually
  yielded) is appended when the stream ends.
- History is trimmed to the last `max_turns` exchanges.
- If the call fails, the user turn is removed again and `LLMError` is raised.
  A 429 raises `LLMRateLimited(LLMError)`.

The default system prompt (in `configs/assistant.yml`, in Azerbaijani) tells the
model to answer only in Azerbaijani, in one to three short spoken sentences,
without markdown, lists, emoji or abbreviations, and to write numbers as words.
The normalizer below does not depend on the model obeying this.

The model name is a config value (default `gemini-2.5-flash`), because the
free-tier model lineup changes.

### `chunker.py`

`SentenceChunker(min_chars=25, max_chars=180)` turns streamed deltas into
chunks to speak.

- `feed(delta) -> list[str]` appends to a buffer and returns every complete
  chunk. A boundary is `.`, `!` or `?` followed by whitespace, except a `.`
  between two digits (`3.5`).
- A sentence shorter than `min_chars` is held and merged with the next one.
- A sentence longer than `max_chars` is split at the last `,` or `;` before the
  limit, or at the last space if there is none.
- `flush() -> list[str]` returns whatever remains when the stream ends, even if
  it is short.

Chunks keep their final punctuation: the G2P uses `?` for question stress.

### `text_norm.py`

`to_spoken(text) -> str` makes any LLM text safe for `phonemize_texts`. It runs
on each chunk, in this order:

1. Markdown and symbols: `[text](url)` becomes `text`; bare URLs and emoji are
   removed; `* _ # \` ~ > |` and list bullets are removed; brackets and
   parentheses become commas; apostrophes are removed.
2. Numbers, via `number_to_words(n)` (Azerbaijani cardinals, 0 up to
   999 999 999 999):
   - Time `18:30` → `on səkkiz otuz`.
   - Decimal `3,5` or `3.5` → `üç vergül beş`.
   - Percent `50%` → `əlli faiz`.
   - Currency `₼` and `AZN` → `manat`, `$` → `dollar`, `€` → `avro`, placed
     after the number.
   - Ordinal suffix `5-ci`, `3-cü`, `6-cı`, `10-cu` → `beşinci`, `üçüncü`,
     `altıncı`, `onuncu` (suffix chosen by vowel harmony from the number word;
     `-nci` form after a final vowel).
   - Other suffixes `30-da` → `otuzda` (number word joined with the suffix).
   - Digit groups separated by spaces or dots as thousands (`1 500`, `1.500`
     when followed by exactly three digits) are read as one number.
   - Numbers above the supported range are read digit by digit.
3. Letters: `w` → `v`; accented letters that are not Azerbaijani are reduced to
   their base letter (`é` → `e`); the Azerbaijani letters `ç ə ğ ı ö ş ü` are
   kept.
4. Guard: any character still outside the Azerbaijani alphabet, `,.!?;:`, space
   and hyphen is removed. Repeated spaces and punctuation are collapsed.

The result may be empty (for example a chunk that was only emoji); the caller
skips empty results. Foreign words written in the Azerbaijani alphabet are read
by the letter rules; individual words can be corrected in
`configs/az_lexicon.tsv`, as today.

### `tts.py`

`TTSService(checkpoint=None, inference_path="configs/inference.json")`:

- Loads the inference defaults, resolves the checkpoint (argument, else the one
  named in `inference.json`), builds `Synth` with `default_config(checkpoint)`,
  and computes the reference style once.
- Runs one warm-up synthesis at startup, so the 1.27 s cold call is not paid in
  the first reply.
- `synthesize(text) -> np.ndarray` — `phonemize_texts([text], g2p=synth.g2p)`
  then `synth(...)` with the tuned parameters; returns 24 kHz float32. A lock
  makes it safe to call from one worker thread at a time.

### `pipeline.py`

`Pipeline(mic, segmenter, stt, llm, tts, player, cfg)` takes its parts as
constructor arguments, so tests pass fakes.

States: `LISTENING → THINKING → SPEAKING → LISTENING`.

One turn:

1. `LISTENING`: the main thread reads mic blocks and feeds the segmenter until
   it returns an utterance.
2. `THINKING`: `stt.transcribe`. If the text is empty, go back to `LISTENING`
   without calling the LLM. Otherwise print the transcript.
3. Three workers run for the reply:
   - LLM thread: `llm.stream_reply` → `chunker.feed` → `text_q`; on end,
     `chunker.flush` and a sentinel.
   - TTS thread: `text_q` → `to_spoken` → `tts.synthesize` → `audio_q`.
   - Playback thread: `audio_q` → `player.play`, with 150 ms of silence between
     chunks. The state becomes `SPEAKING` when the first chunk starts playing.
4. When the sentinel has passed through playback, wait `tail_ms` (300), call
   `mic.drain()` and `segmenter.reset()`, and return to `LISTENING`.

The mic stream runs the whole time; blocks arriving outside `LISTENING` are
dropped. STT and TTS never run at the same moment, so they do not compete for
the GPU.

`--text` mode replaces steps 1–2 with a line read from stdin; the rest is
identical. It needs no microphone and is the main debugging tool.

After each turn the pipeline prints the reply text and one latency line:
`stt`, `llm_first_chunk`, `tts_first`, and `first_audio` (from end of speech to
the start of playback), in milliseconds.

### `scripts/talk.py` and `configs/assistant.yml`

```
.\.venv\Scripts\python.exe scripts\talk.py [--config configs\assistant.yml]
    [--text] [--list-devices] [--input-device N] [--output-device N]
```

`configs/assistant.yml`:

```yaml
tts:
  checkpoint: null            # null = the checkpoint named in configs/inference.json
stt:
  model: large-v3-turbo
  device: cuda
  compute_type: float16
  initial_prompt: "Salam, necəsiniz? Bu gün sizə necə kömək edə bilərəm?"
llm:
  model: gemini-2.5-flash
  max_turns: 10
  max_output_tokens: 300
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

## Latency and memory budget

| Stage | Time to first audio | Source |
|---|---|---|
| End-of-speech silence | 0.6 s | config |
| STT | 0.3–0.6 s | estimate |
| Gemini, first sentence | 0.5–1.0 s | estimate |
| TTS, first sentence | 0.45 s | measured |
| **Total** | **about 2–2.5 s** | |

The estimates are checked with the per-turn latency line during the manual
smoke run.

VRAM: TTS 1.2 GB plus Whisper `large-v3-turbo` in float16 at roughly 1.5 GB
(estimate), inside the 8 GB card. Training and the assistant should not run at
the same time.

## Error handling

| Situation | Behaviour |
|---|---|
| `GEMINI_API_KEY` not set | `talk.py` exits at startup with a message naming the variable. |
| Empty or noise-only transcript | Back to `LISTENING`; no LLM call. |
| `LLMError` before any text was produced | Play a fallback phrase synthesized at startup: "Bağışlayın, hazırda cavab verə bilmirəm. Zəhmət olmasa, bir az sonra yenidən cəhd edin." The error is logged; a rate limit is logged as such. |
| `LLMError` mid-stream | Speak the chunks already produced, then return to `LISTENING`. |
| TTS raises on a chunk | Log the chunk and the error, skip it, continue with the next. |
| Whisper cannot use the GPU | Fall back to CPU `int8` with a logged warning. |
| Ctrl+C | Stop the workers, close the audio streams, exit cleanly. |

## Testing

Unit tests with pytest, in the style of the existing `tests/`; no GPU,
microphone or network:

- `test_text_norm.py` — `number_to_words` across ranges; each number form
  (time, decimal, percent, currency, ordinal and other suffixes, thousands
  groups, oversized numbers); markdown, URL and emoji stripping; letter mapping;
  and a property check that `text_to_ipa(normalize_text(to_spoken(x)))` never
  raises for a set of awkward inputs.
- `test_chunker.py` — boundaries across delta splits, decimals not split, short
  sentences merged, long sentences split at commas, `flush`.
- `test_vad.py` — `Segmenter` with scripted probabilities: pre-roll included,
  end after the configured silence, short bursts discarded, maximum length,
  `reset`.
- `test_llm.py` — history append, trim and rollback on failure, and 429 mapped
  to `LLMRateLimited`, with a fake client.
- `test_pipeline.py` — fakes for every part: chunk order preserved from LLM to
  player, empty transcript skips the LLM, fallback phrase on `LLMError`, a
  failing chunk is skipped, mic drained and segmenter reset after speaking.

Manual end-to-end check: `talk.py --text` first (LLM, normalizer, TTS,
playback), then `talk.py` with the microphone; record the latency lines.

## Known limitations

- Whisper's Azerbaijani is one of its weaker languages and can drift into
  Turkish. `language="az"` and the Azerbaijani prompt reduce this. `WhisperSTT`
  is the only module that knows about Whisper, so a fine-tuned model can replace
  it later. STT accuracy is the expected quality bottleneck.
- The Gemini free plan has per-minute and per-day request limits, and Google may
  use free-tier prompts to improve its models, so sensitive data should not be
  spoken to the assistant.
- Half-duplex only: the user cannot interrupt a reply.

## Out of scope

Barge-in, echo cancellation, browser or telephony clients, wake word, tool use
or retrieval, STT fine-tuning, and any change to the TTS model or training code.
