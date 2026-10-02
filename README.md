# StyleTTS2 Azerbaijani fine-tune

Fine-tunes the [StyleTTS2](https://github.com/yl4579/StyleTTS2) LibriTTS checkpoint
(`epochs_2nd_00020.pth`, English) on a ~3.15 h single-speaker Azerbaijani dataset,
on a single 8 GB-VRAM Windows GPU (RTX 5060). Everything below reflects what was
actually built and measured on this machine, not the original plan — the numbers
come from `configs/calibration.json` and `configs/config_ft_az.yml`.

## Quick start: try the voice (CPU is enough)

The trained voice ships with the repo (`models/az_voice/`, about 510 MB, stored with
[Git LFS](https://git-lfs.com)), so nothing needs to be trained and no GPU is needed.
You need Git, Git LFS and Python 3.10.

```powershell
git lfs install
git clone https://github.com/nadir2609/az-finetune.git
cd az-finetune
py -3.10 -m venv .venv                              # Linux/macOS: python3.10 -m venv .venv
.\.venv\Scripts\python.exe scripts\setup_inference.py   # Linux/macOS: .venv/bin/python scripts/setup_inference.py
```

`setup_inference.py` installs CPU PyTorch and the requirements, clones StyleTTS2 at the
pinned commit and patches it (about 5 minutes, about 1.5 GB of downloads). On Linux the
assistant also needs PortAudio (`sudo apt install libportaudio2`).

**Speak your own sentence:**

```powershell
.\.venv\Scripts\python.exe scripts\synthesize.py --text "Salam, bu gün hava çox gözəldir." --out_dir out
```

The result is `out\01.wav` (24 kHz). Repeat `--text "..."` for several sentences
(`02.wav`, `03.wav`, ...). On a CPU a sentence takes roughly as long to generate as it
lasts, after about 20 s of model loading. Write numbers as words (`yüz əlli`, not `150`);
only the assistant below converts them for you.

**Talk with the LLM (Gemini) in this voice:**

1. Create a free key at https://aistudio.google.com/apikey.
2. Put it in a file named `.env` in the repo root, as one line: `GEMINI_API_KEY=<your key>`.
3. Run one of:

```powershell
.\.venv\Scripts\python.exe scripts\talk.py --text     # type your question, hear the answer
.\.venv\Scripts\python.exe scripts\talk.py            # speak into the microphone instead
```

Microphone mode downloads the Whisper speech recognizer on first run (about 1.6 GB) and
on a CPU needs roughly 10 s per turn to recognize speech; it prints
`cuda unavailable ... falling back to CPU int8` once, which is expected. `--text` mode
needs neither. Details and settings are under *Voice assistant* below.

The rest of this README documents how the voice was trained.

## Why English PL-BERT (no fine-tuning of it as a separate step)

StyleTTS2's second stage uses a PL-BERT text encoder to condition prosody. There
is no multilingual PL-BERT checkpoint that covers Azerbaijani, so this project
uses upstream's stock **English** PL-BERT (`StyleTTS2/Utils/PLBERT/`, bundled with
the pinned StyleTTS2 checkout) as the starting point. PL-BERT is not frozen during
training: `train_finetune.py` calls `optimizer.step('bert')` every step with a
low learning rate (`bert_lr: 1.0e-05` in `configs/config_ft_az.yml`), so it
continues adapting to Azerbaijani IPA phoneme sequences as part of the normal
fine-tuning run — there is no separate PL-BERT pretraining step to run.

Known limitation: English PL-BERT never saw the Azerbaijani-only phonemes
(`ɯ ɫ y c œ ɟ ɣ`, ~11 % of tokens) — their embeddings are near-untrained and barely
move at `bert_lr: 1e-5` — and it carries no Azerbaijani word/phrase knowledge, so the
text → prosody path (duration, pitch, style diffusion) is the model's weakest part
(see the `eval_tts.py` results below). If pitch range stays well below the speaker's
after the second run, the next step is an Azerbaijani PL-BERT (continued
pretraining of this one on phonemized Azerbaijani text).

## Layout

| Path | What |
|---|---|
| `metadata.csv`, `audio/` | Original dataset (read-only, never modified). |
| `scripts/az_text.py` | Azerbaijani text normalization + IPA (`g2p="rules"` default, `"espeak"` for runs trained before it), shared by data prep and inference. |
| `scripts/az_g2p.py`, `configs/az_lexicon.tsv` | Rule-based Azerbaijani G2P (letters, ğ, k/l/q/r context, stress) + editable pronunciation overrides. |
| `scripts/prepare_data.py` | `audio/*.mp3` + `metadata.csv` → `data/wavs/*.wav` (24 kHz mono PCM16) + `data/{train,val,OOD_texts}.txt` + `data/prep_report.json`. |
| `scripts/validate_data.py` | Checks the lists StyleTTS2 will actually train on (format, IPA symbols, sample rate/channels, duration). |
| `scripts/check_env.py` | GPU/torch/espeak/monotonic_align/model/disk sanity checks. |
| `scripts/download_models.py` | Downloads the LibriTTS StyleTTS2 checkpoint + pre-caches WavLM. |
| `scripts/smoke_test.py`, `scripts/make_config.py` | Calibrate batch/max_len/precision against real VRAM, then build `configs/config_ft_az.yml` and a time-budgeted epoch schedule. |
| `configs/calibration.json` | Measured fit results from the smoke test (this machine's numbers). |
| `configs/config_ft_az.yml` | The actual training config, generated by `make_config.py`. |
| `scripts/train.ps1` | **The script you run to start/resume training.** |
| `scripts/monitor.py` | Live progress view of the current run's `train.log` (`log_dir` in the config). |
| `scripts/synthesize.py` | Generate WAVs from a checkpoint for listening/evaluation. |
| `models/az_voice/` | The shipped voice: inference weights (Git LFS), their config and the 16 style reference clips. Built from a training checkpoint by `scripts/export_voice.py`. |
| `scripts/setup_inference.py` | Inference-only setup (CPU or GPU, any OS); see *Quick start*. |
| `assistant/`, `scripts/talk.py`, `configs/assistant.yml` | Voice assistant: microphone → Whisper → Gemini → this model's voice (see *Voice assistant*). |
| `scripts/serve.py`, `assistant/server.py`, `assistant/web/` | Web demo: the same assistant as a web page for phones and laptops (see *Web demo*). |
| `scripts/eval_tts.py`, `configs/inference.json` | Duration / pitch-range comparison against the real recordings; `--tune` writes the inference defaults. |
| `patches/styletts2.patch` | Everything changed vs. upstream StyleTTS2 (applied automatically by `setup_env.ps1`). |
| `StyleTTS2/` | Upstream clone (gitignored, rebuilt by `setup_env.ps1`). |
| `runs/az_ft/` | First run (espeak labels, LibriTTS start): checkpoints, logs, TensorBoard, samples (gitignored). |
| `runs/az_ft2/` | Second run (rule-based G2P, starts from `runs/az_ft/epoch_2nd_00010.pth`, SLM adversarial on). |

## Setup / rebuild

`StyleTTS2/` and `.venv/` are gitignored and rebuilt from scratch:

```powershell
.\scripts\setup_env.ps1
```

This creates `.venv` (Python 3.10), installs torch/cu128 + `requirements.txt` +
the `vendor/monotonic_align` extension, clones StyleTTS2 and checks it out to the
pinned commit `5cedc71c333f8d8b8551ca59378bdcc7af4c9529`, applies
`patches/styletts2.patch`, downloads the pretrained checkpoint + WavLM, and runs
`check_env.py`. Safe to re-run any time.

**Patch contents** (`patches/styletts2.patch`, upstream files it changes):
- `Modules/hifigan.py` — decoder activation (gradient) checkpointing, gated by a
  `grad_checkpoint` config flag; NSF sine source and the whole vocoder forward
  forced to run in fp32 even under bf16 autocast.
- `Modules/slmadv.py` — `.float()` before `.numpy()` (numpy has no bf16 dtype).
- `losses.py` — `WavLMLoss` freezes and `.eval()`s the WavLM backbone (never
  optimized, so no param-grad memory is allocated for it).
- `models.py` — `DurationEncoder` LSTM dropout forced to `0.0` (numerically a
  no-op for `num_layers=1`; upstream's non-zero value crashes the process at
  exit on torch 2.11/Windows with cuDNN RNN dropout state).
- `train_finetune.py` — adds `mixed_precision` (bf16 autocast) support,
  `--max_hours`/`--max_steps`/`--stats_out`/`--resume_from` CLI flags, a hard
  per-process VRAM cap (`set_per_process_memory_fraction`, to fail fast with a
  real OOM instead of silently spilling to slow shared RAM under Windows/WDDM),
  NaN-loss step skipping instead of dropping into a debugger, and a config-copy
  fix for when the working config and log-dir config are the same file.

## Data

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
.\.venv\Scripts\python.exe scripts\validate_data.py
```

`prepare_data.py` decodes `audio/*.mp3`, trims silence, converts to 24 kHz mono
PCM16, phonemizes the Azerbaijani text to IPA with the rule-based G2P
(`scripts/az_g2p.py`, see below), rejects anything with a bad
duration or unphonemizable/unknown IPA symbols, and does a seeded 95/5
train/val split. Current `data/prep_report.json`:

- **2432** clips kept out of 2432 total rows, **0 rejected**.
- **2310 train / 122 val**, **~3.15 h** total audio.
- Duration: min 2.0 s, mean 4.67 s, max 10.72 s (p05 3.28 s, p95 6.48 s).
- 0 clipped inputs.

`validate_data.py` re-checks the actual list files StyleTTS2 will read
(`data/train_list.txt`, `data/val_list.txt`, `data/OOD_texts.txt`): format
`<file>.wav|<ipa>|0`, 0 unknown IPA symbols, correct sample rate/channels,
duration in range. Expect `errors: 0`.

### Pronunciation (G2P)

espeak-ng's `az` voice produced systematic errors in the first run's labels: deleted
vowels (`ağac → ˈaːdʒ`), ğ as extra vowels (`sığorta → sɯɯortˈa`), a spurious `ts`
(`bütün → bytsˈyn`), palatal k in back-vowel words (`bank → bˈanç`, `kod → cod`) and
final stress on suffixes that never take it (`bildirməyin`, `gəlmədi`).
`scripts/az_g2p.py` replaces it:

- ğ: `ɣ` between vowels, otherwise it lengthens the previous vowel (`dağ → dˈaː`).
- k and l follow the vowel of their syllable (following vowel in onset, preceding in
  coda): back → `k`/`ɫ`; front → `c` (onset) / `ç` (coda) and `l`.
- q: `x` word-final or before a voiceless consonant, else `ɡ`; r: `ɾ` between vowels.
- Stress: final by default; before the leftmost unstressable suffix — verbal negation
  (`gəlmədi → ɟˈælmædi`), copula `-dır` (`gözəldir → ɟœzˈældir`), question `-mı` in questions.
- **Fix a word's pronunciation** by adding `word<TAB>ipa` to `configs/az_lexicon.tsv`
  (lowercase word form, StyleTTS2 symbols, `ˈ` before the stressed vowel; no `ˈ` =
  unstressed). Changing pronunciations used in training data needs
  `prepare_data.py` + `validate_data.py` + a new training run.

## Calibration (and when to re-run it)

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --start 2/150   # measures what fits in VRAM, writes configs/calibration.json
.\.venv\Scripts\python.exe scripts\make_config.py --pretrained runs\az_ft\epoch_2nd_00010.pth `
    --log_dir runs\az_ft2 --diff_frac 0.1 --joint_frac 0.3        # builds configs/config_ft_az.yml + prints the schedule
```

`smoke_test.py` runs short (≤50-step) capped training probes against the
**longest 200 training utterances** (the worst case for VRAM). It first searches
batch_size/max_len candidates × SLM clip windows (`slmadv min_len/max_len`:
200/300 → 160/240 → 120/200 frames) for the largest setting where the SLM
adversarial ("joint") phase fits **and the log proves the SLM step actually ran**
(non-zero `DiscLM`/`GenLM` losses); if none does, it falls back to the best
diffusion-phase setting with SLM disabled. `--start 2/150` skips the larger
candidates, which OOM on this 8 GB card in every phase (see `raw` in older
calibrations). `make_config.py` copies `StyleTTS2/Configs/config_ft.yml`, overrides
it with the calibrated settings + this project's data paths, and picks the largest
epoch count that fits the time budget (default 7.0 h).

Why SLM needed fixing: upstream's `SLMAdversarialLoss` returns `None` unless at
least 2 samples reach it, and `batch_percentage: 0.5` × `batch_size: 2` keeps one —
so the first run never trained with it. `make_config.py` now keeps ≥ 2 samples
(`batch_percentage: 1.0` at batch 2) and uses the calibrated window (upstream's
400/500 only admits clips longer than 5 s, and most of this dataset is shorter).

**Current calibration** (`configs/calibration.json`, RTX 5060, 8 GB):
- `batch_size: 2`, `max_len: 150`, `mixed_precision: bf16`, **`slm_ok: true`,
  `slm_window: [200, 300]`**.
- s/step: 2.37 (acoustic) / 2.62 (+diffusion) / **3.91 (+SLM adversarial)**.
- Peak **allocated** 5.55 / 6.12 / **6.08 GB**; peak **reserved** 6.10 / 6.48 /
  6.57 GB against `cap_gb: 6.573` (`free_gb_at_launch: 6.866`). Reserved is the
  allocator's cache (what `monitor.py`'s `peak_gb` shows); the smoke test runs with
  the default allocator, while `train.ps1` sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which kept reserved ~0.2 GB
  lower in earlier measurements. **It fits, with ~0.45 GB of allocated headroom** —
  but in the SLM phase *reserved* memory touched the cap (6.572 of 6.573 GB), so an
  OOM late in the run is possible. Auto-resume restarts from the last epoch's
  checkpoint, but each SLM epoch is ~1.25 h of the 7.5 h cap. Close other GPU apps
  before launching (the precheck enforces this); if an OOM happens, re-run
  calibration, which will drop to a smaller SLM window.
- WavLM is frozen; decoder uses gradient checkpointing; bf16 autocast except the
  HiFi-GAN decoder/vocoder and F0/STFT losses (fp32).

**Current schedule** (`configs/config_ft_az.yml`, second run): starts from the
first run's `runs/az_ft/epoch_2nd_00010.pth` weights (fresh epoch counter and
optimizer), **6 epochs** — epoch 1 acoustic-only (relearn the new G2P labels),
epoch 2 + style diffusion, epochs 3–6 + SLM adversarial — **projected ≈6.84 h**,
written to `runs/az_ft2`. The launcher's **hard 7.5 h wall-clock cap** applies
regardless of the projection.

**Re-run calibration** (`smoke_test.py --start 2/150`, then `make_config.py` with the
full flags shown above — `--pretrained` and `--log_dir` are required, so a config can
never silently start from the wrong weights or point at another run's directory) if VRAM
availability on this machine changes — a driver update, a different GPU, or
simply a different baseline of other GPU apps running. `free_gb_at_launch` in
`calibration.json` is also what `train.ps1`'s launch-time precheck compares
against (see below); a stale value there only warns if the file is *missing*,
not if it's out of date.

## Start training (user)

```powershell
.\scripts\train.ps1
```

This is a launcher around `StyleTTS2/train_finetune.py -p configs\config_ft_az.yml`.
It:
- Takes the run directory from `log_dir` in `configs\config_ft_az.yml`
  (currently `runs\az_ft2`). A fresh launch **refuses a directory that already
  holds checkpoints** — after a crash it would otherwise auto-resume from another
  run's optimizer state. Use `-Resume` to continue that run, or build a config with
  a new `make_config.py --log_dir`.
- Sets `PHONEMIZER_ESPEAK_LIBRARY`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`,
  `PYTHONIOENCODING=utf-8`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (reduces allocator fragmentation over a long run).
- **Precheck**: reads `configs/calibration.json`'s `free_gb_at_launch` and
  compares it (minus a 0.2 GB slack) against current free VRAM from
  `nvidia-smi`. If there isn't enough free VRAM, it refuses to start with a
  message telling you to close browsers/games/Steam/other GPU apps. Pass
  `-Force` to downgrade this to a warning and proceed anyway.
- Runs training with a **7.5 h hard wall-clock cap** (`-MaxHours`, default
  7.5), and **auto-resumes from the newest checkpoint up to 3 times**
  (`-MaxRestarts`) if the process crashes, as long as time remains in the
  budget — the budget is measured from the launcher's own start time and never
  exceeded across restarts.
- Logs to `<log_dir>\train.log` / `<log_dir>\console.log`.

Flags:
- `-Resume` — continue from the newest checkpoint in `<log_dir>` on this
  invocation (not just after a crash).
- `-MaxHours <n>` — total wall-clock cap across the run and any auto-resumes.
- `-Force` — skip/downgrade the free-VRAM precheck.
- `-MaxRestarts <n>` — auto-resume attempts on crash (default 3).
- `-DryRun` — print the command(s) that would run; **never starts training**.

## Monitoring

In a second terminal:

```powershell
.\.venv\Scripts\python.exe scripts\monitor.py --follow 60
```

or TensorBoard:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs\az_ft2\tensorboard
```

`monitor.py` parses the current run's `train.log` (`log_dir` in
`configs\config_ft_az.yml`; `--log` overrides) and prints epoch/step, sec/step,
peak VRAM, elapsed/cap hours + ETA, recent losses, per-epoch validation mel
loss, NaN-skip count, and a live `nvidia-smi` line. `--follow 60` refreshes
every 60 s; omit it for a single snapshot.

**What healthy numbers look like:**
- Mel `Loss` starts around **~0.5** and trends down over the run.
- Validation loss (printed once per epoch) decreases, then flattens —
  flattening late in the schedule is expected and fine.
- From epoch 3 on, `DiscLM Loss` / `GenLM Loss` are **non-zero** (SLM adversarial
  training is running); in the first run they were always 0.
- `NaN skips` should stay at or near **0**. A handful over a 7 h run is not
  alarming (the training loop already skips the optimizer step on a NaN
  `g_loss` and continues); frequent NaNs are a problem (see Troubleshooting).
- `peak_gb` should stay under the calibrated cap (`cap_gb` in
  `configs/calibration.json`, currently 6.573 GB) — it will hover near it by
  design (headroom is tight), not accelerate upward over time.

**Checkpoints:** `<log_dir>\epoch_2nd_%05d.pth` is written once per epoch
(`save_freq: 1`), roughly **2 GB each** — 6 epochs ≈ 12 GB on top of the first
run's ~22 GB in `runs\az_ft`; keep an eye on disk space. If the 7.5 h wall-clock
cap is hit mid-epoch, `epoch_2nd_final.pth` is written and the run stops cleanly.

## Resume

```powershell
.\scripts\train.ps1 -Resume
```

Picks the newest `epoch_2nd_*.pth` in the config's `log_dir` by file modification
time and resumes model + optimizer + epoch state from it. This also happens
automatically on a crash (no flag needed) as long as restart/time budget
remain.

## Listening / evaluation

```powershell
.\.venv\Scripts\python.exe scripts\synthesize.py --checkpoint runs\az_ft2\epoch_2nd_00005.pth
.\.venv\Scripts\python.exe scripts\eval_tts.py --checkpoint runs\az_ft2\epoch_2nd_00005.pth          # measure
.\.venv\Scripts\python.exe scripts\eval_tts.py --checkpoint runs\az_ft2\epoch_2nd_00005.pth --tune   # re-tune defaults
```

`synthesize.py` writes WAVs to `<checkpoint's run dir>\samples\<checkpoint-stem>\`
(or `--out_dir`): 3 default Azerbaijani sentences (`01.wav`–`03.wav`) plus the first
validation item's text (`04.wav`), with its **ground-truth recording copied
alongside as `gt_<name>.wav`** for direct A/B comparison against the real voice.
It uses the config saved next to the checkpoint and phonemizes with the G2P that
checkpoint was trained with (`g2p` in that config; runs without it, like
`runs\az_ft`, used espeak).

Defaults come from `configs/inference.json` (written by `eval_tts.py --tune`);
flags override them:
- `--text "..."` (repeatable) — override the default sentences.
- `--ref <wav>` (repeatable; several are averaged) — style reference.
- `--alpha` / `--beta` — blend of predicted vs. reference style (timbre / prosody).
  **Lower values stay closer to the reference**; higher lets the diffusion-predicted
  style dominate.
- `--embedding_scale` — diffusion classifier-free guidance scale.
- `--speed` — divides predicted durations (>1 speaks faster).
- `--steps` — diffusion sampling steps.

`eval_tts.py` synthesizes 12 validation items (fixed seeds) and compares each with
its recording: **`dur_ratio`** (synth/real duration, >1 = too slow) and
**`f0_ratio`** (synth/real 5–95 % pitch spread in semitones, <1 = flatter than the
speaker); `score = |log dur_ratio| + |log f0_ratio|`, 0 = matches the speaker.
`--tune` picks the reference strategy (single clip / mean of 16 random / 16 most
expressive train clips), then alpha × beta × embedding_scale, then speed.

Tuned for the first run's epoch 10 (current `configs/inference.json`): single
reference clip, alpha 0.3, **beta 0.5, embedding_scale 1.0, speed 1.2** —
`dur_ratio` 1.17 → **1.02**, `f0_ratio` 0.80 → **0.89** vs. the old defaults
(0.3 / 0.7 / 1.5 / 1.0). Averaging several reference clips made speech *flatter*,
and a high beta (trusting the diffusion-predicted style) flattened it most — the
text → style path is the weak part of this model. **Re-run `--tune` on the second
run's final checkpoint** before relying on these defaults for it.

Compare a fine-tuned checkpoint's `04.wav` directly against its `gt_*.wav` —
same text and speaker, so it's the most direct fidelity check available.

## Voice assistant

Speak Azerbaijani into the microphone, get a spoken answer in this model's voice:
microphone → Silero VAD → faster-whisper (`large-v3-turbo`) → Gemini → StyleTTS2 → speakers.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-assistant.txt   # once
# put GEMINI_API_KEY=<your key> in a .env file in the repo root (gitignored); key: https://aistudio.google.com/apikey
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
  one is synthesized (about 0.4 s) and played while the next is still arriving. The TTS
  speed is measured as it runs, and a long sentence early in a reply is cut at a comma
  where it would not be ready before the audio ahead of it ends (on a CPU this avoids a
  silence of several seconds after a short first sentence). Each turn
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
| `tts.short_chars` | Chunks shorter than this (default 40 characters) take their style fully from the reference clips. Under about 2 s of speech the text-predicted style drifts to a different-sounding voice; `0` turns the rule off. |
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
- **Delay** from the end of speech to the first audio, on the RTX 5060: 0.6 s to detect the end of
  speech, 0.3–0.5 s of speech recognition for a 3 s question, Gemini's first sentence (0.7–1.2 s),
  and 0.25 s for the voice: about 2.3 s. Visitors who stop speaking at the same moment are
  recognized one after another, so the last of five waits up to about 1 s longer. Check it with the
  server running and a recorded question:
  `.\.venv\Scripts\python.exe scripts\measure_latency.py --wav question.wav` (add `--sessions 5`).
- **Gemini plan:** the free plan's daily request limit is used up quickly by several visitors; use a
  paid key for a demo. The key and the access code stay on this machine and are never sent to the page.

Do not run the server during a training run: both need the GPU.

## Troubleshooting

- **OOM (`torch.OutOfMemoryError`)**: close other GPU-using apps (browser,
  Steam, etc.), then re-run `scripts\smoke_test.py --start 2/150` to re-measure
  what actually fits (it drops to a smaller SLM window if needed) and regenerate
  `configs/calibration.json` + `configs/config_ft_az.yml`. Allocated headroom at
  the current calibrated setting is only ~0.45 GB (6.12 GB allocated vs. the
  6.573 GB cap, worst-case phase), and reserved memory (the allocator's cache —
  what `monitor.py`'s `peak_gb` shows) sits even closer to the cap, so a busier
  desktop than at calibration time can tip it over.
- **NaN losses / frequent `NaN skips`**: the training loop already skips the
  optimizer step and continues on a NaN `g_loss`, but if it happens often,
  lower `batch_size` isn't the usual first lever here (it's already at the
  calibrated floor) — check `<log_dir>\train.log` for which loss term is
  NaN-ing and consider re-running calibration; a corrupted/too-quiet audio
  clip slipping past `validate_data.py` is also worth checking.
- **Slow step time / step time creeping up**: check `nvidia-smi` for other
  processes using the GPU. On Windows/WDDM, if the process is pushed past
  available VRAM it silently spills to shared system RAM instead of erroring,
  which is ~10x slower rather than an outright crash — `train_finetune.py`'s
  hard per-process VRAM cap (from the patch) is meant to convert that into a
  fast, clean OOM instead, but a very busy desktop can still degrade step
  time before hitting the cap. `monitor.py`'s `sec_per_step` should stay near
  the calibrated 2.4 / 2.6 / 3.9 s/step (acoustic / diffusion / SLM phases); a
  sustained jump beyond that is the signal.
- **`train.ps1` refuses to start (precheck failure)**: it's telling you free
  VRAM is below what calibration measured was needed
  (`free_gb_at_launch` in `configs/calibration.json`, currently 6.866 GB,
  minus 0.2 GB slack). Close other GPU apps and retry, or pass `-Force` to
  proceed anyway at your own risk (more likely to OOM mid-run).
- **`train.ps1` refuses to start ("already holds checkpoints")**: the config's
  `log_dir` already has checkpoints. Use `-Resume` only if the config was built for
  *that same run*; resuming another run's checkpoint (e.g. the espeak-trained
  `runs\az_ft`) with the new labels would continue the wrong run. For a new run,
  rebuild the config with `make_config.py --pretrained <start weights> --log_dir <new dir>`.
- **A word is mispronounced**: add it to `configs/az_lexicon.tsv` (see
  *Pronunciation (G2P)*). Inference picks it up immediately; for training data it
  needs `prepare_data.py` and a new run.
