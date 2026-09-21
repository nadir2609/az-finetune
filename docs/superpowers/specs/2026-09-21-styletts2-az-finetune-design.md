# StyleTTS2 Azerbaijani Fine-Tune — Design & Plan

## Context
Goal: a natural, human-like Azerbaijani single-speaker TTS voice (good intonation/prosody) by fine-tuning the
pretrained StyleTTS2 LibriTTS model on the user's data, on an RTX 5060 (8 GB, Blackwell sm_120), Windows 11.
Hard constraints: **I never start the real training** (user launches it); training must finish in **≤ 7–8 h**
(enforced by a wall-clock stop, not just an estimate). User approved: single speaker, native Windows,
and a short capped smoke test (≤ 50 steps) for calibration.

Path classification (superpowers:brainstorming): **architectural** (new project). After approval: write spec to
`docs/superpowers/specs/2026-09-21-styletts2-az-finetune-design.md`, then superpowers:writing-plans →
superpowers:subagent-driven-development / executing-plans, TDD for data code, verification-before-completion at the end.

## Findings (workspace + system)
- `audio/` 2,432 MP3 (44.1 kHz mono, 128 kbps, ~3–7 s, ≈3.3 h total); `metadata.csv` = `line_number,filename,text`, no missing files.
- Text charset: Azerbaijani Latin + `,.!?` only, no digits → no number normalization needed. Uppercase `İ`/`I` present → needs
  Azerbaijani-aware lowercasing (`İ→i`, `I→ı`; Python `.lower()` gets this wrong).
- eSpeak NG 1.52 with `az` voice installed; IPA is correct when fed UTF-8 (CLI args mangle `ə` → always phonemize via
  Python `phonemizer` library / files, never shell args). All produced symbols (æ ɯ ɟ ç ɾ ɫ ʃ ʒ ɡ ˈ) exist in StyleTTS2's
  178-token `text_utils.py` set; unknown chars are *silently dropped* there → our validator must assert 0 unknowns.
- Python 3.10.8 available (use for venv; 3.14 unsupported by torch stack). No torch yet. ffmpeg, git present. 16 GB RAM, 689 GB free.
- GPU needs PyTorch **cu128** wheels (sm_120). Driver CUDA 13.2 OK.
- PL-BERT: multilingual PL-BERT covers 16 languages, **not az/tr** → keep the English PL-BERT bundled in the LibriTTS
  checkpoint and fine-tune it jointly on az IPA (bert lr 1e-5). Training a new PL-BERT = days → out of budget.

## Target file structure
```
style-tts2-tuning/
├─ audio/                    # original MP3s (read-only, untouched)
├─ metadata.csv              # original (untouched)
├─ .venv/                    # Python 3.10 venv (torch cu128)
├─ StyleTTS2/                # upstream clone (pinned commit) + small Windows/time-limit patches
│  ├─ Models/LibriTTS/       # epochs_2nd_00020.pth + config.yml (HF: yl4579/StyleTTS2-LibriTTS)
│  └─ Utils/{ASR,JDC,PLBERT} # shipped with repo
├─ data/
│  ├─ wavs/                  # 24 kHz mono 16-bit WAV, trimmed silence
│  ├─ train_list.txt         # wav|ipa|0   (~95%)
│  ├─ val_list.txt           # wav|ipa|0   (~5%, ~120 clips, fixed seed)
│  ├─ OOD_texts.txt          # ipa|0 for SLM adversarial phase
│  └─ prep_report.json       # counts, durations, unknown-symbol check, rejects
├─ configs/config_ft_az.yml  # generated/calibrated config
├─ scripts/
│  ├─ check_env.py           # GPU/CUDA/sm_120, torch, espeak DLL, disk, VRAM free
│  ├─ az_text.py             # normalization + phonemization (library module)
│  ├─ prepare_data.py        # mp3→wav, text→IPA, split, lists, report
│  ├─ validate_data.py       # asserts: files exist, 0 unknown symbols, durations sane
│  ├─ smoke_test.py          # ≤50 capped steps per phase → VRAM + s/step → calibration.json
│  ├─ make_config.py         # builds config_ft_az.yml from calibration so total ≤ 7.5 h
│  ├─ train.ps1              # USER runs this (sets env, launches, logs to runs/)
│  ├─ monitor.py             # tails log: epoch/step, losses, it/s, ETA, VRAM, time left
│  └─ synthesize.py          # checkpoint → WAVs for test sentences (diffusion style, alpha/beta)
├─ tests/                    # pytest: text normalization, phonemization, list format
├─ runs/az_ft/               # log_dir: checkpoints, train.log, tensorboard
└─ docs/superpowers/specs/   # design spec
```

## Design

### 1. Environment setup & verification
- `py -3.10 -m venv .venv`; install `torch torchaudio --index-url https://download.pytorch.org/whl/cu128`, then upstream
  requirements (munch, pydub, librosa, nltk, einops, einops-exts, transformers, accelerate, phonemizer, tensorboard,
  soundfile, matplotlib, pyyaml, click) + `numba`, `pytest`.
- `monotonic_align`: upstream needs a Cython build (MSVC). Try the pip git install; if it fails, drop in a small
  numba implementation of `maximum_path`/`mask_from_lens` as a local `monotonic_align` package (same API).
- Env var `PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll`.
- Clone StyleTTS2 (pin commit), download LibriTTS 2nd-stage checkpoint + config.
- `check_env.py` prints/asserts: CUDA available, device capability (12,0), a tiny matmul on GPU, espeak `az` output
  contains `ə`/`æ`, free VRAM, disk.

### 2. Data pipeline (`az_text.py`, `prepare_data.py`, `validate_data.py`)
- Text: Unicode NFC → az lowercase (`İ→i`, `I→ı`) → collapse whitespace → keep `,.!?` → phonemizer espeak `az`
  (`preserve_punctuation=True, with_stress=True`) → tokenize like upstream (word-level join) → filter to StyleTTS2 symbols.
- Audio: ffmpeg MP3 → 24 kHz mono PCM16; trim leading/trailing silence (conservative threshold, keep ~100 ms pad);
  reject clips < 0.8 s or > 15 s (logged in report).
- Split 95/5 seeded; speaker id `0` for all (keep `multispeaker: true` to match LibriTTS weights).
- OOD texts: phonemized sentences from the dataset's texts (shuffled, val texts included) — no external corpus needed.
- `validate_data.py` fails loudly on: missing wav, unknown symbols > 0, empty IPA, bad format.
- TDD: tests for lowercasing edge cases (`İ`, `I`), punctuation, phonemizer output containing expected az phones, list-line format.

### 3. Fine-tuning configuration (`make_config.py` → `configs/config_ft_az.yml`)
Based on upstream `Configs/config_ft.yml`, with changes:
- `pretrained_model: StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`, `load_only_params: true`, `second_stage_load_pretrained: true`.
- Data paths → `data/…`, `log_dir: runs/az_ft`, `root_path: data/wavs`.
- 8 GB-aware: `batch_size` 2–4 and `max_len` ~100–200 frames chosen from smoke-test VRAM (target peak ≤ ~7.2 GB).
- Phases: stage A acoustic+duration fine-tune → `diff_epoch` (style diffusion, key for expressive intonation) →
  `joint_epoch` (SLM adversarial w/ WavLM). If the smoke test shows SLM phase OOMs even at batch 2 / small max_len,
  set `joint_epoch > epochs` (skip) — quality still good; documented in the report.
- `epochs`, `diff_epoch`, `joint_epoch` computed from measured s/step per phase so projected total ≈ 6.5–7 h.
- `save_freq: 1` (checkpoint each epoch) so an early stop never loses much.
- Keep fp32 (upstream fp16 is NaN-prone); lr as upstream (1e-4, bert 1e-5).

### 4. Execution & monitoring
- Minimal patches to `StyleTTS2/train_finetune.py`: (a) `--max_hours` (default 7.5): checked every step; when exceeded
  → save `epoch_2nd_final.pth` and exit cleanly; (b) Windows: `num_workers` configurable (default 0–2), `__main__` guard;
  (c) `--max_steps` for the smoke test; (d) log it/s + peak VRAM to train.log.
- `train.ps1` (user runs): activates venv, sets env vars, launches with `--max_hours 7.5`, tees to `runs/az_ft/train.log`,
  supports resume from latest checkpoint.
- `monitor.py`: live view of epoch/phase, losses, it/s, VRAM, elapsed/remaining, ETA vs 7.5 h cap. Plus TensorBoard
  (`tensorboard --logdir runs/az_ft`).
- `synthesize.py`: loads a checkpoint, synthesizes a fixed set of Azerbaijani test sentences (statement, question,
  long comma-rich sentence) with diffusion style sampling (`alpha≈0.3, beta≈0.7, diffusion_steps 5–10,
  embedding_scale 1–2`), optionally with a reference clip → `runs/az_ft/samples/`.

## Execution order (after approval)
1. `git init` + `.gitignore` (audio, data/wavs, .venv, runs, Models); write spec doc, commit.
2. Env setup + `check_env.py` (verify output).
3. `az_text.py` + tests (TDD) → `prepare_data.py` → run it → `validate_data.py` (must pass, 0 unknown symbols).
4. Clone/patch StyleTTS2, download checkpoint.
5. `smoke_test.py`: ≤50 steps per phase (stage A, diffusion, SLM) with capped steps, writes `calibration.json`; checkpoints discarded.
6. `make_config.py` → config; `train.ps1`, `monitor.py`, `synthesize.py` (verify synthesize on the *pretrained* model
   with az text so the inference path is proven before training).
7. superpowers:requesting-code-review + verification-before-completion; hand off: exact command to start training.
   **Real training is NOT started by me.**

## Verification
- `pytest tests/` green.
- `check_env.py` all checks pass on the 5060.
- `validate_data.py`: all lines valid, 0 unknown symbols, duration histogram in report.
- Smoke test completes all enabled phases without OOM; peak VRAM and s/step recorded.
- `make_config.py` projection ≤ 7.5 h, and the `--max_hours` guard verified by a smoke run with `--max_hours 0.01` (saves + exits).
- `synthesize.py` produces a WAV from the pretrained checkpoint (pipeline sanity; accented/garbled az expected before fine-tune).
