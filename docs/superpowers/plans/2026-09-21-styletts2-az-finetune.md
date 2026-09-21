# StyleTTS2 Azerbaijani Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible, verified pipeline (env → data → calibrated config → launch/monitor/synthesize scripts) to fine-tune the StyleTTS2 LibriTTS checkpoint on 2,432 Azerbaijani single-speaker clips within a hard 7.5 h wall-clock cap on an RTX 5060 8 GB (Windows 11).

**Architecture:** Upstream StyleTTS2 is cloned (pinned) and minimally patched (time cap, step cap, stats, resume, NaN-skip, VRAM cap, configurable workers); the patch is tracked in `patches/`. Our own code lives in `scripts/` (small single-purpose modules + CLIs) with pytest tests in `tests/`. A smoke test measures real s/step + VRAM per training phase; `make_config.py` turns that into an epoch schedule that fits the budget.

**Tech Stack:** Python 3.10 venv, PyTorch 2.11.0+cu128 (installed, sm_120 verified), phonemizer + eSpeak NG 1.52 (`az`), numba, soundfile/librosa, ffmpeg, PowerShell 5.1 launcher.

**Spec:** `docs/superpowers/specs/2026-09-21-styletts2-az-finetune-design.md`

## Global Constraints

- **NEVER start the real training run.** Only the capped smoke runs (≤ 50 steps each) in Task 7 and the time-cap test in Task 6 may execute training code. The user launches `scripts/train.ps1` manually.
- Real training hard cap: `--max_hours 7.5`; schedule target from `make_config.py`: ≤ 7.0 h projected.
- Originals `audio/` and `metadata.csv` are read-only.
- Python: always `.venv\Scripts\python.exe` (3.10). Never system 3.14.
- Phonemize only via the Python `phonemizer` library (never espeak CLI args — they mangle `ə`). eSpeak voice: `az`.
- Azerbaijani lowercasing: `İ→i`, `I→ı` before `.lower()`.
- Audio target: 24 kHz, mono, PCM16 WAV. Keep clips 0.8 s ≤ dur ≤ 15 s.
- List line format: `<file>.wav|<ipa>|0` (speaker id 0, `multispeaker: true` kept to match LibriTTS weights).
- Validation split 5 %, seed 1234. 0 unknown symbols allowed.
- All files written by our code use UTF-8 explicitly (`encoding="utf-8"`); Windows default is cp1252.
- Env vars for anything touching StyleTTS2: `PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch ≥2.6 defaults to weights-only loads; checkpoints are trusted upstream files), `PYTHONIOENCODING=utf-8`.
- Upstream pin: StyleTTS2 commit `5cedc71c333f8d8b8551ca59378bdcc7af4c9529`.
- Commit after each task (`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer). `StyleTTS2/`, `.venv/`, `data/wavs/`, `runs/`, `audio/` are gitignored.

## File map

| File | Responsibility |
|---|---|
| `requirements.txt` | non-torch deps |
| `vendor/monotonic_align/` | numba port of resemble-ai `monotonic_align` (no MSVC on this machine) |
| `scripts/setup_env.ps1` | idempotent env rebuild: venv, torch cu128, deps, shim, clone+pin+patch, models |
| `scripts/download_models.py` | LibriTTS checkpoint + config, WavLM cache |
| `scripts/check_env.py` | environment verification |
| `scripts/az_text.py` | normalize + phonemize + tokenize (shared by data prep and inference) |
| `scripts/prepare_data.py` | mp3→wav, IPA, split, lists, report |
| `scripts/validate_data.py` | hard checks on produced lists |
| `patches/styletts2.patch` | our diff to upstream `train_finetune.py` |
| `scripts/smoke_test.py` | capped runs per phase → `configs/calibration.json` |
| `scripts/make_config.py` | calibrated `configs/config_ft_az.yml` |
| `scripts/synthesize.py` | checkpoint → WAV samples |
| `scripts/train.ps1` | user-run launcher (resume support) |
| `scripts/monitor.py` | log parser / live progress |
| `tests/` | pytest suites; `tests/conftest.py` puts `scripts/` on `sys.path` |
| `README.md` | how to run everything |

---

### Task 1: Dependencies + monotonic_align numba shim

**Files:**
- Create: `requirements.txt`, `vendor/monotonic_align/pyproject.toml`, `vendor/monotonic_align/monotonic_align/__init__.py`, `vendor/monotonic_align/monotonic_align/core.py`
- Test: `tests/test_monotonic_align.py`, `tests/conftest.py`

**Interfaces:**
- Produces: importable `monotonic_align.maximum_path(value, mask)`, `monotonic_align.mask_from_lens(similarity, symbol_lens, mel_lens)`, `monotonic_align.core.maximum_path_c(paths:int32[b,x,y], values:float32[b,x,y], t_xs:int32[b], t_ys:int32[b])` (in-place) — exactly what `StyleTTS2/utils.py` imports.

- [ ] **Step 1: requirements + install**

`requirements.txt`:
```
SoundFile
munch
pydub
pyyaml
librosa
nltk
matplotlib
accelerate
transformers
einops
einops-exts
tqdm
pandas
phonemizer
tensorboard
click
numba
huggingface_hub
pytest
```
Run: `.venv\Scripts\python.exe -m pip install -r requirements.txt`
Expected: success. Then `.venv\Scripts\python.exe -c "import torch, numpy, librosa, phonemizer, numba; print(torch.__version__)"` → `2.11.0+cu128`.

- [ ] **Step 2: Write failing test** `tests/conftest.py`:
```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
```
`tests/test_monotonic_align.py`:
```python
import numpy as np
import torch
from monotonic_align import maximum_path, mask_from_lens


def brute_force_best(value):
    """Exhaustive monotonic alignment: each mel frame y maps to one text x,
    x starts at 0, ends at X-1, and advances by 0 or 1 per frame."""
    X, Y = value.shape
    best, best_path = -1e18, None

    def rec(y, x, score, path):
        nonlocal best, best_path
        if y == Y:
            if x == X - 1 and score > best:
                best, best_path = score, list(path)
            return
        for nx in (x, x + 1):
            if nx < X and (Y - y - 1) >= (X - 1 - nx):
                path.append(nx)
                rec(y + 1, nx, score + value[nx, y], path)
                path.pop()

    rec(1, 0, value[0, 0], [0])
    return best, best_path


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        X, Y = rng.integers(2, 5), rng.integers(5, 9)
        v = rng.normal(size=(X, Y)).astype(np.float32)
        path = maximum_path(torch.from_numpy(v)[None], torch.ones(1, X, Y))[0].numpy()
        _, bp = brute_force_best(v)
        expected = np.zeros((X, Y), dtype=np.float32)
        for y, x in enumerate(bp):
            expected[x, y] = 1
        np.testing.assert_array_equal(path, expected)


def test_mask_limits_path_and_shape():
    sim = torch.randn(2, 6, 20)
    mask = mask_from_lens(sim, torch.tensor([6, 3]), torch.tensor([20, 9]))
    path = maximum_path(sim, mask)
    assert path.shape == sim.shape
    assert path[1, 3:, :].sum() == 0 and path[1, :, 9:].sum() == 0
    assert torch.all(path[0].sum(0) == 1)           # each frame -> exactly one token
    assert torch.all(path[1, :, :9].sum(0) == 1)
```
Run: `.venv\Scripts\python.exe -m pytest tests/test_monotonic_align.py -v` → FAIL (`ModuleNotFoundError: monotonic_align`).

- [ ] **Step 3: Implement shim**

`vendor/monotonic_align/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "monotonic_align"
version = "0.1.0+numba"
dependencies = ["numba", "numpy", "torch"]

[tool.setuptools]
packages = ["monotonic_align"]
```
`vendor/monotonic_align/monotonic_align/core.py` (line-for-line port of resemble-ai `core.pyx`):
```python
"""Numba port of resemble-ai/monotonic_align core.pyx (no C compiler needed)."""
import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def _maximum_path_each(path, value, t_x, t_y, max_neg_val):
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


@numba.njit(nogil=True, cache=True, parallel=True)
def _maximum_path_batch(paths, values, t_xs, t_ys, max_neg_val):
    for i in numba.prange(values.shape[0]):
        _maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)


def maximum_path_c(paths, values, t_xs, t_ys, max_neg_val=-1e9):
    _maximum_path_batch(paths, values, np.asarray(t_xs, dtype=np.int32),
                        np.asarray(t_ys, dtype=np.int32), np.float32(max_neg_val))
```
`vendor/monotonic_align/monotonic_align/__init__.py`:
```python
import numpy as np
import torch

from monotonic_align.core import maximum_path_c


def mask_from_len(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    index = torch.arange(max_len).to(lens).view(1, -1)
    return index < lens.unsqueeze(1)


def mask_from_lens(similarity, symbol_lens, mel_lens):
    _, S, T = similarity.size()
    mask_ST = mask_from_len(symbol_lens, S).unsqueeze(2) * mask_from_len(mel_lens, T).unsqueeze(1)
    return mask_ST.to(similarity)


def maximum_path(value, mask=None):
    if mask is None:
        mask = torch.ones_like(value)
    device, dtype = value.device, value.dtype
    value = np.ascontiguousarray((value * mask).data.cpu().numpy().astype(np.float32))
    mask = mask.data.cpu().numpy()
    path = np.zeros_like(value).astype(np.int32)
    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
```
Run: `.venv\Scripts\python.exe -m pip install -e vendor/monotonic_align`

- [ ] **Step 4: Run tests** → `pytest tests/test_monotonic_align.py -v` PASS (2 passed).
- [ ] **Step 5: Commit** `git add requirements.txt vendor tests && git commit -m "feat: deps and numba monotonic_align shim"`

---

### Task 2: Upstream pin, patch scaffold, model download, env check

**Files:**
- Create: `scripts/download_models.py`, `scripts/check_env.py`, `scripts/setup_env.ps1`

**Interfaces:**
- Produces: `StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`, `StyleTTS2/Models/LibriTTS/config.yml`; `check_env.py` exit code 0 on success.

- [ ] **Step 1: Pin upstream** — `git -C StyleTTS2 checkout 5cedc71c333f8d8b8551ca59378bdcc7af4c9529` (already cloned; HEAD is that commit — verify with `git -C StyleTTS2 rev-parse HEAD`).

- [ ] **Step 2: `scripts/download_models.py`**
```python
"""Download the LibriTTS StyleTTS2 checkpoint and pre-cache WavLM (used by SLM losses)."""
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"


def main():
    for f in ["Models/LibriTTS/epochs_2nd_00020.pth", "Models/LibriTTS/config.yml"]:
        p = hf_hub_download("yl4579/StyleTTS2-LibriTTS", f, local_dir=str(STTS))
        print("ok", p)
    from transformers import WavLMModel
    WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    print("ok WavLM cached")


if __name__ == "__main__":
    main()
```
Run it. Expected: two `ok` lines with paths under `StyleTTS2\Models\LibriTTS\`, then `ok WavLM cached`.

- [ ] **Step 3: `scripts/check_env.py`**
```python
"""Verify GPU/torch/espeak/disk for StyleTTS2 fine-tuning. Exit 1 on any failure."""
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail))
    except Exception as e:  # report every failure, don't stop at first
        results.append((False, name, repr(e)))


def gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), f"torch lacks kernels for sm_{cap[0]}{cap[1]}"
    x = torch.randn(1024, 1024, device="cuda")
    assert torch.isfinite((x @ x).sum()).item()
    free, total = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch {torch.__version__} free {free/2**30:.2f}/{total/2**30:.2f} GB"


def espeak():
    from az_text import phonemize_texts
    ipa = phonemize_texts(["Xidmət keyfiyyəti"])[0]
    assert "æ" in ipa, ipa
    return ipa


def shim():
    from monotonic_align import maximum_path  # noqa: F401
    return "monotonic_align importable"


def models():
    p = ROOT / "StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth"
    assert p.exists(), f"missing {p} (run scripts/download_models.py)"
    return f"{p.stat().st_size/2**20:.0f} MB"


def disk():
    free = shutil.disk_usage(ROOT).free / 2**30
    assert free > 30, f"only {free:.0f} GB free"
    return f"{free:.0f} GB free"


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    for n, f in [("gpu", gpu), ("espeak-az", espeak), ("monotonic_align", shim), ("models", models), ("disk", disk)]:
        check(n, f)
    for ok, n, d in results:
        print(f"[{'OK' if ok else 'FAIL'}] {n}: {d}")
    sys.exit(0 if all(r[0] for r in results) else 1)
```
(Depends on `az_text.phonemize_texts` from Task 3 — run the full check at the end of Task 3; now run and expect only `espeak-az` to FAIL with ImportError.)

- [ ] **Step 4: `scripts/setup_env.ps1`** (idempotent reproduction of Tasks 1–2 + patch)
```powershell
# Rebuilds the environment from scratch. Safe to re-run.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { py -3.10 -m venv .venv }
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
& $Py -m pip install -r requirements.txt
& $Py -m pip install -e vendor/monotonic_align
if (-not (Test-Path "StyleTTS2")) { git clone https://github.com/yl4579/StyleTTS2.git }
git -C StyleTTS2 checkout -- .
git -C StyleTTS2 checkout 5cedc71c333f8d8b8551ca59378bdcc7af4c9529
git -C StyleTTS2 apply (Join-Path $Root "patches\styletts2.patch")
& $Py scripts\download_models.py
& $Py scripts\check_env.py
```
(`patches/styletts2.patch` is produced in Task 6; don't run this script until then.)

- [ ] **Step 5: Commit** `git add scripts/download_models.py scripts/check_env.py scripts/setup_env.ps1 && git commit -m "feat: model download, env check, setup script"`

---

### Task 3: Azerbaijani text front-end (`az_text.py`)

**Files:**
- Create: `scripts/az_text.py`
- Test: `tests/test_az_text.py`

**Interfaces:**
- Produces:
  - `normalize_text(text: str) -> str`
  - `tokenize_ipa(ipa: str) -> str` — punctuation split into own tokens, single spaces (mimics upstream `' '.join(word_tokenize(ps))`)
  - `phonemize_texts(texts: list[str]) -> list[str]` — normalize → espeak az → tokenize_ipa
  - `STYLETTS_SYMBOLS: set[str]` and `unknown_symbols(ipa: str) -> set[str]`

- [ ] **Step 1: Failing tests** `tests/test_az_text.py`:
```python
from az_text import normalize_text, tokenize_ipa, phonemize_texts, unknown_symbols


def test_az_lowercase_dotted_and_dotless_i():
    assert normalize_text("İLHAM İLƏ") == "ilham ilə"
    assert normalize_text("IŞIQ") == "ışıq"


def test_whitespace_and_quotes():
    assert normalize_text('  "Salam",   dünya!  ') == "salam, dünya!"


def test_tokenize_ipa_splits_punctuation():
    assert tokenize_ipa("salˈam,  dynjˈa!") == "salˈam , dynjˈa !"
    assert tokenize_ipa("bu. o?") == "bu . o ?"


def test_phonemize_keeps_schwa_and_punctuation():
    out = phonemize_texts(["Xidmət keyfiyyəti, yaxşıdır."])[0]
    assert "æ" in out            # espeak az renders ə as æ
    assert " , " in out and out.endswith(" .")
    assert unknown_symbols(out) == set()


def test_phonemize_batch_order():
    a, b = phonemize_texts(["bir", "iki"])
    assert a != b and a.strip() and b.strip()


def test_unknown_symbols_detects():
    assert unknown_symbols("abc☃") == {"☃"}
```
Run: `pytest tests/test_az_text.py -v` → FAIL (ModuleNotFoundError az_text).

- [ ] **Step 2: Implement** `scripts/az_text.py`:
```python
"""Azerbaijani text normalization + IPA phonemization for StyleTTS2 (shared by data prep and inference)."""
import os
import re
import unicodedata

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

# Mirrors StyleTTS2/text_utils.py (pinned commit) -- the 178-token vocabulary.
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
STYLETTS_SYMBOLS = set([_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa))

_PUNCT = ",.!?;:"
_backend = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = re.sub(r"[\"“”«»]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_ipa(ipa: str) -> str:
    return " ".join(re.findall(rf"[^\s{re.escape(_PUNCT)}]+|[{re.escape(_PUNCT)}]", ipa))


def _get_backend():
    global _backend
    if _backend is None:
        from phonemizer.backend import EspeakBackend
        _backend = EspeakBackend(language="az", preserve_punctuation=True, with_stress=True)
    return _backend


def phonemize_texts(texts):
    norm = [normalize_text(t) for t in texts]
    ipa = _get_backend().phonemize(norm, strip=True, njobs=1)
    return [tokenize_ipa(p) for p in ipa]


def unknown_symbols(ipa: str) -> set:
    return {c for c in ipa if c not in STYLETTS_SYMBOLS}
```
- [ ] **Step 3: Run tests** → PASS (6 passed). If `test_phonemize_keeps_schwa_and_punctuation` fails on a symbol, print the output and fix normalization — do NOT add symbols to the vocabulary (the checkpoint embedding has exactly 178 rows).
- [ ] **Step 4: Full env check** `.venv\Scripts\python.exe scripts/check_env.py` → all 5 `[OK]`, exit 0.
- [ ] **Step 5: Commit** `git add scripts/az_text.py tests/test_az_text.py && git commit -m "feat: Azerbaijani text normalization and phonemization"`

---

### Task 4: Data preparation (`prepare_data.py`)

**Files:**
- Create: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `az_text.phonemize_texts`, `az_text.unknown_symbols`
- Produces: `data/wavs/<id>.wav`, `data/train_list.txt`, `data/val_list.txt`, `data/OOD_texts.txt`, `data/prep_report.json`; pure helpers `trim_silence(wave: np.ndarray, sr: int, top_db=40, pad_s=0.1) -> np.ndarray`, `split_items(items: list, val_frac=0.05, seed=1234) -> (train, val)`, `list_line(wav_name: str, ipa: str) -> str`.

- [ ] **Step 1: Failing tests** `tests/test_prepare_data.py`:
```python
import numpy as np
from prepare_data import trim_silence, split_items, list_line


def test_trim_silence_keeps_padding():
    sr = 24000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    wave = np.concatenate([np.zeros(sr), tone, np.zeros(sr)])
    out = trim_silence(wave, sr, top_db=40, pad_s=0.1)
    assert abs(len(out) - int(1.2 * sr)) < int(0.15 * sr)  # librosa trims on 2048-sample frames


def test_split_is_deterministic_and_disjoint():
    items = list(range(1000))
    tr1, va1 = split_items(items, 0.05, 1234)
    tr2, va2 = split_items(items, 0.05, 1234)
    assert va1 == va2 and len(va1) == 50 and not set(tr1) & set(va1)
    assert sorted(tr1 + va1) == items


def test_list_line_format():
    assert list_line("000001.wav", "salˈam .") == "000001.wav|salˈam .|0"
```
Run → FAIL (ModuleNotFoundError).

- [ ] **Step 2: Implement** `scripts/prepare_data.py`:
```python
"""metadata.csv + audio/*.mp3 -> data/wavs (24k mono PCM16) + StyleTTS2 lists + report."""
import argparse
import csv
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from az_text import phonemize_texts, unknown_symbols

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
MIN_S, MAX_S = 0.8, 15.0


def trim_silence(wave, sr, top_db=40, pad_s=0.1):
    _, (start, end) = librosa.effects.trim(wave, top_db=top_db)
    pad = int(pad_s * sr)
    return wave[max(0, start - pad):min(len(wave), end + pad)]


def split_items(items, val_frac=0.05, seed=1234):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(len(items) * val_frac))
    val_idx = set(idx[:n_val])
    train = [it for i, it in enumerate(items) if i not in val_idx]
    val = [it for i, it in enumerate(items) if i in val_idx]
    return train, val


def list_line(wav_name, ipa):
    return f"{wav_name}|{ipa}|0"


def decode_mp3(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def convert(mp3, out_wav):
    wave = trim_silence(decode_mp3(mp3), SR)
    peak = float(np.abs(wave).max()) if len(wave) else 0.0
    if peak > 0.99:  # avoid PCM16 clipping only; otherwise keep natural loudness
        wave = wave * (0.95 / peak)
    sf.write(out_wav, wave, SR, subtype="PCM_16")
    return len(wave) / SR, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(ROOT / "metadata.csv"))
    ap.add_argument("--audio_dir", default=str(ROOT / "audio"))
    ap.add_argument("--out_dir", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    with open(args.metadata, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def job(r):
        name = Path(r["filename"]).stem + ".wav"
        dur, peak = convert(Path(args.audio_dir) / r["filename"], wav_dir / name)
        return name, dur, peak

    with ThreadPoolExecutor(args.workers) as ex:
        converted = list(ex.map(job, rows))

    ipas = phonemize_texts([r["text"] for r in rows])

    items, rejects = [], []
    for r, (name, dur, peak), ipa in zip(rows, converted, ipas):
        unk = unknown_symbols(ipa)
        if not (MIN_S <= dur <= MAX_S):
            rejects.append({"file": name, "reason": f"duration {dur:.2f}s"})
        elif not ipa.strip() or unk:
            rejects.append({"file": name, "reason": f"ipa empty/unknown {sorted(unk)}"})
        else:
            items.append((name, ipa, dur))

    train, val = split_items(items)
    for fname, subset in [("train_list.txt", train), ("val_list.txt", val)]:
        (out / fname).write_text("\n".join(list_line(n, i) for n, i, _ in subset) + "\n", encoding="utf-8")
    (out / "OOD_texts.txt").write_text("\n".join(f"{i}|0" for _, i, _ in items) + "\n", encoding="utf-8")

    durs = np.array([d for _, _, d in items])
    report = {
        "total_rows": len(rows), "kept": len(items), "train": len(train), "val": len(val),
        "rejected": rejects,
        "hours": round(float(durs.sum()) / 3600, 3),
        "duration_s": {"min": float(durs.min()), "mean": float(durs.mean()), "max": float(durs.max()),
                       "p05": float(np.percentile(durs, 5)), "p95": float(np.percentile(durs, 95))},
        "clipped_inputs": sum(1 for _, _, p in converted if p > 0.99),
    }
    (out / "prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rejected"}, indent=2), f"\nrejected: {len(rejects)}")


if __name__ == "__main__":
    main()
```
- [ ] **Step 3: Run tests** → PASS (3 passed).
- [ ] **Step 4: Run on real data** `.venv\Scripts\python.exe scripts/prepare_data.py`
Expected: `kept` ≈ 2432, `rejected` small (0–20), `hours` ≈ 3.0–3.4, `train` ≈ 2310, `val` ≈ 122. Spot-check: `soundfile.info("data/wavs/000001.wav")` → 24000 Hz, 1 ch, PCM_16; open `data/train_list.txt` (UTF-8) and eyeball 3 lines.
- [ ] **Step 5: Commit** `git add scripts/prepare_data.py tests/test_prepare_data.py data/*.txt data/prep_report.json && git commit -m "feat: data preparation pipeline + generated lists"`

---

### Task 5: Data validation (`validate_data.py`)

**Files:**
- Create: `scripts/validate_data.py`
- Test: `tests/test_validate_data.py`

**Interfaces:**
- Consumes: `az_text.unknown_symbols`
- Produces: `validate_lines(lines: list[str], wav_dir: Path) -> list[str]` (list of error strings); CLI exits 1 if any error.

- [ ] **Step 1: Failing tests** `tests/test_validate_data.py`:
```python
import numpy as np
import soundfile as sf
from validate_data import validate_lines


def _wav(tmp_path, name, seconds, sr=24000):
    sf.write(tmp_path / name, np.zeros(int(seconds * sr)), sr, subtype="PCM_16")


def test_good_line_passes(tmp_path):
    _wav(tmp_path, "a.wav", 2.0)
    assert validate_lines(["a.wav|salˈam .|0"], tmp_path) == []


def test_detects_problems(tmp_path):
    _wav(tmp_path, "short.wav", 0.3)
    _wav(tmp_path, "sr.wav", 2.0, sr=22050)
    errs = validate_lines([
        "missing.wav|salam|0",
        "short.wav|salam|0",
        "sr.wav|salam|0",
        "a.wav||0",
        "a.wav|sal☃m|0",
        "bad line",
    ], tmp_path)
    joined = "\n".join(errs)
    for needle in ["missing", "duration", "sample rate", "empty", "unknown", "format"]:
        assert needle in joined, needle
```
Run → FAIL.

- [ ] **Step 2: Implement** `scripts/validate_data.py`:
```python
"""Fail loudly if StyleTTS2 lists are not trainable."""
import sys
from pathlib import Path

import soundfile as sf

from az_text import unknown_symbols

ROOT = Path(__file__).resolve().parents[1]


def validate_lines(lines, wav_dir):
    errors = []
    for n, line in enumerate(lines, 1):
        parts = line.rstrip("\n").split("|")
        if len(parts) != 3:
            errors.append(f"line {n}: bad format ({line[:60]!r})")
            continue
        name, ipa, spk = parts
        if not ipa.strip():
            errors.append(f"line {n}: empty IPA")
        unk = unknown_symbols(ipa)
        if unk:
            errors.append(f"line {n}: unknown symbols {sorted(unk)}")
        if spk != "0":
            errors.append(f"line {n}: speaker id {spk!r} != '0'")
        p = Path(wav_dir) / name
        if not p.exists():
            errors.append(f"line {n}: missing wav {name}")
            continue
        info = sf.info(str(p))
        if info.samplerate != 24000 or info.channels != 1:
            errors.append(f"line {n}: sample rate/channels {info.samplerate}/{info.channels}")
        if not (0.8 <= info.duration <= 15.0):
            errors.append(f"line {n}: duration {info.duration:.2f}s")
    return errors


def main():
    data = ROOT / "data"
    total = 0
    all_errors = []
    for fname in ["train_list.txt", "val_list.txt"]:
        lines = [l for l in (data / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        total += len(lines)
        all_errors += [f"{fname}: {e}" for e in validate_lines(lines, data / "wavs")]
    ood = [l for l in (data / "OOD_texts.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    for n, l in enumerate(ood, 1):
        if unknown_symbols(l.split("|")[0]):
            all_errors.append(f"OOD_texts.txt line {n}: unknown symbols")
    print(f"checked {total} list lines + {len(ood)} OOD lines; errors: {len(all_errors)}")
    for e in all_errors[:50]:
        print("  ", e)
    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
```
- [ ] **Step 3: Run tests** → PASS (2 passed).
- [ ] **Step 4: Run on real data** `.venv\Scripts\python.exe scripts/validate_data.py` → `errors: 0`, exit 0. If not 0 → fix `prepare_data.py`/`az_text.py` (superpowers:systematic-debugging), re-run Task 4 step 4.
- [ ] **Step 5: Commit** `git add scripts/validate_data.py tests/test_validate_data.py && git commit -m "feat: dataset validator"`

---

### Task 6: Patch `train_finetune.py` (time cap, step cap, stats, resume, NaN-skip, VRAM cap, workers)

**Files:**
- Modify: `StyleTTS2/train_finetune.py` (pinned commit; line numbers refer to it)
- Create: `patches/styletts2.patch`, `tests/fixtures/` not needed

**Interfaces:**
- Produces CLI: `python train_finetune.py -p CONFIG [--max_hours FLOAT=7.5] [--max_steps INT=0] [--stats_out PATH] [--resume_from PATH]`
- Log line (every `log_interval` steps) consumed by `monitor.py`:
  `PERF epoch=%d step=%d/%d sec_per_step=%.3f peak_gb=%.2f elapsed_h=%.3f max_hours=%.2f`
- Time-cap checkpoint: `<log_dir>/epoch_2nd_final.pth` (same dict format as upstream).
- Stats JSON (`--stats_out`): `{"steps", "sec_per_step_mean", "sec_per_step_median", "peak_reserved_gb", "peak_allocated_gb", "stop_reason"}`.
- Config key `num_workers` (default 2).

- [ ] **Step 1: Edits** (exact replacements in `StyleTTS2/train_finetune.py`):

a) imports — after `import warnings` (line 14) add `import json`.

b) lines 50–53 →
```python
@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
@click.option('--max_hours', default=7.5, type=float, help='wall-clock cap; saves epoch_2nd_final.pth and exits')
@click.option('--max_steps', default=0, type=int, help='stop after N train steps (smoke tests); 0 = off')
@click.option('--stats_out', default='', type=str, help='write timing/VRAM stats json here on exit')
@click.option('--resume_from', default='', type=str, help='checkpoint to resume (params + optimizer + epoch)')
def main(config_path, max_hours, max_steps, stats_out, resume_from):
    config = yaml.safe_load(open(config_path, encoding='utf-8'))
```
c) after `device = 'cuda'` (line 91) add:
```python
    # Hard VRAM cap: on Windows/WDDM an over-allocation silently spills to shared RAM (10x slower)
    # instead of raising OOM. Cap to what is actually free so we get a real OOM instead.
    free_b, total_b = torch.cuda.mem_get_info()
    vram_frac = max(0.5, min(0.95, (free_b - 300 * 1024 ** 2) / total_b))
    torch.cuda.set_per_process_memory_fraction(vram_frac)
    logger.info('VRAM cap: %.2f GB of %.2f GB' % (vram_frac * total_b / 1024 ** 3, total_b / 1024 ** 3))
```
d) line 98 `num_workers=2,` → `num_workers=config.get('num_workers', 2),`

e) after the `if load_pretrained:` block (after line 212) add:
```python
    if resume_from:
        model, optimizer, last_epoch, _ = load_checkpoint(model, optimizer, resume_from, load_only_params=False)
        start_epoch = last_epoch + 1
        logger.info('Resumed from %s; continuing at epoch %d' % (resume_from, start_epoch + 1))
```
f) before `for epoch in range(start_epoch, epochs):` (line 243) add:
```python
    run_start = time.time()
    step_times = []
    global_step = 0
    stop_reason = None
```
g) first line inside `for i, batch in enumerate(train_dataloader):` add `step_t0 = time.time()`.

h) lines 458–460 (the IPython `set_trace`) →
```python
            if torch.isnan(g_loss):
                logger.warning('NaN loss at epoch %d step %d - skipping update' % (epoch + 1, i + 1))
                optimizer.zero_grad()
                continue
```
(`g_loss.backward()` stays directly above it, unchanged.)

i) after `iters = iters + 1` (line 542) add:
```python
            torch.cuda.synchronize()
            step_times.append(time.time() - step_t0)
            global_step += 1
            elapsed_h = (time.time() - run_start) / 3600
            if (i + 1) % log_interval == 0:
                logger.info('PERF epoch=%d step=%d/%d sec_per_step=%.3f peak_gb=%.2f elapsed_h=%.3f max_hours=%.2f'
                            % (epoch + 1, i + 1, len(train_dataloader), float(np.mean(step_times[-log_interval:])),
                               torch.cuda.max_memory_reserved() / 1024 ** 3, elapsed_h, max_hours))
            if max_steps and global_step >= max_steps:
                stop_reason = 'max_steps'
                break
            if elapsed_h >= max_hours:
                stop_reason = 'max_hours'
                break
```
j) immediately after the train `for i, batch ...` loop ends (before `loss_test = 0`, line 565) add, at epoch-loop indentation:
```python
        if stop_reason:
            break
```
k) before `print('Epochs:', epoch + 1)` (line 676) add `iters_test = max(iters_test, 1)`.

l) at the end of `main()` (after the epoch loop, before `if __name__`), at function indentation:
```python
    if stop_reason == 'max_hours':
        state = {
            'net': {key: model[key].state_dict() for key in model},
            'optimizer': optimizer.state_dict(),
            'iters': iters,
            'val_loss': float('nan'),
            'epoch': epoch - 1,  # resume restarts the interrupted epoch
        }
        torch.save(state, osp.join(log_dir, 'epoch_2nd_final.pth'))
        logger.info('Time budget of %.2f h reached - saved epoch_2nd_final.pth and stopped cleanly.' % max_hours)
    if stats_out:
        warm = step_times[3:] if len(step_times) > 6 else step_times
        with open(stats_out, 'w', encoding='utf-8') as f:
            json.dump({'steps': global_step,
                       'sec_per_step_mean': float(np.mean(warm)) if warm else None,
                       'sec_per_step_median': float(np.median(warm)) if warm else None,
                       'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1024 ** 3,
                       'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1024 ** 3,
                       'stop_reason': stop_reason}, f)
```
- [ ] **Step 2: Save patch** `git -C StyleTTS2 diff > patches/styletts2.patch`; verify it applies cleanly: `git -C StyleTTS2 stash; git -C StyleTTS2 apply --check ..\patches\styletts2.patch; git -C StyleTTS2 stash pop`.
- [ ] **Step 3: Syntax check** `.venv\Scripts\python.exe -m py_compile StyleTTS2/train_finetune.py` → no output.
- [ ] **Step 4: Commit** `git add patches/styletts2.patch && git commit -m "feat: StyleTTS2 patch - time cap, stats, resume, NaN skip, VRAM cap"`

(Runtime verification of this patch happens in Task 7 — it needs `make_config.build_config`.)

---

### Task 7: Config builder + smoke test calibration

**Files:**
- Create: `scripts/make_config.py`, `scripts/smoke_test.py`
- Test: `tests/test_make_config.py`

**Interfaces:**
- `make_config.build_config(batch_size: int, max_len: int, epochs: int, diff_epoch: int, joint_epoch: int, log_dir: str, num_workers: int = 2) -> dict` (upstream `Configs/config_ft.yml` + overrides, absolute paths)
- `make_config.plan_schedule(n_train: int, batch_size: int, sec_per_step: dict, val_sec_per_epoch: float, budget_h: float = 7.0, slm_ok: bool = True, diff_frac: float = 0.2, joint_frac: float = 0.6) -> dict` → `{"epochs","diff_epoch","joint_epoch","projected_h","steps_per_epoch"}`; `sec_per_step` keys `"a"`, `"diff"`, `"joint"`.
- `configs/calibration.json` (from smoke test): `{"batch_size","max_len","slm_ok","sec_per_step":{"a","diff","joint"},"peak_gb":{...},"n_train","n_val"}`
- CLI `make_config.py [--budget_h 7.0]` writes `configs/config_ft_az.yml` and prints the schedule.

- [ ] **Step 1: Failing tests** `tests/test_make_config.py`:
```python
from make_config import plan_schedule, build_config


def test_schedule_fits_budget_and_orders_phases():
    s = plan_schedule(n_train=2310, batch_size=4, sec_per_step={"a": 0.6, "diff": 0.8, "joint": 1.6},
                      val_sec_per_epoch=30, budget_h=7.0)
    assert s["projected_h"] <= 7.0
    assert 1 <= s["diff_epoch"] < s["joint_epoch"] < s["epochs"]
    bigger = plan_schedule(2310, 4, {"a": 0.6, "diff": 0.8, "joint": 1.6}, 30, budget_h=8.0)
    assert bigger["epochs"] >= s["epochs"]


def test_schedule_without_slm_skips_joint():
    s = plan_schedule(2310, 2, {"a": 0.5, "diff": 0.7, "joint": None}, 30, budget_h=7.0, slm_ok=False)
    assert s["joint_epoch"] > s["epochs"] and s["projected_h"] <= 7.0


def test_build_config_overrides():
    c = build_config(batch_size=3, max_len=200, epochs=12, diff_epoch=2, joint_epoch=7, log_dir="X")
    assert c["batch_size"] == 3 and c["max_len"] == 200 and c["epochs"] == 12
    assert c["loss_params"]["diff_epoch"] == 2 and c["loss_params"]["joint_epoch"] == 7
    assert c["save_freq"] == 1 and c["model_params"]["multispeaker"] is True
    assert c["data_params"]["train_data"].endswith("train_list.txt")
    assert c["pretrained_model"].endswith("epochs_2nd_00020.pth")
```
Run → FAIL.

- [ ] **Step 2: Implement** `scripts/make_config.py`:
```python
"""Build configs/config_ft_az.yml from upstream config_ft.yml + smoke-test calibration."""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
DATA = ROOT / "data"
LOAD_OVERHEAD_H = 0.1  # model/WavLM load + first-batch warmup


def build_config(batch_size, max_len, epochs, diff_epoch, joint_epoch, log_dir, num_workers=2):
    with open(STTS / "Configs" / "config_ft.yml", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c.update({
        "log_dir": str(log_dir), "save_freq": 1, "log_interval": 10, "epochs": epochs,
        "batch_size": batch_size, "max_len": max_len, "num_workers": num_workers,
        "pretrained_model": str(STTS / "Models/LibriTTS/epochs_2nd_00020.pth"),
        "second_stage_load_pretrained": True, "load_only_params": True,
    })
    c["data_params"].update({
        "train_data": str(DATA / "train_list.txt"), "val_data": str(DATA / "val_list.txt"),
        "root_path": str(DATA / "wavs"), "OOD_data": str(DATA / "OOD_texts.txt"), "min_length": 50,
    })
    c["loss_params"]["diff_epoch"] = diff_epoch
    c["loss_params"]["joint_epoch"] = joint_epoch
    return c


def plan_schedule(n_train, batch_size, sec_per_step, val_sec_per_epoch, budget_h=7.0,
                  slm_ok=True, diff_frac=0.2, joint_frac=0.6):
    steps = n_train // batch_size
    best = None
    for E in range(3, 301):
        d = max(1, round(diff_frac * E))
        j = max(d + 1, round(joint_frac * E)) if slm_ok else E + 1
        n_a, n_d, n_j = d, min(j, E) - d, max(0, E - j)
        sec = steps * (n_a * sec_per_step["a"] + n_d * sec_per_step["diff"]
                       + (n_j * sec_per_step["joint"] if n_j else 0)) + E * val_sec_per_epoch
        h = sec / 3600 + LOAD_OVERHEAD_H
        if h > budget_h:
            break
        best = {"epochs": E, "diff_epoch": d, "joint_epoch": j, "projected_h": round(h, 2), "steps_per_epoch": steps}
    if best is None:
        raise ValueError("even 3 epochs exceed the budget - reduce data or batch size")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_h", type=float, default=7.0)
    ap.add_argument("--calibration", default=str(ROOT / "configs/calibration.json"))
    ap.add_argument("--out", default=str(ROOT / "configs/config_ft_az.yml"))
    args = ap.parse_args()
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    val_sec = (cal["n_val"] / cal["batch_size"]) * cal["sec_per_step"]["a"] * 0.5
    s = plan_schedule(cal["n_train"], cal["batch_size"], cal["sec_per_step"], val_sec,
                      budget_h=args.budget_h, slm_ok=cal["slm_ok"])
    c = build_config(cal["batch_size"], cal["max_len"], s["epochs"], s["diff_epoch"], s["joint_epoch"],
                     log_dir=str(ROOT / "runs/az_ft"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    print(json.dumps(s, indent=2), "\nwritten", args.out)


if __name__ == "__main__":
    main()
```
Run tests → PASS (3 passed).

- [ ] **Step 3: Implement** `scripts/smoke_test.py`:
```python
"""Capped (<=50 step) runs per training phase to measure VRAM + s/step. Never a real training run."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from make_config import build_config

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
PY = str(ROOT / ".venv/Scripts/python.exe")
SMOKE = ROOT / "runs/smoke"
CANDIDATES = [(4, 300), (4, 200), (3, 200), (2, 200), (2, 150)]  # (batch_size, max_len), best first
STEPS = 30
ENV = dict(os.environ, PHONEMIZER_ESPEAK_LIBRARY=r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONIOENCODING="utf-8")


def run_phase(bs, max_len, phase, max_hours=1.0, max_steps=STEPS):
    """phase: 'a' (acoustic only), 'diff' (+style diffusion), 'joint' (+SLM adversarial)."""
    diff_epoch, joint_epoch = {"a": (99, 99), "diff": (0, 99), "joint": (0, 0)}[phase]
    log_dir = SMOKE / f"bs{bs}_len{max_len}_{phase}"
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True)
    cfg = build_config(bs, max_len, epochs=1, diff_epoch=diff_epoch, joint_epoch=joint_epoch,
                       log_dir=str(log_dir), num_workers=0)
    cfg_path = log_dir / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    stats = log_dir / "stats.json"
    p = subprocess.run([PY, "train_finetune.py", "-p", str(cfg_path), "--max_steps", str(max_steps),
                        "--max_hours", str(max_hours), "--stats_out", str(stats)],
                       cwd=STTS, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (log_dir / "stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    if p.returncode != 0 or not stats.exists():
        oom = "out of memory" in (p.stderr + p.stdout).lower()
        return {"ok": False, "oom": oom, "tail": p.stderr[-2000:]}
    return {"ok": True, **json.loads(stats.read_text(encoding="utf-8"))}


def count(name):
    return sum(1 for l in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if l.strip())


def main():
    results = {}
    chosen = None
    for bs, ml in CANDIDATES:  # 1) find the largest setting where the heaviest phase (joint) fits
        r = run_phase(bs, ml, "joint")
        results[f"{bs}/{ml}/joint"] = r
        print(bs, ml, "joint", {k: r.get(k) for k in ("ok", "oom", "sec_per_step_median", "peak_reserved_gb")})
        if r["ok"]:
            chosen, slm_ok = (bs, ml), True
            break
        if not r["oom"]:
            sys.exit(f"non-OOM failure - see runs/smoke/bs{bs}_len{ml}_joint/stdout.txt\n{r['tail']}")
    if chosen is None:  # 2) SLM phase never fits: pick the best setting for the diffusion phase
        slm_ok = False
        for bs, ml in CANDIDATES:
            r = run_phase(bs, ml, "diff")
            results[f"{bs}/{ml}/diff"] = r
            print(bs, ml, "diff", {k: r.get(k) for k in ("ok", "oom", "sec_per_step_median", "peak_reserved_gb")})
            if r["ok"]:
                chosen = (bs, ml)
                break
            if not r["oom"]:
                sys.exit(f"non-OOM failure:\n{r['tail']}")
    if chosen is None:
        sys.exit("nothing fits in VRAM - close other GPU apps and retry")
    bs, ml = chosen
    sps, peak = {}, {}
    for phase in ("a", "diff", "joint"):
        key = f"{bs}/{ml}/{phase}"
        if phase == "joint" and not slm_ok:
            sps[phase], peak[phase] = None, None
            continue
        r = results.get(key) or run_phase(bs, ml, phase)
        results[key] = r
        if not r["ok"]:
            sys.exit(f"phase {phase} failed at chosen setting:\n{r['tail']}")
        sps[phase], peak[phase] = r["sec_per_step_median"], r["peak_reserved_gb"]
    cal = {"batch_size": bs, "max_len": ml, "slm_ok": slm_ok, "sec_per_step": sps, "peak_gb": peak,
           "n_train": count("train_list.txt"), "n_val": count("val_list.txt"), "raw": results}
    out = ROOT / "configs/calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in cal.items() if k != "raw"}, indent=2))


if __name__ == "__main__":
    main()
```
- [ ] **Step 4: Run smoke test** (allowed: capped, ≤ 50 steps per run; user approved). Close nothing of the user's; just run:
`.venv\Scripts\python.exe scripts/smoke_test.py`
Expected: prints per-candidate lines, then calibration JSON with `sec_per_step` for a/diff/(joint) and `peak_gb` < VRAM cap. On a non-OOM failure → superpowers:systematic-debugging on `runs/smoke/*/stdout.txt` (likely suspects: torch.load weights_only, transformers API drift, Windows paths), fix, add the fix to `patches/styletts2.patch` if it's upstream code, re-run.
Also confirm in a joint-phase `train.log` that `DiscLM Loss`/`GenLM Loss` are non-zero in at least one logged line (SLM actually exercised); if always 0 at 30 steps, re-run that phase once with `max_steps=50`.

- [ ] **Step 5: Verify time cap** —
`.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'scripts'); import smoke_test as s, json; print(s.run_phase(<bs>, <ml>, 'a', max_hours=0.01, max_steps=0))"`
(fill `<bs>/<ml>` from calibration). Expected: `ok: True`, `stop_reason: 'max_hours'`, and `runs/smoke/bs<bs>_len<ml>_a/epoch_2nd_final.pth` exists; `train.log` contains `Time budget of 0.01 h reached`.

- [ ] **Step 6: Generate config** `.venv\Scripts\python.exe scripts/make_config.py` → prints schedule with `projected_h` ≤ 7.0; `configs/config_ft_az.yml` exists. Delete smoke artifacts: `Remove-Item -Recurse -Force runs\smoke`.
- [ ] **Step 7: Commit** `git add scripts/make_config.py scripts/smoke_test.py tests/test_make_config.py configs && git commit -m "feat: smoke-test calibration and budgeted config"`

---

### Task 8: Inference script (`synthesize.py`)

**Files:**
- Create: `scripts/synthesize.py`

**Interfaces:**
- Consumes: `az_text.phonemize_texts`; StyleTTS2 `models.build_model/load_ASR_models/load_F0_models`, `Utils.PLBERT.util.load_plbert`, `text_utils.TextCleaner`, `Modules.diffusion.sampler`.
- CLI: `synthesize.py --checkpoint PATH [--config PATH] [--ref WAV] [--text "..."]... [--out_dir DIR] [--alpha 0.3] [--beta 0.7] [--steps 10] [--embedding_scale 1.5]`
  Defaults: config = `runs/az_ft/config_ft_az.yml` if present (it carries the estimated `sigma_data`) else `StyleTTS2/Models/LibriTTS/config.yml`; ref = first line of `data/val_list.txt`; out_dir = `runs/az_ft/samples/<checkpoint-stem>`.
  Default texts also include the first val item's text; its ground-truth wav is copied as `gt_<name>.wav` for A/B listening.

- [ ] **Step 1: Implement**
```python
"""Synthesize Azerbaijani test sentences from a StyleTTS2 checkpoint (adapted from Demo/Inference_LibriTTS.ipynb)."""
import argparse
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
sys.path.insert(0, str(STTS))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import yaml  # noqa: E402

from az_text import phonemize_texts  # noqa: E402

DEFAULT_TEXTS = [
    "Salam! Bu gün hava çox gözəldir, gəlin parka gedək.",
    "Siz sabah bizim ofisə gələ biləcəksinizmi?",
    "Müraciətiniz qeydə alındı, tezliklə əməkdaşlarımız sizinlə əlaqə saxlayacaq və bütün suallarınızı cavablandıracaqlar.",
]
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def preprocess(wave):
    mel = to_mel(torch.from_numpy(wave).float())
    return (torch.log(1e-5 + mel.unsqueeze(0)) + 4) / 4


class Synth:
    def __init__(self, config_path, checkpoint, device="cuda"):
        cwd = os.getcwd()
        os.chdir(STTS)  # upstream config paths (Utils/...) are relative to the repo
        try:
            from models import build_model, load_ASR_models, load_F0_models
            from utils import recursive_munch
            from Utils.PLBERT.util import load_plbert
            from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
            from text_utils import TextCleaner
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
            pitch_extractor = load_F0_models(config["F0_path"])
            plbert = load_plbert(config["PLBERT_dir"])
            self.params = recursive_munch(config["model_params"])
            self.model = build_model(self.params, text_aligner, pitch_extractor, plbert)
        finally:
            os.chdir(cwd)
        state = torch.load(checkpoint, map_location="cpu")["net"]
        for key in self.model:
            if key in state:
                sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state[key].items())
                self.model[key].load_state_dict(sd, strict=False)
        self.device = device
        for key in self.model:
            self.model[key].eval().to(device)
        self.sampler = DiffusionSampler(self.model.diffusion.diffusion, sampler=ADPM2Sampler(),
                                        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                                        clamp=False)
        self.cleaner = TextCleaner()

    @torch.no_grad()
    def compute_style(self, path):
        wave, _ = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        mel = preprocess(audio).to(self.device)
        return torch.cat([self.model.style_encoder(mel.unsqueeze(1)),
                          self.model.predictor_encoder(mel.unsqueeze(1))], dim=1)

    @torch.no_grad()
    def __call__(self, ipa, ref_s, alpha=0.3, beta=0.7, steps=10, embedding_scale=1.5):
        m, dev = self.model, self.device
        tokens = self.cleaner(ipa)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(dev).unsqueeze(0)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(dev)
        text_mask = length_to_mask(input_lengths).to(dev)
        t_en = m.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = m.bert(tokens, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(dev), embedding=bert_dur,
                              embedding_scale=embedding_scale, features=ref_s, num_steps=steps).squeeze(1)
        s, ref = s_pred[:, 128:], s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c = 0
        for i in range(aln.size(0)):
            aln[i, c:c + int(pred_dur[i].data)] = 1
            c += int(pred_dur[i].data)
        aln = aln.unsqueeze(0).to(dev)
        en = d.transpose(-1, -2) @ aln
        asr = t_en @ aln
        if self.params.decoder.type == "hifigan":
            en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=-1)
            asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=-1)
        F0_pred, N_pred = m.predictor.F0Ntrain(en, s)
        out = m.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--text", action="append", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--embedding_scale", type=float, default=1.5)
    a = ap.parse_args()

    run_cfg = ROOT / "runs/az_ft/config_ft_az.yml"
    config = a.config or str(run_cfg if run_cfg.exists() else STTS / "Models/LibriTTS/config.yml")
    val_first = (ROOT / "data/val_list.txt").read_text(encoding="utf-8").splitlines()[0].split("|")
    ref = a.ref or str(ROOT / "data/wavs" / val_first[0])
    out_dir = Path(a.out_dir or ROOT / "runs/az_ft/samples" / Path(a.checkpoint).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = a.text or DEFAULT_TEXTS
    ipas = phonemize_texts(texts)
    if a.text is None:  # add the val item (ground-truth IPA) for A/B against the real recording
        texts, ipas = texts + ["<val> " + val_first[0]], ipas + [val_first[1]]
        shutil.copy(ROOT / "data/wavs" / val_first[0], out_dir / f"gt_{val_first[0]}")

    synth = Synth(config, a.checkpoint)
    ref_s = synth.compute_style(ref)
    for n, (t, ipa) in enumerate(zip(texts, ipas), 1):
        wav = synth(ipa, ref_s, a.alpha, a.beta, a.steps, a.embedding_scale)
        path = out_dir / f"{n:02d}.wav"
        sf.write(path, wav, 24000)
        print(f"{path}  ({len(wav)/24000:.1f}s)  {t}")


if __name__ == "__main__":
    main()
```
- [ ] **Step 2: Run on pretrained checkpoint** (inference, not training):
`.venv\Scripts\python.exe scripts/synthesize.py --checkpoint StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth --out_dir runs/pretrained_samples`
Expected: 4 WAVs + `gt_*.wav`, each 1–12 s, finite, non-silent (check `np.abs(wav).max() > 0.01` via a one-liner). English-accented/garbled Azerbaijani is expected before fine-tuning.
- [ ] **Step 3: Commit** `git add scripts/synthesize.py && git commit -m "feat: inference script"`

---

### Task 9: Launcher + monitor

**Files:**
- Create: `scripts/train.ps1`, `scripts/monitor.py`
- Test: `tests/test_monitor.py`

**Interfaces:**
- Consumes: `PERF` log line format and upstream `Epoch [e/E], Step [s/S], Loss: ...` / `Validation loss: ...` lines from Task 6.
- Produces: `monitor.parse_log(text: str) -> dict` with keys `epoch, epochs, step, steps, sec_per_step, peak_gb, elapsed_h, max_hours, last_losses (dict), val_losses (list[float]), finished (bool), nan_warnings (int)`; `monitor.eta_hours(state: dict) -> float`.

- [ ] **Step 1: Failing test** `tests/test_monitor.py`:
```python
from monitor import parse_log, eta_hours

LOG = """INFO:2026-09-21 12:00:00,000: VRAM cap: 6.80 GB of 7.96 GB
INFO:2026-09-21 12:01:00,000: Epoch [3/20], Step [10/577], Loss: 0.41000, Disc Loss: 3.9, Dur Loss: 1.2, CE Loss: 0.1, Norm Loss: 0.5, F0 Loss: 1.1, LM Loss: 1.4, Gen Loss: 6.2, Sty Loss: 0.0, Diff Loss: 0.0, DiscLM Loss: 0.0, GenLM Loss: 0.0, SLoss: 0.0, S2S Loss: 0.3, Mono Loss: 0.05
INFO:2026-09-21 12:01:00,000: PERF epoch=3 step=10/577 sec_per_step=0.800 peak_gb=6.10 elapsed_h=1.000 max_hours=7.50
INFO:2026-09-21 12:02:00,000: Validation loss: 0.380, Dur loss: 0.900, F0 loss: 1.500
WARNING:2026-09-21 12:03:00,000: NaN loss at epoch 3 step 11 - skipping update
"""


def test_parse_log():
    s = parse_log(LOG)
    assert (s["epoch"], s["epochs"], s["step"], s["steps"]) == (3, 20, 10, 577)
    assert s["sec_per_step"] == 0.8 and s["peak_gb"] == 6.1 and s["elapsed_h"] == 1.0
    assert s["last_losses"]["Loss"] == 0.41 and s["last_losses"]["Mono Loss"] == 0.05
    assert s["val_losses"] == [0.38] and s["nan_warnings"] == 1 and s["finished"] is False


def test_eta_capped_by_max_hours():
    s = parse_log(LOG)
    # remaining steps: (577-10) + 17*577 = 10376 * 0.8 s = 2.31 h ; cap leaves 6.5 h
    assert abs(eta_hours(s) - 10376 * 0.8 / 3600) < 0.01
    s["sec_per_step"] = 5.0
    assert eta_hours(s) == 6.5
```
Run → FAIL.

- [ ] **Step 2: Implement** `scripts/monitor.py`:
```python
"""Progress view for runs/az_ft/train.log. Usage: monitor.py [--log PATH] [--follow SECONDS]"""
import argparse
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\], (.*)")
PERF_RE = re.compile(r"PERF epoch=(\d+) step=(\d+)/(\d+) sec_per_step=([\d.]+) peak_gb=([\d.]+) "
                     r"elapsed_h=([\d.]+) max_hours=([\d.]+)")
VAL_RE = re.compile(r"Validation loss: ([\d.naninf]+)")


def parse_log(text):
    s = {"epoch": 0, "epochs": 0, "step": 0, "steps": 0, "sec_per_step": None, "peak_gb": None,
         "elapsed_h": 0.0, "max_hours": 7.5, "last_losses": {}, "val_losses": [],
         "finished": "Time budget" in text, "nan_warnings": text.count("NaN loss")}
    for line in text.splitlines():
        if m := EPOCH_RE.search(line):
            s["epoch"], s["epochs"], s["step"], s["steps"] = map(int, m.groups()[:4])
            s["last_losses"] = {k.strip(): float(v) for k, v in
                                (kv.split(":") for kv in m.group(5).split(","))}
        elif m := PERF_RE.search(line):
            s["sec_per_step"], s["peak_gb"] = float(m.group(4)), float(m.group(5))
            s["elapsed_h"], s["max_hours"] = float(m.group(6)), float(m.group(7))
        elif m := VAL_RE.search(line):
            s["val_losses"].append(float(m.group(1)))
    if s["epochs"] and s["epoch"] == s["epochs"] and s["val_losses"] and len(s["val_losses"]) >= s["epochs"]:
        s["finished"] = True
    return s


def eta_hours(s):
    if not s["sec_per_step"] or not s["steps"]:
        return float("nan")
    remaining = (s["steps"] - s["step"]) + (s["epochs"] - s["epoch"]) * s["steps"]
    return min(remaining * s["sec_per_step"] / 3600, s["max_hours"] - s["elapsed_h"])


def gpu_line():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "nvidia-smi unavailable"


def render(s):
    phase_losses = ", ".join(f"{k}={v:.3f}" for k, v in s["last_losses"].items()
                             if k in ("Loss", "Dur Loss", "F0 Loss", "Diff Loss", "GenLM Loss", "Mono Loss"))
    lines = [
        f"epoch {s['epoch']}/{s['epochs']}  step {s['step']}/{s['steps']}  "
        f"{s['sec_per_step'] or 0:.2f}s/step  peak {s['peak_gb'] or 0:.2f} GB",
        f"elapsed {s['elapsed_h']:.2f} h / cap {s['max_hours']:.2f} h   ETA {eta_hours(s):.2f} h",
        f"losses: {phase_losses}",
        f"val mel loss (per epoch): {', '.join(f'{v:.3f}' for v in s['val_losses'][-8:])}",
        f"NaN skips: {s['nan_warnings']}   finished: {s['finished']}",
        f"GPU: {gpu_line()}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "runs/az_ft/train.log"))
    ap.add_argument("--follow", type=float, default=0, help="refresh every N seconds")
    a = ap.parse_args()
    while True:
        p = Path(a.log)
        text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        print("\n" + time.strftime("%H:%M:%S") + "\n" + render(parse_log(text)), flush=True)
        if not a.follow:
            break
        time.sleep(a.follow)


if __name__ == "__main__":
    main()
```
Run tests → PASS (2 passed).

- [ ] **Step 3: `scripts/train.ps1`** (USER runs it; never executed by the agent):
```powershell
<#
Starts (or resumes) the Azerbaijani StyleTTS2 fine-tune.
  .\scripts\train.ps1                 # fresh run, 7.5 h hard cap
  .\scripts\train.ps1 -Resume         # continue from newest checkpoint in runs\az_ft
  .\scripts\train.ps1 -MaxHours 3     # custom cap (use remaining budget when resuming)
Monitor in a second terminal:  .\.venv\Scripts\python.exe scripts\monitor.py --follow 60
TensorBoard:                    .\.venv\Scripts\tensorboard.exe --logdir runs\az_ft\tensorboard
#>
param([double]$MaxHours = 7.5, [switch]$Resume)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Config = Join-Path $Root "configs\config_ft_az.yml"
$LogDir = Join-Path $Root "runs\az_ft"
if (-not (Test-Path $Config)) { throw "Missing $Config - run scripts\make_config.py first." }
$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
$env:PYTHONIOENCODING = "utf-8"

$used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if ($used -gt 1500) { Write-Warning "Other apps already use $used MiB VRAM - close browsers/games for best speed." }

$argsList = @("train_finetune.py", "-p", $Config, "--max_hours", $MaxHours)
if ($Resume) {
    $ckpt = Get-ChildItem $LogDir -Filter "epoch_2nd_*.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $ckpt) { throw "No checkpoint in $LogDir to resume from." }
    Write-Host "Resuming from $($ckpt.FullName)"
    $argsList += @("--resume_from", $ckpt.FullName)
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
Set-Location (Join-Path $Root "StyleTTS2")
Write-Host "Training started $(Get-Date) - hard cap $MaxHours h. Log: $LogDir\train.log"
# PS 5.1 wraps native stderr lines in ErrorRecords; with "Stop" the first warning would kill the run.
$ErrorActionPreference = "Continue"
& $Py @argsList 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "console.log") -Append
Write-Host "Finished $(Get-Date). Synthesize: .\.venv\Scripts\python.exe scripts\synthesize.py --checkpoint <runs\az_ft\epoch_2nd_XXXXX.pth>"
```
Verify syntax only (does NOT run training):
(PowerShell tool) `$e = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("$PWD\scripts\train.ps1", [ref]$null, [ref]$e); $e.Count` → `0`.

- [ ] **Step 4: Commit** `git add scripts/train.ps1 scripts/monitor.py tests/test_monitor.py && git commit -m "feat: training launcher and monitor"`

---

### Task 10: README, final verification, review, handoff

**Files:**
- Create: `README.md`

- [ ] **Step 1: README.md** — sections: Overview (what/why, PL-BERT decision), Layout (file map table above), Setup (`scripts\setup_env.ps1`), Data (`prepare_data.py`, `validate_data.py`, report fields), Calibration (`smoke_test.py`, `make_config.py`, the chosen batch/max_len/schedule copied from `configs/calibration.json` + `make_config` output), **Start training (user)** (`.\scripts\train.ps1`), Monitoring (`monitor.py --follow 60`, TensorBoard; what healthy numbers look like: mel `Loss` trending down from ~0.5, val loss decreasing then flattening, `NaN skips` ≈ 0), Resume, Listening/evaluation (`synthesize.py`, alpha/beta/embedding_scale guidance: higher `embedding_scale` (1.5–2) = more expressive intonation; lower `alpha/beta` = closer to the reference voice), Troubleshooting (OOM → rerun smoke test after closing GPU apps; NaN → lower `batch_size`; slow → check nothing else uses the GPU).
- [ ] **Step 2: Full verification** (superpowers:verification-before-completion) — run and paste real output:
  - `.venv\Scripts\python.exe -m pytest tests -v` → all pass
  - `.venv\Scripts\python.exe scripts/check_env.py` → all OK
  - `.venv\Scripts\python.exe scripts/validate_data.py` → `errors: 0`
  - `configs/config_ft_az.yml` exists; `make_config` projected_h ≤ 7.0
  - `git -C StyleTTS2 diff --stat` matches `patches/styletts2.patch`
  - Confirm `runs/az_ft` has **no** checkpoints (training not started).
- [ ] **Step 3: Code review** — superpowers:requesting-code-review over the whole repo; address findings.
- [ ] **Step 4: Commit** `git add README.md && git commit -m "docs: README with run instructions"`
- [ ] **Step 5: Handoff message** — calibration numbers, schedule (epochs / diff / joint, projected hours), exact start command `.\scripts\train.ps1`, monitor command, how to synthesize afterwards. **Do not run train.ps1.**
