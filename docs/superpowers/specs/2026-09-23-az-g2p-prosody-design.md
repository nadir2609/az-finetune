# Azerbaijani G2P + Prosody Fixes — Design

## Context
The fine-tuned StyleTTS2 voice (`runs/az_ft`, epoch 10) sounds good but mispronounces some words and has
flat, slightly slow intonation. Analysis (2026-09-23) measured on 12 val items with their own recordings as
style references: synthesized speech is **1.12x longer** and has **0.66x the pitch range** of the speaker.
Root causes found:

1. **espeak-ng `az` G2P errors** baked into the training labels: vowel deletion (`ağac → ˈaːdʒ`), ğ rendered
   as extra vowels (`sığorta → sɯɯortˈa`, 147 train lines), spurious `ts` (`bütün → bytsˈyn`, 232 lines),
   k palatalized in back-vowel words (`bank → bˈanç`, `kod → cod`), stress ignoring unstressable suffixes.
2. **SLM adversarial training never ran**: `joint_epoch` > `epochs`, and `batch_percentage 0.5` × batch 2
   leaves 1 sample, so `SLMAdversarialLoss` returns `None`; `min_len 400` also admits only clips > 5 s.
3. **Inference**: one fixed reference clip, no speed correction, untuned alpha/beta/guidance.
4. English PL-BERT has untrained embeddings for az-only phonemes (ɯ ɫ y c œ ɟ ɣ) — out of scope here
   (Azerbaijani PL-BERT is a later step if pitch range stays low).

User decisions: fix steps 1 (inference), 2 (G2P), 3 (training recipe); ğ convention **hybrid**; stress by
**rules + editable lexicon**; implement directly after design approval. Carried-over constraint: only capped
smoke-test probes (≤ 50 steps) are run by the assistant; the user launches the real run (`train.ps1`).

## Section 1 — Rule-based G2P (`scripts/az_g2p.py`, `configs/az_lexicon.tsv`)
- Letter → IPA with context rules, output restricted to StyleTTS2's 178 symbols; `ˈ` right before the
  stressed vowel (espeak's convention); words/punctuation space-separated (`word , word .`).
- ğ: `ɣ` between vowels, else lengthens the previous vowel (`ː`). Never deletes a vowel.
- k: context vowel = following vowel for an onset consonant (followed by a vowel, or word-initial), preceding
  vowel for a coda consonant. Back → `k`; front → `c` in onset, `ç` in coda.
- l: `ɫ`/`l` by the same context rule. q: `x` word-final / before a voiceless consonant, else `ɡ`.
  r: `ɾ` between vowels, else `r`. Geminates take the realization of their second copy (`diqqət → diɡɡˈæt`).
- Stress: last vowel before the **leftmost unstressable suffix**, else the last vowel:
  negation `-ma/-mə` + tense/imperative ending (bare `-z` only after a consonant), present negative `-mır`
  after a consonant; copula `-dır/-dir/-dur/-dür` after a consonant; question particle `-mı/-mi/-mu/-mü`
  after a consonant, only in sentences ending with `?`. Harmony is checked for every suffix.
- Lexicon TSV overrides whole word forms (clitics `və ki da də ya mı mi mu mü` unstressed; `bakı`).
- Unsupported characters raise `ValueError`; hyphens separate words.
- `az_text.phonemize_texts(texts, g2p="rules")` is the default; `g2p="espeak"` stays for checkpoints
  trained on espeak labels.

## Section 2 — Data re-prep and run start
- `prepare_data.py` unchanged (uses the default G2P); regenerate lists; `validate_data.py` must be clean.
  The split is identical (same items, same seed), so val comparisons stay like-for-like.
- `make_config.py`: `g2p: rules` recorded in the config; `--pretrained runs/az_ft/epoch_2nd_00010.pth`
  (weights only, fresh epoch counter/optimizer); `--log_dir runs/az_ft2` (old run kept for A/B).
- `train.ps1` / `monitor.py` take the run directory from the config's `log_dir`; a fresh launch refuses a
  directory that already holds checkpoints (a crash would auto-resume from the old run's optimizer state).

## Section 3 — Training recipe
- `slmadv_params.batch_percentage = max(upstream, 2 / batch_size)` so ≥ 2 samples reach the SLM loss.
- SLM clip window calibrated from `(200,300) → (160,240) → (120,200)` frames; `smoke_test.py` accepts a joint
  setting only when the log proves the SLM step ran; writes `slm_window` to `calibration.json`; falls back to
  no-SLM (previous behavior) if nothing fits. `--start 2/150` skips settings the old calibration saw OOM.
- Schedule for the run from epoch 10: `--diff_frac 0.1 --joint_frac 0.3` (≈1 acoustic epoch to relearn the
  new labels, then diffusion, SLM adversarial from ~30 %); budget 7.0 h, hard cap 7.5 h.
- Deferred: freezing the aligner for longer crops — only if calibration shows ≥ 0.8 GB headroom.

## Section 4 — Inference and evaluation
- `synthesize.py`: `--speed` (durations / speed), repeatable `--ref` (mean style), defaults from
  `configs/inference.json`, G2P from the run config's `g2p` key, config defaulting to the checkpoint's own.
- `eval_tts.py`: 12 val items (after the default reference clip), fixed seeds; median `dur_ratio`,
  `f0_ratio` (5–95 % pitch spread, semitones, pYIN); `score = |log dur| + |log f0|`. `--tune` searches
  reference strategy (single / mean of 16 random / mean of 16 most expressive train clips), then
  alpha × beta × embedding_scale (3×4×3), then speed, and writes `configs/inference.json`.
  Re-run `--tune` on the new run's checkpoints.

## Testing
TDD throughout: golden-word and guard tests per G2P rule, whole-corpus properties (vowel count preserved,
one stress per word, only StyleTTS2 symbols), unit tests for inference helpers, config building, smoke-test
candidate ordering and monitor log discovery. Model-bound code verified by running it (tuning on epoch 10,
smoke test on the GPU, `train.ps1 -DryRun`).
