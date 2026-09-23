"""Objective prosody check of a checkpoint against the real recordings, and inference-setting tuning.

Synthesizes validation items (fixed seeds) and compares each with its recording:
  dur_ratio  = synthesized / recorded duration   (>1: too slow)
  f0_ratio   = synthesized / recorded pitch range (<1: flatter than the speaker)
score = |log dur_ratio| + |log f0_ratio| on the medians (0 = matches the speaker).

  eval_tts.py --checkpoint runs\\az_ft2\\epoch_2nd_00010.pth           # evaluate current defaults
  eval_tts.py --checkpoint ... --tune                                  # tune, write configs/inference.json
"""
import argparse
import itertools
import json
import math
import random
from pathlib import Path

import librosa
import numpy as np
import torch

from az_text import phonemize_texts
from synthesize import ROOT, Synth, default_config, load_inference_defaults, metadata_text

SR = 24000
F0_SR = 16000
N_REF = 16          # clips averaged by the multi-clip reference strategies
N_CANDIDATES = 100  # train clips scanned for the most expressive references
GRID = {"alpha": [0.1, 0.3, 0.5], "beta": [0.3, 0.5, 0.7, 0.9], "embedding_scale": [1.0, 1.5, 2.0]}
SPEEDS = [1.0, 1.05, 1.1, 1.15, 1.2]


def f0_range_st(wave, sr):
    """5th..95th percentile pitch spread in semitones over voiced frames (nan if barely voiced)."""
    f0, voiced, _ = librosa.pyin(wave, fmin=60, fmax=500, sr=sr, frame_length=1024)
    f = f0[voiced]
    if len(f) < 10:
        return float("nan")
    st = 12 * np.log2(f / np.median(f))
    return float(np.percentile(st, 95) - np.percentile(st, 5))


def score(dur_ratio, f0_ratio):
    return abs(math.log(dur_ratio)) + abs(math.log(f0_ratio))


def summarize(rows):
    dur = float(np.nanmedian([r["dur_ratio"] for r in rows]))
    f0 = float(np.nanmedian([r["f0_ratio"] for r in rows]))
    return {"dur_ratio": dur, "f0_ratio": f0, "score": score(dur, f0)}


def pick_expressive(paths, k, range_fn):
    """The k clips with the widest pitch range."""
    return sorted(paths, key=lambda p: -range_fn(p))[:k]


def _wav_range(path):
    return f0_range_st(librosa.load(path, sr=F0_SR)[0], F0_SR)


class Evaluator:
    def __init__(self, synth, wav_names):
        self.synth = synth
        ipas = phonemize_texts([metadata_text(n) for n in wav_names], g2p=synth.g2p)
        self.items = []
        for name, ipa in zip(wav_names, ipas):
            gt = librosa.load(ROOT / "data/wavs" / name, sr=F0_SR)[0]
            self.items.append((ipa, len(gt) / F0_SR, f0_range_st(gt, F0_SR)))

    def run(self, ref_s, alpha, beta, embedding_scale, speed, steps=10):
        rows = []
        for n, (ipa, gt_sec, gt_range) in enumerate(self.items):
            torch.manual_seed(1234 + n)  # same diffusion noise per item across settings
            y = self.synth(ipa, ref_s, alpha, beta, steps, embedding_scale, speed)
            rng = f0_range_st(librosa.resample(y, orig_sr=SR, target_sr=F0_SR), F0_SR)
            rows.append({"dur_ratio": len(y) / SR / gt_sec, "f0_ratio": rng / gt_range})
        return summarize(rows)


def _list(name):
    return [l.split("|")[0] for l in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def reference_sets():
    """Candidate style references (paths relative to the repo root)."""
    val, train = _list("val_list.txt"), _list("train_list.txt")
    rnd = random.Random(0)
    candidates = rnd.sample(train, N_CANDIDATES)
    ranges = {w: _wav_range(ROOT / "data/wavs" / w) for w in candidates}
    return {
        "single": [f"data/wavs/{val[0]}"],
        "mean_random": [f"data/wavs/{w}" for w in candidates[:N_REF]],
        "mean_expressive": [f"data/wavs/{w}" for w in pick_expressive(candidates, N_REF, ranges.get)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n_items", type=int, default=12)
    ap.add_argument("--tune", action="store_true", help="search settings and write configs/inference.json")
    a = ap.parse_args()

    checkpoint = Path(a.checkpoint).resolve()
    synth = Synth(Path(a.config).resolve() if a.config else default_config(checkpoint), checkpoint)
    # val[0] is the default single reference clip, so evaluate on the items after it
    ev = Evaluator(synth, _list("val_list.txt")[1:1 + a.n_items])
    d = load_inference_defaults()
    style = lambda refs: synth.compute_styles([ROOT / r for r in refs])  # noqa: E731

    if not a.tune:
        refs = d["ref"] or reference_sets()["single"]
        s = ev.run(style(refs), d["alpha"], d["beta"], d["embedding_scale"], d["speed"], d["steps"])
        print(json.dumps({"settings": {**d, "ref": refs}, **s}, indent=2))
        return

    base = {"alpha": 0.3, "beta": 0.7, "embedding_scale": 1.5, "speed": 1.0}
    report = {}
    for name, refs in reference_sets().items():  # stage 1: reference strategy
        report[name] = {"ref": refs, **ev.run(style(refs), **base)}
        print("ref", name, {k: round(v, 3) for k, v in report[name].items() if k != "ref"})
    best_ref = min(("single", "mean_random", "mean_expressive"), key=lambda k: report[k]["score"])
    ref_s = style(report[best_ref]["ref"])

    best = None
    for alpha, beta, es in itertools.product(*GRID.values()):  # stage 2: style blend + guidance
        s = ev.run(ref_s, alpha, beta, es, 1.0)
        print("grid", alpha, beta, es, {k: round(v, 3) for k, v in s.items()})
        if best is None or s["score"] < best[1]["score"]:
            best = ({"alpha": alpha, "beta": beta, "embedding_scale": es}, s)
    params, best_s = {**best[0], "speed": 1.0}, best[1]  # grid ran at speed 1.0
    for speed in SPEEDS[1:]:  # stage 3: speaking rate
        s = ev.run(ref_s, **{**params, "speed": speed})
        print("speed", speed, {k: round(v, 3) for k, v in s.items()})
        if s["score"] < best_s["score"]:
            params, best_s = {**params, "speed": speed}, s
    best = (params, best_s)

    try:
        ckpt_name = str(checkpoint.relative_to(ROOT))
    except ValueError:  # checkpoint outside the repo
        ckpt_name = str(checkpoint)
    out = {"checkpoint": ckpt_name, "ref_strategy": best_ref,
           "ref": report[best_ref]["ref"], **best[0], "steps": 10,
           "metrics": best[1], "baseline": {k: report["single"][k] for k in ("dur_ratio", "f0_ratio", "score")}}
    path = ROOT / "configs/inference.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "ref"}, indent=2), "\nwritten", path)


if __name__ == "__main__":
    main()
