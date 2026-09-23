"""Capped (<=50 step) runs per training phase to measure VRAM + s/step. Never a real training run."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import yaml

from make_config import build_config

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
PY = str(ROOT / ".venv/Scripts/python.exe")
SMOKE = ROOT / "runs/smoke"
CANDIDATES = [(4, 300), (4, 200), (3, 200), (2, 200), (2, 150), (2, 100)]  # (batch_size, max_len), best first
# SLM adversarial clip window (slmadv min_len, max_len in frames), largest first. Upstream's 400/500 only
# admits utterances > 5 s (most of this dataset is shorter) and does not fit 8 GB.
SLM_WINDOWS = [(200, 300), (160, 240), (120, 200)]
STEPS = 30
MIXED_PRECISION = "bf16"  # rung 2 of the 8 GB memory ladder (fp32 + grad checkpointing alone OOMs)
# Fit-check on the longest utterances: text aligner / BERT / style encoders run on the full utterance, so a
# random 30-step sample can pass and the real run still OOM on a batch of long files.
STRESS_N = 200
ENV = dict(os.environ, PHONEMIZER_ESPEAK_LIBRARY=r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONIOENCODING="utf-8")


def slm_active(log_text):
    """True if any logged DiscLM/GenLM loss is non-zero (the SLM adversarial step actually ran)."""
    vals = re.findall(r"(?:DiscLM|GenLM) Loss: ([-+0-9.eE]+|nan|inf)", log_text)
    return any(v in ("nan", "inf") or float(v) != 0.0 for v in vals)


def longest_lines(lines, frames_fn, n):
    """The n list lines whose wav (first '|' field) is longest, longest first."""
    return sorted(lines, key=lambda l: -frames_fn(l.split("|")[0]))[:n]


def stress_list():
    lines = [l for l in (ROOT / "data/train_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = SMOKE / "stress_train_list.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    longest = longest_lines(lines, lambda w: sf.info(str(ROOT / "data/wavs" / w)).frames, STRESS_N)
    out.write_text("\n".join(longest) + "\n", encoding="utf-8")
    return out


def vram_conditions(runs):
    """Tightest per-process VRAM cap / free VRAM at launch over the successful runs (None if none)."""
    ok = [r for r in runs if r.get("ok") and r.get("cap_gb") is not None]
    if not ok:
        return {"cap_gb": None, "free_gb_at_launch": None}
    return {"cap_gb": round(min(r["cap_gb"] for r in ok), 3),
            "free_gb_at_launch": round(min(r["free_gb_at_launch"] for r in ok), 3)}


def joint_candidates(start=None):
    """(batch_size, max_len, slm_window) to try for the joint phase, best first; `start` ("2/150") skips
    the settings before it (e.g. ones a previous calibration already saw OOM)."""
    cands = CANDIDATES
    if start:
        key = tuple(int(x) for x in start.split("/"))
        if key not in CANDIDATES:
            raise ValueError(f"--start {start} is not one of {CANDIDATES}")
        cands = CANDIDATES[CANDIDATES.index(key):]
    return [(bs, ml, w) for bs, ml in cands for w in SLM_WINDOWS]


def run_phase(bs, max_len, phase, slm_window=None, max_hours=1.0, max_steps=STEPS):
    """phase: 'a' (acoustic only), 'diff' (+style diffusion), 'joint' (+SLM adversarial)."""
    diff_epoch, joint_epoch = {"a": (99, 99), "diff": (0, 99), "joint": (0, 0)}[phase]
    tag = f"_slm{slm_window[0]}-{slm_window[1]}" if phase == "joint" and slm_window else ""
    log_dir = SMOKE / f"bs{bs}_len{max_len}_{phase}{tag}"
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True)
    cfg = build_config(bs, max_len, epochs=1, diff_epoch=diff_epoch, joint_epoch=joint_epoch,
                       log_dir=str(log_dir), num_workers=0, mixed_precision=MIXED_PRECISION,
                       slm_window=slm_window)
    cfg["data_params"]["train_data"] = str(stress_list())
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
    log = (log_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    return {"ok": True, "slm_active": slm_active(log), **json.loads(stats.read_text(encoding="utf-8"))}


def count(name):
    return sum(1 for l in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if l.strip())


def _key(bs, ml, phase, w=None):
    return f"{bs}/{ml}/{phase}" + (f"/{w[0]}-{w[1]}" if w else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help='first "batch/max_len" candidate to try, e.g. 2/150')
    args = ap.parse_args()
    results = {}
    chosen = None
    for bs, ml, w in joint_candidates(args.start):  # 1) largest setting where the SLM phase really runs
        r = run_phase(bs, ml, "joint", slm_window=w)
        results[_key(bs, ml, "joint", w)] = r
        print(bs, ml, "joint", w, {k: r.get(k) for k in ("ok", "oom", "slm_active", "sec_per_step_median",
                                                         "peak_reserved_gb")})
        if r["ok"] and r["slm_active"]:
            chosen, slm_ok, window = (bs, ml), True, w
            break
        if not r["ok"] and not r["oom"]:
            run_dir = SMOKE / f"bs{bs}_len{ml}_joint_slm{w[0]}-{w[1]}"
            sys.exit(f"non-OOM failure - see {run_dir}/stdout.txt\n{r['tail']}")
        # OOM, or it fit only because the SLM step never ran: try a smaller window / setting
    if chosen is None:  # 2) SLM phase never fits: pick the best setting for the diffusion phase
        slm_ok, window = False, None
        for bs, ml in dict.fromkeys((b, m) for b, m, _ in joint_candidates(args.start)):
            r = run_phase(bs, ml, "diff")
            results[_key(bs, ml, "diff")] = r
            print(bs, ml, "diff", {k: r.get(k) for k in ("ok", "oom", "sec_per_step_median", "peak_reserved_gb")})
            if r["ok"]:
                chosen = (bs, ml)
                break
            if not r["oom"]:
                sys.exit(f"non-OOM failure:\n{r['tail']}")
    if chosen is None:
        sys.exit("nothing fits in VRAM - close other GPU apps and retry")
    write_calibration(*chosen, slm_ok, results, window)


def write_calibration(bs, ml, slm_ok, results, slm_window=None):
    """Measure every needed phase at the chosen setting (reusing runs already in `results`) and write
    configs/calibration.json, including the VRAM conditions the numbers were measured under."""
    sps, peak, used = {}, {}, []
    for phase in ("a", "diff", "joint"):
        key = _key(bs, ml, phase, slm_window if phase == "joint" else None)
        if phase == "joint" and not slm_ok:
            sps[phase], peak[phase] = None, None
            continue
        r = results.get(key) or run_phase(bs, ml, phase, slm_window=slm_window)
        results[key] = r
        if not r["ok"]:
            sys.exit(f"phase {phase} failed at chosen setting:\n{r['tail']}")
        sps[phase], peak[phase] = r["sec_per_step_median"], r["peak_reserved_gb"]
        used.append(r)
    cal = {"batch_size": bs, "max_len": ml, "slm_ok": slm_ok,
           "slm_window": list(slm_window) if slm_window else None, "mixed_precision": MIXED_PRECISION,
           "sec_per_step": sps, "peak_gb": peak, **vram_conditions(used),
           "n_train": count("train_list.txt"), "n_val": count("val_list.txt"),
           "fit_check": f"longest {STRESS_N} train utterances, {STEPS} steps", "raw": results}
    out = ROOT / "configs/calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in cal.items() if k != "raw"}, indent=2))


if __name__ == "__main__":
    main()
