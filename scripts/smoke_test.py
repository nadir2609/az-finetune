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
