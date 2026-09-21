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
