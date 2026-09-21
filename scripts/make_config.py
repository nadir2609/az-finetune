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
