"""Build configs/config_ft_az.yml from upstream config_ft.yml + smoke-test calibration."""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
DATA = ROOT / "data"
LOAD_OVERHEAD_H = 0.1  # model/WavLM load + first-batch warmup


def build_config(batch_size, max_len, epochs, diff_epoch, joint_epoch, log_dir, num_workers=2,
                 grad_checkpoint=True, mixed_precision=None, pretrained=None, slm_window=None, g2p="rules"):
    """slm_window: (min_len, max_len) of the SLM adversarial clips (None = upstream 400/500)."""
    with open(STTS / "Configs" / "config_ft.yml", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c.update({
        "log_dir": str(log_dir), "save_freq": 1, "log_interval": 10, "epochs": epochs,
        "batch_size": batch_size, "max_len": max_len, "num_workers": num_workers,
        "pretrained_model": str(pretrained or STTS / "Models/LibriTTS/epochs_2nd_00020.pth"),
        "second_stage_load_pretrained": True, "load_only_params": True,
        "grad_checkpoint": grad_checkpoint,  # decoder activation checkpointing (fits 8 GB VRAM)
        "mixed_precision": mixed_precision,  # None = fp32; "bf16" = bf16 autocast (needed for 8 GB VRAM)
        "g2p": g2p,  # which az_text G2P produced the lists; synthesize.py phonemizes with the same one
    })
    # SLMAdversarialLoss returns None (no SLM training at all) unless >= 2 samples pass its length gate,
    # so never let batch_percentage keep fewer than 2 of the batch.
    sp = c["slmadv_params"]
    sp["batch_percentage"] = max(sp["batch_percentage"], min(1.0, 2 / batch_size))
    if slm_window:
        sp["min_len"], sp["max_len"] = slm_window
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


def config_from_calibration(cal, budget_h, pretrained, log_dir, diff_frac=0.2, joint_frac=0.6):
    """Epoch schedule that fits budget_h at the calibrated speed, and the matching config."""
    val_sec = (cal["n_val"] / cal["batch_size"]) * cal["sec_per_step"]["a"] * 0.5
    s = plan_schedule(cal["n_train"], cal["batch_size"], cal["sec_per_step"], val_sec, budget_h=budget_h,
                      slm_ok=cal["slm_ok"], diff_frac=diff_frac, joint_frac=joint_frac)
    c = build_config(cal["batch_size"], cal["max_len"], s["epochs"], s["diff_epoch"], s["joint_epoch"],
                     log_dir=str(log_dir), mixed_precision=cal.get("mixed_precision"), pretrained=pretrained,
                     slm_window=cal.get("slm_window") if cal["slm_ok"] else None)
    return c, s


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_h", type=float, default=7.0)
    ap.add_argument("--calibration", default=str(ROOT / "configs/calibration.json"))
    ap.add_argument("--out", default=str(ROOT / "configs/config_ft_az.yml"))
    # Both required: with several runs on disk, a silent default could start from the wrong weights or
    # point train.ps1 at another run's checkpoints.
    ap.add_argument("--pretrained", required=True,
                    help=r"weights that start the run, e.g. runs\az_ft\epoch_2nd_00010.pth or "
                         r"StyleTTS2\Models\LibriTTS\epochs_2nd_00020.pth")
    ap.add_argument("--log_dir", required=True,
                    help="new, empty run directory (train.ps1 auto-resumes from the newest checkpoint there)")
    ap.add_argument("--diff_frac", type=float, default=0.2, help="style diffusion starts at this share of epochs")
    ap.add_argument("--joint_frac", type=float, default=0.6, help="SLM adversarial starts at this share of epochs")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    c, s = config_from_calibration(cal, args.budget_h, str(Path(args.pretrained).resolve()),
                                   Path(args.log_dir).resolve(), args.diff_frac, args.joint_frac)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    print(json.dumps({**s, "pretrained_model": c["pretrained_model"], "log_dir": c["log_dir"]}, indent=2),
          "\nwritten", args.out)


if __name__ == "__main__":
    main()
