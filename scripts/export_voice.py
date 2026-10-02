"""Bundle a training checkpoint for inference: the weights synthesis needs, its config and the reference clips.

Writes <out>/az_styletts2.pth, <out>/config_ft_az.yml and <out>/ref/*.wav, and points configs/inference.json at them.
A training checkpoint is about 2.2 GB (optimizer state, discriminators); the bundle is about 590 MB.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_NETS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder", "predictor_encoder",
                  "style_encoder", "diffusion")  # the modules Synth.__call__ / compute_style run
CONFIG_KEYS = ("ASR_config", "ASR_path", "F0_path", "PLBERT_dir", "g2p", "preprocess_params", "model_params")
WEIGHTS_NAME = "az_styletts2.pth"


def strip_checkpoint(ckpt):
    """Only the networks used at inference; no optimizer, discriminators, aligner or pitch extractor."""
    return {"net": {k: ckpt["net"][k] for k in INFERENCE_NETS}}


def inference_config(config):
    """The keys Synth reads, without the training machine's data paths and schedule."""
    return {k: config[k] for k in CONFIG_KEYS if k in config}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="default: the one configs/inference.json was tuned on")
    ap.add_argument("--out", default=str(ROOT / "models/az_voice"))
    ap.add_argument("--inference", default=str(ROOT / "configs/inference.json"))
    a = ap.parse_args()

    inference_path = Path(a.inference)
    d = json.loads(inference_path.read_text(encoding="utf-8"))
    checkpoint = Path(a.checkpoint).resolve() if a.checkpoint else ROOT / d["checkpoint"]
    out = Path(a.out).resolve()
    (out / "ref").mkdir(parents=True, exist_ok=True)

    torch.save(strip_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False)), out / WEIGHTS_NAME)
    config = yaml.safe_load((checkpoint.parent / "config_ft_az.yml").read_text(encoding="utf-8"))
    (out / "config_ft_az.yml").write_text(yaml.safe_dump(inference_config(config), sort_keys=False),
                                          encoding="utf-8")
    refs = []
    for r in d.get("ref") or []:
        shutil.copy(ROOT / r, out / "ref" / Path(r).name)
        refs.append((out / "ref" / Path(r).name).relative_to(ROOT).as_posix())
    d["checkpoint"] = (out / WEIGHTS_NAME).relative_to(ROOT).as_posix()
    if refs:
        d["ref"] = refs
    inference_path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"{out / WEIGHTS_NAME}  {(out / WEIGHTS_NAME).stat().st_size / 2**20:.0f} MB, {len(refs)} reference clip(s)")


if __name__ == "__main__":
    main()
