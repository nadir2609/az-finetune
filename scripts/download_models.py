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
