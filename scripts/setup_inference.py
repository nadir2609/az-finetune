"""Set up this repo for synthesis and the voice assistant only (no training); works without a GPU.

Run it with the Python of a fresh virtual environment (3.10):
    python scripts/setup_inference.py
Installs torch (CPU build unless torch is already installed) and the requirements, clones StyleTTS2 at the
pinned commit and applies patches/styletts2.patch. Safe to re-run.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
STTS_COMMIT = "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"  # same pin as setup_env.ps1
MIN_WEIGHTS_BYTES = 2**20


def run(*cmd, cwd=ROOT):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def is_lfs_pointer(path):
    """True if `path` is the small text stub git leaves when Git LFS did not download the real file."""
    return Path(path).stat().st_size < MIN_WEIGHTS_BYTES


def main():
    pip = (sys.executable, "-m", "pip", "install")
    run(*pip, "--upgrade", "pip")
    if importlib.util.find_spec("torch") is None:
        index = () if sys.platform == "darwin" else ("--index-url", "https://download.pytorch.org/whl/cpu")
        run(*pip, "torch", "torchaudio", *index)
    run(*pip, "-r", "requirements.txt", "-r", "requirements-assistant.txt")
    run(*pip, "-e", "vendor/monotonic_align")

    if not STTS.exists():
        run("git", "clone", "https://github.com/yl4579/StyleTTS2.git", STTS)
    run("git", "checkout", "--", ".", cwd=STTS)
    run("git", "checkout", STTS_COMMIT, cwd=STTS)
    run("git", "apply", ROOT / "patches/styletts2.patch", cwd=STTS)

    weights = ROOT / json.loads((ROOT / "configs/inference.json").read_text(encoding="utf-8"))["checkpoint"]
    if not weights.exists() or is_lfs_pointer(weights):
        sys.exit(f"\nThe voice model {weights} was not downloaded. Install Git LFS (https://git-lfs.com), then run:\n"
                 "    git lfs install\n    git lfs pull")
    print(f"\nReady. Voice model: {weights} ({weights.stat().st_size / 2**20:.0f} MB)\n"
          'Try:  python scripts/synthesize.py --text "Salam, bu gün hava çox gözəldir." --out_dir out')


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Azerbaijani letters on a cp1252 Windows console
    main()
