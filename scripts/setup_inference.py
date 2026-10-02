"""Set up this repo for synthesis and the voice assistant only (no training); works without a GPU.

Run it with the Python of a fresh virtual environment (3.10):
    python scripts/setup_inference.py
Installs torch (the CPU build, or the CUDA 12.8 build with --gpu) and the requirements, clones StyleTTS2 at the
pinned commit and applies patches/styletts2.patch. Safe to re-run.
"""
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
STTS_COMMIT = "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"  # same pin as setup_env.ps1
MIN_WEIGHTS_BYTES = 2**20
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
GPU_INDEX = "https://download.pytorch.org/whl/cu128"  # same build as setup_env.ps1


def run(*cmd, cwd=ROOT):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def is_lfs_pointer(path):
    """True if `path` is the small text stub git leaves when Git LFS did not download the real file."""
    return Path(path).stat().st_size < MIN_WEIGHTS_BYTES


def torch_plan(installed, gpu, platform):
    """(uninstall_first, index_url) to get the wanted torch build, or None if it is already installed.
    `installed` is the installed torch version, or None."""
    if gpu:
        if installed and "+cu" in installed:
            return None
        return installed is not None, GPU_INDEX
    if installed:
        return None
    return False, None if platform == "darwin" else CPU_INDEX


def installed_torch():
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true",
                    help="install the CUDA 12.8 build of PyTorch (NVIDIA GPU), replacing a CPU build")
    a = ap.parse_args(argv)

    pip = (sys.executable, "-m", "pip", "install")
    run(*pip, "--upgrade", "pip")
    plan = torch_plan(installed_torch(), a.gpu, sys.platform)
    if plan:
        uninstall, index = plan
        if uninstall:
            run(sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio")
        run(*pip, "torch", "torchaudio", *(("--index-url", index) if index else ()))
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
