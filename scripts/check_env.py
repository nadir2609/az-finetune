"""Verify GPU/torch/espeak/disk for StyleTTS2 fine-tuning. Exit 1 on any failure."""
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail))
    except Exception as e:  # report every failure, don't stop at first
        results.append((False, name, repr(e)))


def gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), f"torch lacks kernels for sm_{cap[0]}{cap[1]}"
    x = torch.randn(1024, 1024, device="cuda")
    assert torch.isfinite((x @ x).sum()).item()
    free, total = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch {torch.__version__} free {free/2**30:.2f}/{total/2**30:.2f} GB"


def espeak():
    from az_text import phonemize_texts
    ipa = phonemize_texts(["Xidmət keyfiyyəti"])[0]
    assert "æ" in ipa, ipa
    return ipa


def shim():
    from monotonic_align import maximum_path  # noqa: F401
    return "monotonic_align importable"


def models():
    p = ROOT / "StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth"
    assert p.exists(), f"missing {p} (run scripts/download_models.py)"
    return f"{p.stat().st_size/2**20:.0f} MB"


def disk():
    free = shutil.disk_usage(ROOT).free / 2**30
    assert free > 30, f"only {free:.0f} GB free"
    return f"{free:.0f} GB free"


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    for n, f in [("gpu", gpu), ("espeak-az", espeak), ("monotonic_align", shim), ("models", models), ("disk", disk)]:
        check(n, f)
    for ok, n, d in results:
        print(f"[{'OK' if ok else 'FAIL'}] {n}: {d}")
    sys.exit(0 if all(r[0] for r in results) else 1)
