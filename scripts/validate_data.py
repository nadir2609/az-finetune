"""Fail loudly if StyleTTS2 lists are not trainable."""
import sys
from pathlib import Path

import soundfile as sf

from az_text import unknown_symbols

ROOT = Path(__file__).resolve().parents[1]


def validate_lines(lines, wav_dir):
    errors = []
    for n, line in enumerate(lines, 1):
        parts = line.rstrip("\n").split("|")
        if len(parts) != 3:
            errors.append(f"line {n}: bad format ({line[:60]!r})")
            continue
        name, ipa, spk = parts
        if not ipa.strip():
            errors.append(f"line {n}: empty IPA")
        unk = unknown_symbols(ipa)
        if unk:
            errors.append(f"line {n}: unknown symbols {sorted(unk)}")
        if spk != "0":
            errors.append(f"line {n}: speaker id {spk!r} != '0'")
        p = Path(wav_dir) / name
        if not p.exists():
            errors.append(f"line {n}: missing wav {name}")
            continue
        info = sf.info(str(p))
        if info.samplerate != 24000 or info.channels != 1:
            errors.append(f"line {n}: sample rate/channels {info.samplerate}/{info.channels}")
        if not (0.8 <= info.duration <= 15.0):
            errors.append(f"line {n}: duration {info.duration:.2f}s")
    return errors


def main():
    data = ROOT / "data"
    total = 0
    all_errors = []
    for fname in ["train_list.txt", "val_list.txt"]:
        lines = [l for l in (data / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        total += len(lines)
        all_errors += [f"{fname}: {e}" for e in validate_lines(lines, data / "wavs")]
    ood = [l for l in (data / "OOD_texts.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    for n, l in enumerate(ood, 1):
        if unknown_symbols(l.split("|")[0]):
            all_errors.append(f"OOD_texts.txt line {n}: unknown symbols")
    print(f"checked {total} list lines + {len(ood)} OOD lines; errors: {len(all_errors)}")
    for e in all_errors[:50]:
        print("  ", e)
    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
