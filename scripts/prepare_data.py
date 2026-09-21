"""metadata.csv + audio/*.mp3 -> data/wavs (24k mono PCM16) + StyleTTS2 lists + report."""
import argparse
import csv
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from az_text import phonemize_texts, unknown_symbols

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
MIN_S, MAX_S = 0.8, 15.0


def trim_silence(wave, sr, top_db=40, pad_s=0.1):
    _, (start, end) = librosa.effects.trim(wave, top_db=top_db)
    pad = int(pad_s * sr)
    return wave[max(0, start - pad):min(len(wave), end + pad)]


def split_items(items, val_frac=0.05, seed=1234):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(len(items) * val_frac))
    val_idx = set(idx[:n_val])
    train = [it for i, it in enumerate(items) if i not in val_idx]
    val = [it for i, it in enumerate(items) if i in val_idx]
    return train, val


def list_line(wav_name, ipa):
    return f"{wav_name}|{ipa}|0"


def decode_mp3(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def convert(mp3, out_wav):
    wave = trim_silence(decode_mp3(mp3), SR)
    peak = float(np.abs(wave).max()) if len(wave) else 0.0
    if peak > 0.99:  # avoid PCM16 clipping only; otherwise keep natural loudness
        wave = wave * (0.95 / peak)
    sf.write(out_wav, wave, SR, subtype="PCM_16")
    return len(wave) / SR, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(ROOT / "metadata.csv"))
    ap.add_argument("--audio_dir", default=str(ROOT / "audio"))
    ap.add_argument("--out_dir", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    with open(args.metadata, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def job(r):
        name = Path(r["filename"]).stem + ".wav"
        dur, peak = convert(Path(args.audio_dir) / r["filename"], wav_dir / name)
        return name, dur, peak

    with ThreadPoolExecutor(args.workers) as ex:
        converted = list(ex.map(job, rows))

    ipas = phonemize_texts([r["text"] for r in rows])

    items, rejects = [], []
    for r, (name, dur, peak), ipa in zip(rows, converted, ipas):
        unk = unknown_symbols(ipa)
        if not (MIN_S <= dur <= MAX_S):
            rejects.append({"file": name, "reason": f"duration {dur:.2f}s"})
        elif not ipa.strip() or unk:
            rejects.append({"file": name, "reason": f"ipa empty/unknown {sorted(unk)}"})
        else:
            items.append((name, ipa, dur))

    train, val = split_items(items)
    for fname, subset in [("train_list.txt", train), ("val_list.txt", val)]:
        (out / fname).write_text(
            "\n".join(list_line(n, i) for n, i, _ in subset) + "\n", encoding="utf-8", newline="\n")
    (out / "OOD_texts.txt").write_text(
        "\n".join(f"{i}|0" for _, i, _ in items) + "\n", encoding="utf-8", newline="\n")

    durs = np.array([d for _, _, d in items])
    report = {
        "total_rows": len(rows), "kept": len(items), "train": len(train), "val": len(val),
        "rejected": rejects,
        "hours": round(float(durs.sum()) / 3600, 3),
        "duration_s": {"min": float(durs.min()), "mean": float(durs.mean()), "max": float(durs.max()),
                       "p05": float(np.percentile(durs, 5)), "p95": float(np.percentile(durs, 95))},
        "clipped_inputs": sum(1 for _, _, p in converted if p > 0.99),
    }
    (out / "prep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rejected"}, indent=2), f"\nrejected: {len(rejects)}")


if __name__ == "__main__":
    main()
