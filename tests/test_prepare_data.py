import numpy as np
from prepare_data import trim_silence, split_items, list_line


def test_trim_silence_keeps_padding():
    sr = 24000
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    wave = np.concatenate([np.zeros(sr), tone, np.zeros(sr)])
    out = trim_silence(wave, sr, top_db=40, pad_s=0.1)
    assert abs(len(out) - int(1.2 * sr)) < int(0.15 * sr)  # librosa trims on 2048-sample frames


def test_split_is_deterministic_and_disjoint():
    items = list(range(1000))
    tr1, va1 = split_items(items, 0.05, 1234)
    tr2, va2 = split_items(items, 0.05, 1234)
    assert va1 == va2 and len(va1) == 50 and not set(tr1) & set(va1)
    assert sorted(tr1 + va1) == items


def test_list_line_format():
    assert list_line("000001.wav", "salˈam .") == "000001.wav|salˈam .|0"
