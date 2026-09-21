import numpy as np
import soundfile as sf
from validate_data import validate_lines


def _wav(tmp_path, name, seconds, sr=24000):
    sf.write(tmp_path / name, np.zeros(int(seconds * sr)), sr, subtype="PCM_16")


def test_good_line_passes(tmp_path):
    _wav(tmp_path, "a.wav", 2.0)
    assert validate_lines(["a.wav|salˈam .|0"], tmp_path) == []


def test_detects_problems(tmp_path):
    _wav(tmp_path, "short.wav", 0.3)
    _wav(tmp_path, "sr.wav", 2.0, sr=22050)
    errs = validate_lines([
        "missing.wav|salam|0",
        "short.wav|salam|0",
        "sr.wav|salam|0",
        "a.wav||0",
        "a.wav|sal☃m|0",
        "bad line",
    ], tmp_path)
    joined = "\n".join(errs)
    for needle in ["missing", "duration", "sample rate", "empty", "unknown", "format"]:
        assert needle in joined, needle
