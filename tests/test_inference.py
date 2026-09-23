import json
import math

import numpy as np
import pytest
import torch


def test_predict_durations_divides_by_speed_and_keeps_minimum_one():
    from synthesize import predict_durations
    logits = torch.full((1, 3, 10), 10.0)  # sigmoid ~1 per bin -> ~10 frames per token
    assert predict_durations(logits, speed=1.0).tolist() == [10, 10, 10]
    assert predict_durations(logits, speed=2.0).tolist() == [5, 5, 5]
    assert predict_durations(torch.full((1, 2, 10), -10.0), speed=1.0).tolist() == [1, 1]


def test_average_styles_is_elementwise_mean():
    from synthesize import average_styles
    a, b = torch.zeros(1, 256), torch.ones(1, 256) * 2
    out = average_styles([a, b])
    assert out.shape == (1, 256) and torch.allclose(out, torch.ones(1, 256))


def test_g2p_for_config_defaults_to_espeak_for_old_runs():
    from synthesize import g2p_for_config
    assert g2p_for_config({}) == "espeak"
    assert g2p_for_config({"g2p": "rules"}) == "rules"


def test_inference_defaults_merge_file_over_builtin(tmp_path):
    from synthesize import load_inference_defaults
    assert load_inference_defaults(tmp_path / "missing.json")["speed"] == 1.0
    p = tmp_path / "inference.json"
    p.write_text(json.dumps({"beta": 0.5, "speed": 1.1, "ref": ["data/wavs/a.wav"]}), encoding="utf-8")
    d = load_inference_defaults(p)
    assert (d["alpha"], d["beta"], d["speed"], d["ref"]) == (0.3, 0.5, 1.1, ["data/wavs/a.wav"])


def test_f0_range_st_of_one_octave_glide():
    from eval_tts import f0_range_st
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    f = 100 * 2 ** (t / 2.0)  # 100 -> 200 Hz, log-linear: 12 semitones
    wave = 0.5 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    assert f0_range_st(wave, sr) == pytest.approx(0.9 * 12, abs=1.0)  # 5th..95th percentile


def test_score_is_zero_only_at_perfect_ratios():
    from eval_tts import score
    assert score(1.0, 1.0) == 0.0
    assert score(1.12, 0.66) == pytest.approx(abs(math.log(1.12)) + abs(math.log(0.66)))
    assert score(0.9, 1.0) == pytest.approx(score(1 / 0.9, 1.0))


def test_summarize_takes_medians():
    from eval_tts import summarize
    rows = [{"dur_ratio": r, "f0_ratio": f} for r, f in [(1.0, 0.5), (1.2, 0.7), (1.1, 0.6)]]
    s = summarize(rows)
    assert (s["dur_ratio"], s["f0_ratio"]) == (1.1, 0.6)
    assert s["score"] == pytest.approx(abs(math.log(1.1)) + abs(math.log(0.6)))


def test_pick_expressive_returns_top_k_by_range():
    from eval_tts import pick_expressive
    ranges = {"a.wav": 5.0, "b.wav": 20.0, "c.wav": 12.0}
    assert pick_expressive(list(ranges), 2, ranges.get) == ["b.wav", "c.wav"]


def test_default_config_is_the_one_next_to_the_checkpoint(tmp_path):
    from synthesize import default_config
    ckpt = tmp_path / "az_ft2" / "epoch_2nd_00003.pth"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"")
    (ckpt.parent / "config_ft_az.yml").write_text("g2p: rules\n", encoding="utf-8")
    assert default_config(ckpt) == ckpt.parent / "config_ft_az.yml"
    lone = tmp_path / "elsewhere.pth"
    assert default_config(lone) != ckpt.parent / "config_ft_az.yml"
