from monitor import parse_log, eta_hours

LOG = """INFO:2026-09-21 12:00:00,000: VRAM cap: 6.80 GB of 7.96 GB
INFO:2026-09-21 12:01:00,000: Epoch [3/20], Step [10/577], Loss: 0.41000, Disc Loss: 3.9, Dur Loss: 1.2, CE Loss: 0.1, Norm Loss: 0.5, F0 Loss: 1.1, LM Loss: 1.4, Gen Loss: 6.2, Sty Loss: 0.0, Diff Loss: 0.0, DiscLM Loss: 0.0, GenLM Loss: 0.0, SLoss: 0.0, S2S Loss: 0.3, Mono Loss: 0.05
INFO:2026-09-21 12:01:00,000: PERF epoch=3 step=10/577 sec_per_step=0.800 peak_gb=6.10 elapsed_h=1.000 max_hours=7.50
INFO:2026-09-21 12:02:00,000: Validation loss: 0.380, Dur loss: 0.900, F0 loss: 1.500
WARNING:2026-09-21 12:03:00,000: NaN loss at epoch 3 step 11 - skipping update
"""


def test_parse_log():
    s = parse_log(LOG)
    assert (s["epoch"], s["epochs"], s["step"], s["steps"]) == (3, 20, 10, 577)
    assert s["sec_per_step"] == 0.8 and s["peak_gb"] == 6.1 and s["elapsed_h"] == 1.0
    assert s["last_losses"]["Loss"] == 0.41 and s["last_losses"]["Mono Loss"] == 0.05
    assert s["val_losses"] == [0.38] and s["nan_warnings"] == 1 and s["finished"] is False


def test_eta_capped_by_max_hours():
    s = parse_log(LOG)
    # remaining steps: (577-10) + 17*577 = 10376 * 0.8 s = 2.31 h ; cap leaves 6.5 h
    assert abs(eta_hours(s) - 10376 * 0.8 / 3600) < 0.01
    s["sec_per_step"] = 5.0
    assert eta_hours(s) == 6.5


def test_default_log_follows_the_config_log_dir(tmp_path):
    from monitor import default_log
    cfg = tmp_path / "config_ft_az.yml"
    cfg.write_text(f"log_dir: {tmp_path / 'az_ft2'}\nepochs: 3\n", encoding="utf-8")
    assert default_log(cfg) == tmp_path / "az_ft2" / "train.log"
    assert default_log(tmp_path / "missing.yml").name == "train.log"  # falls back to runs/az_ft
