import pytest


def test_strip_checkpoint_keeps_only_the_inference_networks():
    from export_voice import INFERENCE_NETS, strip_checkpoint
    net = {k: {"w": 1} for k in INFERENCE_NETS + ("mpd", "msd", "wd", "text_aligner", "pitch_extractor")}
    out = strip_checkpoint({"net": net, "optimizer": [1], "iters": 5, "epoch": 13})
    assert list(out) == ["net"] and set(out["net"]) == set(INFERENCE_NETS)


def test_inference_config_drops_the_training_machine_paths():
    from export_voice import inference_config
    cfg = inference_config({"g2p": "rules", "model_params": {"n_token": 178}, "PLBERT_dir": "Utils/PLBERT/",
                            "log_dir": "C:\\runs\\x", "data_params": {"root_path": "C:\\data"}, "epochs": 14})
    assert cfg == {"PLBERT_dir": "Utils/PLBERT/", "g2p": "rules", "model_params": {"n_token": 178}}


def test_pick_device_keeps_an_explicit_choice_and_falls_back_to_cpu(monkeypatch):
    import torch
    from synthesize import pick_device
    assert pick_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device() == "cuda"


def test_check_weights_names_git_lfs_when_the_file_is_a_pointer_stub(tmp_path):
    from synthesize import check_weights
    stub = tmp_path / "az_styletts2.pth"
    stub.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 617000000\n")
    with pytest.raises(RuntimeError, match="git lfs pull"):
        check_weights(stub)
    with pytest.raises(FileNotFoundError, match="nope.pth"):
        check_weights(tmp_path / "nope.pth")
