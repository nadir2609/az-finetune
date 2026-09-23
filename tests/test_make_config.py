from make_config import plan_schedule, build_config


def test_schedule_fits_budget_and_orders_phases():
    s = plan_schedule(n_train=2310, batch_size=4, sec_per_step={"a": 0.6, "diff": 0.8, "joint": 1.6},
                      val_sec_per_epoch=30, budget_h=7.0)
    assert s["projected_h"] <= 7.0
    assert 1 <= s["diff_epoch"] < s["joint_epoch"] < s["epochs"]
    bigger = plan_schedule(2310, 4, {"a": 0.6, "diff": 0.8, "joint": 1.6}, 30, budget_h=8.0)
    assert bigger["epochs"] >= s["epochs"]


def test_schedule_without_slm_skips_joint():
    s = plan_schedule(2310, 2, {"a": 0.5, "diff": 0.7, "joint": None}, 30, budget_h=7.0, slm_ok=False)
    assert s["joint_epoch"] > s["epochs"] and s["projected_h"] <= 7.0


def test_build_config_overrides():
    c = build_config(batch_size=3, max_len=200, epochs=12, diff_epoch=2, joint_epoch=7, log_dir="X")
    assert c["batch_size"] == 3 and c["max_len"] == 200 and c["epochs"] == 12
    assert c["loss_params"]["diff_epoch"] == 2 and c["loss_params"]["joint_epoch"] == 7
    assert c["save_freq"] == 1 and c["model_params"]["multispeaker"] is True
    assert c["data_params"]["train_data"].endswith("train_list.txt")
    assert c["pretrained_model"].endswith("epochs_2nd_00020.pth")


def test_build_config_enables_grad_checkpoint_by_default():
    c = build_config(batch_size=2, max_len=150, epochs=5, diff_epoch=1, joint_epoch=3, log_dir="X")
    assert c["grad_checkpoint"] is True
    off = build_config(2, 150, 5, 1, 3, "X", grad_checkpoint=False)
    assert off["grad_checkpoint"] is False


def test_build_config_mixed_precision_passthrough():
    assert build_config(2, 150, 5, 1, 3, "X")["mixed_precision"] is None
    assert build_config(2, 150, 5, 1, 3, "X", mixed_precision="bf16")["mixed_precision"] == "bf16"


def test_build_config_pretrained_and_g2p():
    c = build_config(2, 150, 5, 1, 3, "X", pretrained="runs/az_ft/epoch_2nd_00010.pth")
    assert c["pretrained_model"] == "runs/az_ft/epoch_2nd_00010.pth" and c["load_only_params"] is True
    assert c["g2p"] == "rules"  # synthesize.py picks the G2P a checkpoint was trained with from this key


def test_build_config_slm_window_and_two_sample_batch_percentage():
    c = build_config(2, 150, 5, 1, 3, "X", slm_window=(200, 300))
    sp = c["slmadv_params"]
    assert (sp["min_len"], sp["max_len"]) == (200, 300)
    assert sp["batch_percentage"] == 1.0  # SLMAdversarialLoss needs >= 2 samples or it returns None
    assert build_config(4, 150, 5, 1, 3, "X")["slmadv_params"]["batch_percentage"] == 0.5
    assert build_config(2, 150, 5, 1, 3, "X")["slmadv_params"]["min_len"] == 400  # upstream default


def test_config_from_calibration_uses_slm_window_and_fractions():
    from make_config import config_from_calibration
    cal = {"batch_size": 2, "max_len": 150, "slm_ok": True, "slm_window": [200, 300], "mixed_precision": "bf16",
           "sec_per_step": {"a": 1.8, "diff": 1.85, "joint": 3.0}, "n_train": 2310, "n_val": 122}
    c, s = config_from_calibration(cal, budget_h=7.0, pretrained="P.pth", log_dir="L", diff_frac=0.1, joint_frac=0.3)
    assert s["projected_h"] <= 7.0 and s["diff_epoch"] == 1 and s["joint_epoch"] < s["epochs"]
    assert c["slmadv_params"]["min_len"] == 200 and c["pretrained_model"] == "P.pth" and c["log_dir"] == "L"
    assert c["loss_params"]["joint_epoch"] == s["joint_epoch"] and c["mixed_precision"] == "bf16"


def test_config_from_calibration_without_slm_keeps_upstream_window():
    from make_config import config_from_calibration
    cal = {"batch_size": 2, "max_len": 150, "slm_ok": False, "mixed_precision": "bf16",
           "sec_per_step": {"a": 1.8, "diff": 1.85, "joint": None}, "n_train": 2310, "n_val": 122}
    c, s = config_from_calibration(cal, budget_h=7.0, pretrained="P.pth", log_dir="L")
    assert s["joint_epoch"] > s["epochs"] and c["slmadv_params"]["min_len"] == 400
