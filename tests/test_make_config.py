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
