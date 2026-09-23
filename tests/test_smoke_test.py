from smoke_test import slm_active, longest_lines

LOG_ZERO = ("INFO: Epoch [1/1], Step [10/1155], Loss: 0.5, DiscLM Loss: 0.00000, GenLM Loss: 0.00000, SLoss: 0.0\n"
            "INFO: Epoch [1/1], Step [20/1155], Loss: 0.4, DiscLM Loss: 0.00000, GenLM Loss: 0.00000, SLoss: 0.0\n")


def test_slm_active_false_when_slm_losses_always_zero():
    assert slm_active(LOG_ZERO) is False
    assert slm_active("no loss lines at all") is False


def test_slm_active_true_when_any_slm_loss_nonzero():
    assert slm_active(LOG_ZERO + "INFO: Step [30/1155], DiscLM Loss: 0.00000, GenLM Loss: 1.23400, SLoss: 0\n")
    assert slm_active(LOG_ZERO + "INFO: Step [30/1155], DiscLM Loss: 0.51000, GenLM Loss: 0.00000, SLoss: 0\n")


def test_longest_lines_picks_n_longest_in_descending_order():
    lines = ["a.wav|x|0", "b.wav|y|0", "c.wav|z|0", "d.wav|w|0"]
    frames = {"a.wav": 10, "b.wav": 40, "c.wav": 30, "d.wav": 20}
    assert longest_lines(lines, lambda wav: frames[wav], 2) == ["b.wav|y|0", "c.wav|z|0"]


def test_vram_conditions_reports_tightest_run():
    from smoke_test import vram_conditions
    runs = [{"ok": True, "cap_gb": 6.57, "free_gb_at_launch": 7.17},
            {"ok": True, "cap_gb": 6.50, "free_gb_at_launch": 7.10},
            {"ok": False, "oom": True}]
    assert vram_conditions(runs) == {"cap_gb": 6.5, "free_gb_at_launch": 7.1}
    assert vram_conditions([{"ok": False}]) == {"cap_gb": None, "free_gb_at_launch": None}


def test_joint_candidates_try_every_slm_window_best_first():
    from smoke_test import CANDIDATES, SLM_WINDOWS, joint_candidates
    c = joint_candidates()
    assert len(c) == len(CANDIDATES) * len(SLM_WINDOWS)
    assert c[0] == (*CANDIDATES[0], SLM_WINDOWS[0]) and c[1] == (*CANDIDATES[0], SLM_WINDOWS[1])


def test_joint_candidates_can_start_at_a_known_fitting_setting():
    import pytest
    from smoke_test import SLM_WINDOWS, joint_candidates
    c = joint_candidates("2/150")
    assert c[0] == (2, 150, SLM_WINDOWS[0]) and {(b, m) for b, m, _ in c} == {(2, 150), (2, 100)}
    with pytest.raises(ValueError):
        joint_candidates("9/9")
