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
