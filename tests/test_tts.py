import pytest

from assistant import ROOT
from assistant.tts import resolve_checkpoint


def test_explicit_checkpoint_wins_and_is_resolved_against_the_repo_root():
    assert resolve_checkpoint("configs/az_lexicon.tsv", {"checkpoint": "other.pth"}) == \
           (ROOT / "configs/az_lexicon.tsv").resolve()


def test_falls_back_to_the_checkpoint_inference_json_was_tuned_on():
    assert resolve_checkpoint(None, {"checkpoint": "configs\\az_lexicon.tsv"}) == \
           (ROOT / "configs/az_lexicon.tsv").resolve()


def test_absolute_path_is_kept(tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    assert resolve_checkpoint(str(ckpt), {}) == ckpt.resolve()


def test_no_checkpoint_anywhere_is_a_clear_error():
    with pytest.raises(ValueError, match="tts.checkpoint"):
        resolve_checkpoint(None, {})


def test_missing_file_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="nope.pth"):
        resolve_checkpoint("runs/nope.pth", {})
