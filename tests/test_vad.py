import numpy as np

from assistant.vad import BLOCK, Segmenter, VadConfig

# 3 silent blocks end an utterance, 2 speech blocks are the minimum, 2 blocks of pre-roll
CFG = VadConfig(start_threshold=0.5, end_threshold=0.35, end_silence_ms=96, min_speech_ms=64,
                preroll_ms=64, max_utterance_s=30)


def run(probs, cfg=CFG):
    """Feed one block per probability (block n is filled with the value n); returns feed()'s results."""
    it = iter(probs)
    seg = Segmenter(lambda block: next(it), cfg)
    return [seg.feed(np.full(BLOCK, n, dtype=np.float32)) for n in range(len(probs))]


def block_ids(utterance):
    return utterance[::BLOCK].astype(int).tolist()


def test_utterance_includes_preroll_and_ends_after_the_silence_window():
    out = run([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert [o is not None for o in out] == [False] * 8 + [True, False]
    assert block_ids(out[8]) == [2, 3, 4, 5, 6, 7, 8]  # block 2 is pre-roll, block 3 triggered


def test_a_dip_between_the_thresholds_does_not_end_the_utterance():
    out = run([0.9, 0.4, 0.4, 0.4, 0.4, 0.9, 0.1, 0.1, 0.1])
    assert [o is not None for o in out] == [False] * 8 + [True]


def test_short_burst_is_discarded():
    out = run([0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert all(o is None for o in out)


def test_listening_resumes_after_an_utterance():
    out = run([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
    assert block_ids(out[4]) == [0, 1, 2, 3, 4]
    assert block_ids(out[10]) == [5, 6, 7, 8, 9, 10]


def test_utterance_is_cut_at_the_maximum_length():
    cfg = VadConfig(end_silence_ms=96, min_speech_ms=64, preroll_ms=64, max_utterance_s=0.16)  # 5 blocks
    out = run([0.9] * 8, cfg)
    assert [o is not None for o in out] == [False] * 4 + [True] + [False] * 3
    assert len(out[4]) == 5 * BLOCK


def test_reset_drops_a_partial_utterance_and_resets_the_model():
    calls = []
    it = iter([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    seg = Segmenter(lambda block: next(it), CFG, reset_model=lambda: calls.append(1))
    seg.feed(np.zeros(BLOCK, dtype=np.float32))
    seg.feed(np.zeros(BLOCK, dtype=np.float32))
    seg.reset()
    assert calls == [1]
    assert all(seg.feed(np.zeros(BLOCK, dtype=np.float32)) is None for _ in range(4))
