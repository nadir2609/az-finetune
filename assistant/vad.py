"""Cut the microphone stream into utterances using per-block speech probabilities."""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 512  # samples per block: the size Silero VAD expects at 16 kHz (32 ms)
BLOCK_MS = 1000 * BLOCK / SAMPLE_RATE


@dataclass
class VadConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    end_silence_ms: int = 600
    min_speech_ms: int = 250
    preroll_ms: int = 300
    max_utterance_s: float = 30


class Segmenter:
    """feed() 512-sample float32 blocks; returns a whole utterance when the speaker stops."""

    def __init__(self, speech_prob, cfg=None, reset_model=None):
        self.cfg = cfg or VadConfig()
        self._prob, self._reset_model = speech_prob, reset_model
        self._end_blocks = math.ceil(self.cfg.end_silence_ms / BLOCK_MS)
        self._min_blocks = math.ceil(self.cfg.min_speech_ms / BLOCK_MS)
        self._max_blocks = int(self.cfg.max_utterance_s * 1000 / BLOCK_MS)
        self._pre = deque(maxlen=math.ceil(self.cfg.preroll_ms / BLOCK_MS))
        self._clear()

    def _clear(self):
        self._pre.clear()
        self._blocks, self._speech, self._silence = None, 0, 0

    def reset(self):
        self._clear()
        if self._reset_model:
            self._reset_model()

    def feed(self, block):
        p = self._prob(block)
        if self._blocks is None:
            self._pre.append(block)
            if p >= self.cfg.start_threshold:
                self._blocks, self._speech = list(self._pre), 1
            return None
        self._blocks.append(block)
        if p >= self.cfg.end_threshold:
            self._speech, self._silence = self._speech + 1, 0
        else:
            self._silence += 1
        if self._silence < self._end_blocks and len(self._blocks) < self._max_blocks:
            return None
        blocks, speech = self._blocks, self._speech
        self._clear()
        return np.concatenate(blocks) if speech >= self._min_blocks else None


def silero_segmenter(cfg=None):
    """Segmenter backed by the Silero VAD model (CPU)."""
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad()

    def speech_prob(block):
        with torch.no_grad():
            return model(torch.from_numpy(block), SAMPLE_RATE).item()

    return Segmenter(speech_prob, cfg, reset_model=model.reset_states)
