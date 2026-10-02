import sys
from types import SimpleNamespace

import numpy as np
import pytest

from assistant.audio_io import MicStream, Player


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs, self.written, self.events = kwargs, [], []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")

    def write(self, data):
        self.written.append(data)


@pytest.fixture(autouse=True)
def fake_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeStream, OutputStream=FakeStream))


def test_mic_opens_16k_mono_512_sample_blocks():
    mic = MicStream(device=3)
    kw = mic._stream.kwargs
    assert (kw["samplerate"], kw["blocksize"], kw["channels"], kw["dtype"], kw["device"]) == \
           (16000, 512, 1, "float32", 3)


def test_mic_read_returns_callback_blocks_in_order_as_mono_copies():
    mic = MicStream()
    callback = mic._stream.kwargs["callback"]
    buf = np.ones((512, 1), dtype=np.float32)
    callback(buf, 512, None, None)
    buf[:] = 2  # PortAudio reuses the buffer: the queued block must be a copy
    callback(buf, 512, None, None)
    first, second = mic.read(timeout=0), mic.read(timeout=0)
    assert first.shape == (512,) and first[0] == 1 and second[0] == 2


def test_mic_read_times_out_with_none_and_drain_empties_the_queue():
    mic = MicStream()
    assert mic.read(timeout=0.01) is None
    for _ in range(3):
        mic._stream.kwargs["callback"](np.zeros((512, 1), dtype=np.float32), 512, None, None)
    mic.drain()
    assert mic.read(timeout=0.01) is None


def test_player_opens_24k_mono_and_writes_clipped_float32_columns():
    player = Player(device=5)
    stream = player._stream
    assert (stream.kwargs["samplerate"], stream.kwargs["channels"], stream.kwargs["device"]) == (24000, 1, 5)
    assert stream.events == ["start"]
    player.play(np.array([0.5, 2.0, -3.0], dtype=np.float64))
    out = stream.written[0]
    assert out.dtype == np.float32 and out.shape == (3, 1)
    assert out[:, 0].tolist() == [0.5, 1.0, -1.0]
    player.close()
    assert stream.events == ["start", "stop", "close"]
