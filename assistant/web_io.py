"""Socket-backed stand-ins for MicStream and Player: a browser connection instead of the sound card."""
import queue
import time

import numpy as np

from assistant.vad import BLOCK

_BLOCK_BYTES = 2 * BLOCK  # 16-bit samples


class Disconnected(Exception):
    """The visitor is gone."""


class IdleTimeout(Disconnected):
    """The visitor sent no audio for too long."""


class SocketMic:
    """push() what the browser sends (16 kHz mono 16-bit little-endian, any message size);
    read() hands out 512-sample float32 blocks like MicStream."""

    def __init__(self, idle_timeout_s=300, clock=time.monotonic):
        self._q, self._buf = queue.Queue(), b""
        self._idle, self._clock = idle_timeout_s, clock
        self._last, self._closed = clock(), False

    def push(self, data):
        self._last = self._clock()
        self._buf += data
        while len(self._buf) >= _BLOCK_BYTES:
            block, self._buf = self._buf[:_BLOCK_BYTES], self._buf[_BLOCK_BYTES:]
            self._q.put(np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self._closed = True

    def read(self, timeout=None):
        """Next block, or None if none arrived within `timeout` seconds."""
        if self._closed:
            raise Disconnected("connection closed")
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            if self._clock() - self._last > self._idle:
                raise IdleTimeout(f"no audio for {self._idle} s") from None
            return None

    def drain(self):
        """Discard the blocks queued so far (a partial block is kept, so samples stay aligned)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


class SocketPlayer:
    """play() sends 24 kHz mono 16-bit little-endian audio through `send`, which raises Disconnected
    once the connection is closed. It does not wait for the audio to be heard."""

    def __init__(self, send):
        self._send = send

    def play(self, wav):
        samples = np.clip(np.nan_to_num(np.asarray(wav, dtype=np.float32)), -1.0, 1.0)
        self._send((samples * 32767).astype("<i2").tobytes())
