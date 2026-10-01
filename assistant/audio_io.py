"""Microphone capture and speaker playback (sounddevice / PortAudio)."""
import queue

import numpy as np

from assistant.vad import BLOCK, SAMPLE_RATE

OUT_RATE = 24000  # StyleTTS2's output sample rate


class MicStream:
    """16 kHz mono capture; the audio callback only queues blocks, read() hands them out."""

    def __init__(self, device=None):
        import sounddevice as sd
        self._q = queue.Queue()
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1, dtype="float32",
                                      device=device, callback=self._on_audio)

    def _on_audio(self, indata, frames, time, status):
        self._q.put(indata[:, 0].copy())

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()

    def close(self):
        self._stream.close()

    def read(self, timeout=None):
        """Next 512-sample float32 block, or None if none arrived within `timeout` seconds."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        """Discard everything captured so far."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


class Player:
    def __init__(self, device=None):
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="float32", device=device)
        self._stream.start()

    def play(self, wav):
        """Blocks until the 24 kHz mono buffer has been handed to the sound card."""
        self._stream.write(np.clip(wav, -1.0, 1.0).astype(np.float32).reshape(-1, 1))

    def close(self):
        self._stream.stop()
        self._stream.close()


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
