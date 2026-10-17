"""Measure the web demo's reply delay: stream a recorded question to a running server the way the page does.

    python scripts/serve.py                                   (in another terminal)
    python scripts/measure_latency.py --wav question.wav
    python scripts/measure_latency.py --wav question.wav --sessions 5

The WAV is one spoken question (a few seconds; `synthesize.py --text "..."` makes one in this voice) and
should end where the speech ends: the delay is counted from its last sample.
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import talk  # noqa: E402

RATE, BLOCK = 16000, 512
BLOCK_S = BLOCK / RATE
SILENCE = bytes(2 * BLOCK)
GIVE_UP_S = 30


def load_pcm(path):
    """Any WAV -> 16 kHz mono 16-bit little-endian bytes."""
    import librosa
    import numpy as np
    audio, _ = librosa.load(path, sr=RATE, mono=True)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def run_session(url, pcm, results, index):
    from websockets.sync.client import connect
    seen = {}
    with connect(url, max_size=None) as ws:
        def receive():
            try:
                for message in ws:
                    if isinstance(message, bytes):
                        seen.setdefault("audio", time.perf_counter())
                    elif '"latency"' in message:
                        seen["server"] = message
                    elif '"reply_end"' in message:
                        return
            except Exception as e:
                seen["error"] = f"{type(e).__name__}: {e}"

        receiver = threading.Thread(target=receive, daemon=True)
        receiver.start()
        for i in range(0, len(pcm), 2 * BLOCK):  # in real time, like a microphone
            ws.send(pcm[i:i + 2 * BLOCK])
            time.sleep(BLOCK_S)
        end = time.perf_counter()
        while receiver.is_alive() and time.perf_counter() - end < GIVE_UP_S:
            if "audio" not in seen:  # the page stops sending once the reply starts playing
                ws.send(SILENCE)
            time.sleep(BLOCK_S)
    results[index] = (seen.get("audio", math.nan) - end, seen.get("server"), seen.get("error"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="a recorded Azerbaijani question")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--sessions", type=int, default=1, help="visitors speaking at the same moment")
    a = ap.parse_args(argv)

    talk.load_env_file(talk.ENV_FILE)
    code = os.environ.get("ASSISTANT_ACCESS_CODE", "")
    pcm, results = load_pcm(a.wav), [None] * a.sessions

    def session(i):
        try:
            run_session(f"{a.url}?code={quote(code)}", pcm, results, i)
        except Exception as e:  # refused (wrong code, demo full) or the server is not running
            results[i] = (math.nan, None, f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=session, args=(i,)) for i in range(a.sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, (delay, server, error) in enumerate(results, 1):
        print(f"session {i}: first audio {delay:.2f} s after the end of speech | server: {server} | error: {error}")
    delays = [delay for delay, _, _ in results]
    if any(math.isnan(delay) for delay in delays):
        sys.exit("some sessions got no reply")
    print(f"worst: {max(delays):.2f} s ({'OK' if max(delays) < 3 else 'over the 3 s target'})")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    main()
