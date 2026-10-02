"""Serve the Azerbaijani voice assistant as a web page: browser -> Whisper -> Gemini -> StyleTTS2 -> browser."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import talk  # noqa: E402  (same folder: the .env loader and the missing-key message)
from assistant.audio_io import OUT_RATE  # noqa: E402
from assistant.llm import GeminiLLM  # noqa: E402
from assistant.pipeline import FALLBACK_TEXT, load_config  # noqa: E402
from assistant.text_norm import to_spoken  # noqa: E402

NO_CODE = ("ASSISTANT_ACCESS_CODE is not set. Choose a code visitors must type on the page and put this line\n"
           "in the .env file in the repo root:\n"
           "    ASSISTANT_ACCESS_CODE=<your code>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/assistant.yml"))
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    talk.load_env_file(talk.ENV_FILE)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(talk.NO_KEY)
    code = os.environ.get("ASSISTANT_ACCESS_CODE")
    if not code:
        sys.exit(NO_CODE)

    from assistant.tts import TTSService  # torch and StyleTTS2: only after the cheap checks passed
    print("[serve] loading the TTS model ...")
    tts = TTSService(**cfg["tts"])
    start = time.perf_counter()
    fallback = tts.synthesize(to_spoken(FALLBACK_TEXT))
    rtf = (time.perf_counter() - start) / (len(fallback) / OUT_RATE)  # TTS speed on this machine

    import uvicorn
    from assistant.server import create_app
    from assistant.stt import WhisperSTT
    from assistant.vad import VadConfig, silero_segmenter
    print("[serve] loading the STT model ...")
    stt = WhisperSTT(**cfg["stt"])
    app = create_app(stt, tts, fallback, cfg, code, make_llm=lambda: GeminiLLM(**cfg["llm"]),
                     make_segmenter=lambda: silero_segmenter(VadConfig(**cfg["vad"])), rtf=rtf)
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    print(f"[serve] open http://{host}:{port} (Ctrl+C to quit)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Azerbaijani letters on a cp1252 Windows console
    main()
