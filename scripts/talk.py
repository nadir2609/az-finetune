"""Talk to the Azerbaijani voice assistant: microphone -> Whisper -> Gemini -> StyleTTS2 -> speakers."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.audio_io import MicStream, Player, list_devices  # noqa: E402
from assistant.llm import GeminiLLM  # noqa: E402
from assistant.pipeline import FALLBACK_TEXT, Pipeline, load_config  # noqa: E402
from assistant.text_norm import to_spoken  # noqa: E402

ENV_FILE = ROOT / ".env"  # gitignored; holds GEMINI_API_KEY=<key>
NO_KEY = ("GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey, then put this line\n"
          f"in {ENV_FILE}:\n"
          "    GEMINI_API_KEY=<your key>")


def load_env_file(path, environ=os.environ):
    """Set NAME=value lines from a .env file; variables already set in the shell win."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() and not name.startswith("#"):
            environ.setdefault(name.strip(), value.strip().strip("'\""))


def stdin_lines():
    while True:
        try:
            yield input("> ")
        except EOFError:
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/assistant.yml"))
    ap.add_argument("--text", action="store_true", help="type instead of speaking (no microphone or STT)")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
    a = ap.parse_args(argv)

    if a.list_devices:
        list_devices()
        return
    cfg = load_config(a.config)
    load_env_file(ENV_FILE)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(NO_KEY)

    from assistant.tts import TTSService  # torch and StyleTTS2: only after the cheap checks passed
    print("[talk] loading the TTS model ...")
    tts = TTSService(**cfg["tts"])
    fallback = tts.synthesize(to_spoken(FALLBACK_TEXT))
    llm = GeminiLLM(**cfg["llm"])
    mic = segmenter = stt = None
    if not a.text:
        from assistant.stt import WhisperSTT
        from assistant.vad import VadConfig, silero_segmenter
        print("[talk] loading the STT model ...")
        stt = WhisperSTT(**cfg["stt"])
        segmenter = silero_segmenter(VadConfig(**cfg["vad"]))
        mic = MicStream(a.input_device)
    player = Player(a.output_device)
    pipeline = Pipeline(llm, tts, player, mic=mic, segmenter=segmenter, stt=stt, chunker=cfg["chunker"],
                        gap_ms=cfg["playback"]["gap_ms"], tail_ms=cfg["playback"]["tail_ms"],
                        fallback_wav=fallback)
    print("[talk] ready: " + ("type a message" if a.text else "speak") + " (Ctrl+C to quit)")
    try:
        if a.text:
            pipeline.run_text(stdin_lines())
        else:
            pipeline.run()
    except KeyboardInterrupt:
        print("\n[talk] bye")
    finally:
        player.close()
        if mic:
            mic.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Azerbaijani letters on a cp1252 Windows console
    main()
