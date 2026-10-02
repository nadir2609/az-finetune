"""Azerbaijani voice-to-voice assistant: mic -> VAD -> STT -> LLM -> StyleTTS2 -> speakers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))  # az_text, synthesize (same convention as tests/conftest.py)
