"""The project's fine-tuned StyleTTS2 voice, loaded once and reused for every chunk."""
import threading

import numpy as np

from assistant import ROOT
from az_text import phonemize_texts


def resolve_checkpoint(checkpoint, defaults):
    """`checkpoint` (relative to the repo root, or absolute), else the one configs/inference.json was tuned on."""
    checkpoint = checkpoint or defaults.get("checkpoint")
    if not checkpoint:
        raise ValueError("no TTS checkpoint: set tts.checkpoint in configs/assistant.yml")
    path = (ROOT / checkpoint).resolve()
    if not path.exists():
        raise FileNotFoundError(f"TTS checkpoint not found: {path}")
    return path


def style_blend(text, alpha, beta, short_chars):
    """(alpha, beta) for one chunk. Text shorter than `short_chars` takes its style fully from the reference
    clips: for under about 2 s of speech the style predicted from the text is unreliable, and the voice comes
    out as a different speaker (measured: style similarity 0.30 instead of 0.91, pitch 199 Hz instead of 134)."""
    return (0.0, 0.0) if len(text) < short_chars else (alpha, beta)


class TTSService:
    def __init__(self, checkpoint=None, short_chars=40, inference_path=ROOT / "configs/inference.json",
                 device="cuda"):
        import synthesize as S  # imports torch and StyleTTS2
        d = S.load_inference_defaults(inference_path)
        checkpoint = resolve_checkpoint(checkpoint, d)
        self._synth = S.Synth(S.default_config(checkpoint), checkpoint, device)
        if d["ref"]:
            refs = [ROOT / r for r in d["ref"]]  # inference.json paths are relative to the repo root
        else:
            first = (ROOT / "data/val_list.txt").read_text(encoding="utf-8").splitlines()[0].split("|")[0]
            refs = [ROOT / "data/wavs" / first]
        self._ref = self._synth.compute_styles(refs)
        self._blend, self._short_chars = (d["alpha"], d["beta"]), short_chars
        self._args = (d["steps"], d["embedding_scale"], d["speed"])
        self._lock = threading.Lock()
        self.synthesize("salam, mən hazıram.")  # warm-up: the first call is about 1 s slower

    def synthesize(self, text):
        """Speakable text (see text_norm.to_spoken) -> 24 kHz mono float32 audio."""
        ipa = phonemize_texts([text], g2p=self._synth.g2p)[0]
        with self._lock:
            wav = self._synth(ipa, self._ref, *style_blend(text, *self._blend, self._short_chars), *self._args)
        return np.asarray(wav, dtype=np.float32)
