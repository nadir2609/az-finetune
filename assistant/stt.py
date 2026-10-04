"""Azerbaijani speech-to-text with faster-whisper."""
import os
import re
import shutil
import threading
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "stt"  # converted models (gitignored)


def _key(text):
    return re.sub(r"[\W_]+", "", text).casefold()


def clean_transcript(segments, initial_prompt=None, max_no_speech=0.6):
    """Join Whisper segments into one line. "" when nothing was really said: segments Whisper marks as
    non-speech are dropped, and so is a result with no letters or one that just echoes the prompt
    (what Whisper tends to produce on noise)."""
    text = " ".join(s.text.strip() for s in segments if s.no_speech_prob <= max_no_speech).strip()
    if not _key(text) or (initial_prompt and _key(text) == _key(initial_prompt)):
        return ""
    return text


def _add_torch_dlls():
    """CTranslate2 needs the CUDA 12 cuBLAS/cuDNN DLLs on Windows; torch ships them in torch/lib."""
    import torch
    lib = Path(torch.__file__).parent / "lib"
    if os.name == "nt" and lib.is_dir():
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"


def ct2_model(model, root=MODEL_DIR):
    """The name or directory faster-whisper should load. A Hugging Face repo in transformers format
    (a fine-tune such as LocalDoc/azerbaijani-whisper-turbo) is converted to CTranslate2 once, into root."""
    if "/" not in model or Path(model).exists():
        return model  # a built-in size name or a local directory
    out = Path(root) / model.replace("/", "--")
    if (out / "model.bin").exists():
        return str(out)
    from huggingface_hub import list_repo_files
    if "model.bin" in list_repo_files(model):
        return model  # already CTranslate2
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperFeatureExtractor
    print(f"[stt] converting {model} for faster-whisper (once; downloads the model) ...")
    tmp = out.with_name(out.name + ".tmp")  # renamed when complete, so a stopped conversion is redone
    shutil.rmtree(tmp, ignore_errors=True)
    TransformersConverter(model, copy_files=["tokenizer.json"]).convert(str(tmp), quantization="float16")
    # faster-whisper reads the mel settings from here; without the file it assumes 80 bins, not 128
    WhisperFeatureExtractor.from_pretrained(model).to_json_file(str(tmp / "preprocessor_config.json"))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return str(out)


class WhisperSTT:
    def __init__(self, model="large-v3-turbo", device="cuda", compute_type="float16", initial_prompt=None):
        from faster_whisper import WhisperModel
        _add_torch_dlls()
        model = ct2_model(model)
        self.initial_prompt = initial_prompt
        self._lock = threading.Lock()  # one model shared by every web session
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
            self.transcribe(np.zeros(16000, dtype=np.float32))  # GPU libraries are first used here
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[stt] {device} unavailable ({type(e).__name__}: {e}); falling back to CPU int8")
            self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        """16 kHz mono float32 array -> Azerbaijani text ("" if no speech)."""
        with self._lock:  # the segments are decoded while they are read, so that stays inside
            segments, _ = self.model.transcribe(audio, language="az", beam_size=5, vad_filter=False,
                                                condition_on_previous_text=False,
                                                initial_prompt=self.initial_prompt)
            return clean_transcript(segments, self.initial_prompt)
