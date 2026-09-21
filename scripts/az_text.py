"""Azerbaijani text normalization + IPA phonemization for StyleTTS2 (shared by data prep and inference)."""
import os
import re
import unicodedata

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

# Mirrors StyleTTS2/text_utils.py (pinned commit) -- the 178-token vocabulary.
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
STYLETTS_SYMBOLS = set([_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa))

_PUNCT = ",.!?;:"
_backend = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = re.sub(r"[\"“”«»]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_ipa(ipa: str) -> str:
    return " ".join(re.findall(rf"[^\s{re.escape(_PUNCT)}]+|[{re.escape(_PUNCT)}]", ipa))


def _get_backend():
    global _backend
    if _backend is None:
        from phonemizer.backend import EspeakBackend
        _backend = EspeakBackend(language="az", preserve_punctuation=True, with_stress=True)
    return _backend


def phonemize_texts(texts):
    norm = [normalize_text(t) for t in texts]
    ipa = _get_backend().phonemize(norm, strip=True, njobs=1)
    return [tokenize_ipa(p) for p in ipa]


def unknown_symbols(ipa: str) -> set:
    return {c for c in ipa if c not in STYLETTS_SYMBOLS}
