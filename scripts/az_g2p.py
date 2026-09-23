"""Rule-based Azerbaijani grapheme-to-phoneme (IPA) conversion for StyleTTS2.

Replaces espeak-ng's `az` voice, whose output deletes vowels (ağac -> ˈaːdʒ), turns ğ into extra vowels,
affricates t (bütün -> bytsˈyn), palatalizes k in back-vowel words (bank -> bˈanç) and misplaces stress.
Azerbaijani spelling is close to phonemic, so every letter maps to a phoneme with a few context rules.
Output uses only symbols from StyleTTS2's 178-token vocabulary; the stress mark ˈ precedes the stressed vowel.
"""
import re
from functools import lru_cache
from pathlib import Path

LEXICON = Path(__file__).resolve().parents[1] / "configs" / "az_lexicon.tsv"
VOWELS = "aeəiıoöuü"
BACK = "aıou"
VOICELESS = "ptkçsşfhx"  # q is left out: in "qq" both copies are voiced (diqqət -> diɡɡˈæt)

_VOWEL_IPA = {"a": "a", "e": "e", "ə": "æ", "i": "i", "ı": "ɯ", "o": "o", "ö": "œ", "u": "u", "ü": "y"}
_CONSONANT_IPA = {"b": "b", "c": "dʒ", "ç": "tʃ", "d": "d", "f": "f", "g": "ɟ", "h": "h", "x": "x", "j": "ʒ",
                  "m": "m", "n": "n", "p": "p", "s": "s", "ş": "ʃ", "t": "t", "v": "v", "y": "j", "z": "z"}


def _is_vowel(word, i):
    return 0 <= i < len(word) and word[i] in VOWELS


def _context_is_back(word, i):
    """Backness of the vowel the consonant at i belongs to: the following vowel for an onset consonant
    (followed by a vowel, or word-initial), the preceding vowel for a coda consonant."""
    after = [c for c in word[i + 1:] if c in VOWELS]
    before = [c for c in reversed(word[:i]) if c in VOWELS]
    onset = i == 0 or _is_vowel(word, i + 1)
    order = (after, before) if onset else (before, after)
    for vowels in order:
        if vowels:
            return vowels[0] in BACK
    return False


def _consonant(word, i):
    ch = word[i]
    if i + 1 < len(word) and word[i + 1] == ch:  # geminate: realize like the second copy
        return _consonant(word, i + 1)
    if ch == "ğ":
        if _is_vowel(word, i - 1) and _is_vowel(word, i + 1):
            return "ɣ"
        return "ː" if _is_vowel(word, i - 1) else "ɣ"
    if ch == "k":
        if _context_is_back(word, i):
            return "k"
        return "c" if i == 0 or _is_vowel(word, i + 1) else "ç"
    if ch == "l":
        return "ɫ" if _context_is_back(word, i) else "l"
    if ch == "q":
        return "x" if i == len(word) - 1 or word[i + 1] in VOICELESS else "ɡ"
    if ch == "r":
        return "ɾ" if _is_vowel(word, i - 1) and _is_vowel(word, i + 1) else "r"
    return _CONSONANT_IPA[ch]


def _last_vowel(s):
    vowels = [c for c in s if c in VOWELS]
    return vowels[-1] if vowels else None


def _high(v):  # 4-way harmony: -dır/-dir/-dur/-dür, -mı/-mi/-mu/-mü
    return {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "ə": "i", "i": "i", "ö": "ü", "ü": "ü"}[v]


def _low(v):  # 2-way harmony: -ma/-mə
    return "a" if v in BACK else "ə"


_NEG_ENDINGS = {"a": ("dı", "yın", "z", "yacaq", "yacağ", "mış", "sa", "yıb"),
                "ə": ("di", "yin", "z", "yəcək", "yəcəy", "miş", "sə", "yib")}


def _negation_start(word):
    """Index of the leftmost verbal negation suffix (-ma/-mə + tense/imperative ending, or -mır/-mir/...)."""
    for i in range(1, len(word) - 1):
        stem = word[:i]
        v = _last_vowel(stem)
        if word[i] != "m" or v is None:
            continue
        low = _low(v)
        if word[i + 1] == low:
            for end in _NEG_ENDINGS[low]:
                # bare -z (aorist) only after a consonant: "namaz" is a noun, "yazmaz" a verb
                if word.startswith(end, i + 2) and (end != "z" or stem[-1] not in VOWELS):
                    return i
        # present tense -mır: only after a consonant-final stem ("dəmir" is a noun)
        if word[i + 1] == _high(v) and word[i + 2:i + 3] == "r" and stem[-1] not in VOWELS:
            return i
    return None


def _copula_start(word):
    """Index of a final copula -dır/-dir/-dur/-dür after a consonant (after a vowel it is usually the
    present tense of a d-final verb stem: gedir, yaradır)."""
    v = _last_vowel(word[:-3])
    if len(word) >= 4 and v and word[-3] == "d" and word[-2] == _high(v) and word[-1] == "r" \
            and word[-4] not in VOWELS:
        return len(word) - 3
    return None


def _question_particle_start(word):
    """Index of a final question particle -mı/-mi/-mu/-mü after a consonant (after a vowel it is usually
    an accusative/possessive: sistemi)."""
    v = _last_vowel(word[:-2])
    if len(word) >= 3 and v and word[-2] == "m" and word[-1] == _high(v) and word[-3] not in VOWELS:
        return len(word) - 2
    return None


def _stress_index(word, question):
    """Letter index of the stressed vowel: the last vowel before the leftmost unstressable suffix,
    otherwise the last vowel (None if the word has no vowel)."""
    idx = [i for i, c in enumerate(word) if c in VOWELS]
    if not idx:
        return None
    starts = [_negation_start(word), _copula_start(word),
              _question_particle_start(word) if question else None]
    starts = [s for s in starts if s is not None]
    if starts:
        before = [i for i in idx if i < min(starts)]
        if before:
            return before[-1]
    return idx[-1]


@lru_cache(maxsize=None)
def load_lexicon(path=LEXICON):
    """word -> IPA overrides from a UTF-8 TSV (`word<TAB>ipa`, `#` comments)."""
    lex = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            word, ipa = line.split("\t")
            lex[word.strip()] = ipa.strip()
    return lex


def word_to_ipa(word, question=False, lexicon=None):
    """IPA for one normalized (lowercase) Azerbaijani word. `question`: the word is in a question
    sentence (enables the -mı/-mi particle stress rule). `lexicon`: overrides (default: the TSV file)."""
    lex = load_lexicon() if lexicon is None else lexicon
    if word in lex:
        return lex[word]
    stressed = _stress_index(word, question)
    out = []
    for i, ch in enumerate(word):
        if ch in VOWELS:
            out.append(("ˈ" if i == stressed else "") + _VOWEL_IPA[ch])
        else:
            out.append(_consonant(word, i))
    return "".join(out)


PUNCT = ",.!?;:"
_SENTENCE_END = ".!?"
_LETTERS = set(VOWELS) | set(_CONSONANT_IPA) | set("ğkqlr")
_TOKEN = re.compile(rf"[^\s{re.escape(PUNCT)}-]+|[{re.escape(PUNCT)}]")


def text_to_ipa(text, lexicon=None):
    """IPA for normalized text (see az_text.normalize_text): words and punctuation as space-separated
    tokens, the format StyleTTS2 trains on. Hyphens separate words; other characters raise ValueError."""
    tokens = _TOKEN.findall(text)
    out = []
    for n, tok in enumerate(tokens):
        if tok in PUNCT:
            out.append(tok)
            continue
        bad = sorted(set(tok) - _LETTERS)
        if bad:
            raise ValueError(f"unsupported character {bad[0]!r} in {tok!r}")
        end = next((t for t in tokens[n + 1:] if t in _SENTENCE_END), None)
        out.append(word_to_ipa(tok, question=end == "?", lexicon=lexicon))
    return " ".join(out)
