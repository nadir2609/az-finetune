"""Turn LLM output into text the Azerbaijani G2P accepts: numbers as words, no markdown, no foreign symbols."""
import re
import unicodedata

from az_text import normalize_text

_ONES = ["sıfır", "bir", "iki", "üç", "dörd", "beş", "altı", "yeddi", "səkkiz", "doqquz"]
_TENS = ["", "on", "iyirmi", "otuz", "qırx", "əlli", "altmış", "yetmiş", "səksən", "doxsan"]
_SCALES = [(10 ** 9, "milyard"), (10 ** 6, "milyon"), (10 ** 3, "min")]
MAX_NUMBER = 999_999_999_999

_LETTERS = set("abcçdeəfgğhxıijkqlmnoöprsştuüvyz")
_KEEP = _LETTERS | set(" ,.!?;:-")
_PUNCT = ",.!?;:"
_LETTER = r"[^\W\d_]"
_SUFFIX = rf"(?:-(?={_LETTER}))?"  # the hyphen of "30-da": dropped, so the suffix joins the number word
_NUM = r"\d+(?:[.,]\d+)?"
_UNITS = [("%", "faiz"), ("₼", "manat"), (rf"(?<!{_LETTER})AZN\b", "manat"), (r"\$", "dollar"), ("€", "avro")]
_HIGH = {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "ə": "i", "i": "i", "ö": "ü", "ü": "ü"}


def _below_thousand(n):
    words = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += ([_ONES[hundreds]] if hundreds > 1 else []) + ["yüz"]
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if ones:
        words.append(_ONES[ones])
    return words


def number_to_words(n):
    """Azerbaijani cardinal for 0 <= n <= MAX_NUMBER."""
    if not 0 <= n <= MAX_NUMBER:
        raise ValueError(f"number out of range: {n}")
    if n == 0:
        return _ONES[0]
    words = []
    for value, name in _SCALES:
        count, n = divmod(n, value)
        if count:
            if not (value == 1000 and count == 1):  # 1000 is "min", not "bir min"
                words += _below_thousand(count)
            words.append(name)
    return " ".join(words + _below_thousand(n))


def _say(digits):
    """Words for a digit string; digit by digit if it has a leading zero or is above MAX_NUMBER."""
    if (len(digits) > 1 and digits[0] == "0") or int(digits) > MAX_NUMBER:
        return " ".join(_ONES[int(d)] for d in digits)
    return number_to_words(int(digits))


def _ordinal(words):
    vowel = _HIGH[[c for c in words if c in _HIGH][-1]]
    return words + ("nc" if words[-1] in _HIGH else vowel + "nc") + vowel


def _numbers(text):
    for symbol, word in _UNITS:
        text = re.sub(rf"({_NUM})\s*(?:{symbol}){_SUFFIX}", rf"\1 {word}", text, flags=re.I)
        text = re.sub(rf"(?:{symbol})\s*({_NUM})", rf"\1 {word}", text, flags=re.I)
        text = re.sub(symbol, f" {word} ", text, flags=re.I)
    text = re.sub(r"(?<![\d.,])\d{1,3}(?:[ .]\d{3})+(?!\d)", lambda m: re.sub(r"[ .]", "", m[0]), text)
    text = re.sub(rf"\b(\d{{1,2}}):(\d{{2}})\b{_SUFFIX}", lambda m: f"{_say(m[1])} {_say(m[2])}", text)
    text = re.sub(rf"(?<![\d.,])(\d+)[.,](\d+)(?![.,]?\d){_SUFFIX}",
                  lambda m: f"{_say(m[1])} vergül {_say(m[2])}", text)
    text = re.sub(rf"(\d+)-c[iıuü](?!{_LETTER})", lambda m: _ordinal(_say(m[1])), text, flags=re.I)
    text = re.sub(rf"(\d+)-(?={_LETTER})", lambda m: _say(m[1]), text)
    return re.sub(r"\d+", lambda m: f" {_say(m[0])} ", text)


def _markup(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"[*_#`~>|\\]", "", text)
    text = re.sub(r"[()\[\]{}]|\s*\n\s*", " , ", text)
    return re.sub(r"['’‘ʼ]", "", text)


def _letters(text):
    out = []
    for ch in text.replace("w", "v"):
        if ch not in _KEEP:
            ch = unicodedata.normalize("NFD", ch)[0]  # é -> e
        if ch in _KEEP:
            out.append(ch)
    return "".join(out)


def _strongest(match):
    run = re.sub(r"\s", "", match[0])
    return next((c for c in run if c in ".!?"), run[0])


def to_spoken(text):
    """Text that az_text.phonemize_texts accepts without raising; "" if nothing speakable is left."""
    text = _letters(normalize_text(_numbers(_markup(text))))
    text = re.sub(rf"(?<!{_LETTER})-|-(?!{_LETTER})", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(rf"\s+([{_PUNCT}])", r"\1", text)
    text = re.sub(rf"[{_PUNCT}](?:\s*[{_PUNCT}])+", _strongest, text)
    text = text.strip().lstrip(_PUNCT + " ")
    return text if re.search(_LETTER, text) else ""
