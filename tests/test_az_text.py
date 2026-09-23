from az_text import normalize_text, tokenize_ipa, phonemize_texts, unknown_symbols


def test_az_lowercase_dotted_and_dotless_i():
    assert normalize_text("İLHAM İLƏ") == "ilham ilə"
    assert normalize_text("IŞIQ") == "ışıq"


def test_whitespace_and_quotes():
    assert normalize_text('  "Salam",   dünya!  ') == "salam, dünya!"


def test_tokenize_ipa_splits_punctuation():
    assert tokenize_ipa("salˈam,  dynjˈa!") == "salˈam , dynjˈa !"
    assert tokenize_ipa("bu. o?") == "bu . o ?"


def test_phonemize_keeps_schwa_and_punctuation():
    out = phonemize_texts(["Xidmət keyfiyyəti, yaxşıdır."])[0]
    assert "æ" in out            # espeak az renders ə as æ
    assert " , " in out and out.endswith(" .")
    assert unknown_symbols(out) == set()


def test_phonemize_batch_order():
    a, b = phonemize_texts(["bir", "iki"])
    assert a != b and a.strip() and b.strip()


def test_unknown_symbols_detects():
    assert unknown_symbols("abc☃") == {"☃"}


def test_normalize_maps_cyrillic_schwa_and_typographic_punctuation():
    assert normalize_text("Әli gəldi… Sonra — getdi") == "əli gəldi. sonra , getdi"
    assert normalize_text("ǝlaqǝ") == "əlaqə"
