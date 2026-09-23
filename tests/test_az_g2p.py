import pytest

from az_g2p import word_to_ipa


# Letter rules with the default (final-syllable) stress. Words chosen to cover each rule and the
# espeak-ng failures found in the analysis (vowel deletion, ğ, spurious "ts", k in back-vowel words).
@pytest.mark.parametrize("word, ipa", [
    # ğ: [ɣ] between vowels, otherwise lengthens the previous vowel; never deletes a vowel
    ("ağac", "aɣˈadʒ"),
    ("sığorta", "sɯɣortˈa"),
    ("danışığı", "danɯʃɯɣˈɯ"),
    ("dağ", "dˈaː"),
    ("dağı", "daɣˈɯ"),
    ("oğlan", "oːɫˈan"),
    ("bağlı", "baːɫˈɯ"),
    # plain t, no affrication
    ("bütün", "bytˈyn"),
    ("etibarən", "etibaɾˈæn"),
    # k: back-vowel context -> k; front: c before a vowel / word-initial, ç in coda
    ("bank", "bˈank"),
    ("bakıda", "bakɯdˈa"),
    ("kod", "kˈod"),
    ("kitab", "citˈab"),
    ("kök", "cˈœç"),
    ("çörək", "tʃœɾˈæç"),
    ("kredit", "credˈit"),
    ("əməkdaş", "æmæçdˈaʃ"),
    ("bankçılıq", "banktʃɯɫˈɯx"),
    ("mümkün", "mymcˈyn"),
    # q: x word-final / before a voiceless consonant, else ɡ
    ("qazanıb", "ɡazanˈɯb"),
    ("olaraq", "oɫaɾˈax"),
    ("saxlayacaqlar", "saxɫajadʒaɡɫˈar"),
    # doubled consonants: both copies get the realization of the second one
    ("diqqət", "diɡɡˈæt"),
    ("müəllim", "myællˈim"),
    ("keyfiyyət", "cejfijjˈæt"),
    # l: ɫ next to back vowels, l next to front vowels
    ("tezliklə", "tezliçlˈæ"),
    ("gözəl", "ɟœzˈæl"),
    ("evdə", "evdˈæ"),
])
def test_word_letter_rules(word, ipa):
    assert word_to_ipa(word) == ipa


# Stress falls on the syllable before the leftmost unstressable suffix.
@pytest.mark.parametrize("word, ipa", [
    # negation -ma/-mə before tense/imperative endings
    ("gəlmədi", "ɟˈælmædi"),
    ("bildirməyin", "bildˈirmæjin"),
    ("almadı", "ˈaɫmadɯ"),
    ("işləmədi", "iʃlˈæmædi"),
    ("gəlməyəcək", "ɟˈælmæjædʒæç"),
    ("gəlmir", "ɟˈælmir"),
    ("almırlar", "ˈaɫmɯrɫar"),
    # guards: infinitive, nouns that merely contain ma/mə/mir
    ("gəlmək", "ɟælmˈæç"),
    ("namaz", "namˈaz"),
    ("məsələ", "mæsælˈæ"),
    ("dəmir", "dæmˈir"),
    # copula -dır/-dir/-dur/-dür after a consonant
    ("gözəldir", "ɟœzˈældir"),
    ("yoxdur", "jˈoxdur"),
    ("mümkündür", "mymcˈyndyr"),
    # guards: present tense -ır after a d-final verb stem
    ("gedir", "ɟedˈir"),
    ("yaradır", "jaɾadˈɯr"),
])
def test_word_stress_rules(word, ipa):
    assert word_to_ipa(word) == ipa


@pytest.mark.parametrize("word, ipa", [
    ("gəlirsənmi", "ɟælirsˈænmi"),
    ("biləcəksinizmi", "bilædʒæçsinˈizmi"),
    ("varmı", "vˈarmɯ"),
    ("gəlmədinmi", "ɟˈælmædinmi"),  # negation is further left than the particle
    ("sistemi", "sistemˈi"),        # accusative, not the particle (vowel before "mi")
])
def test_question_particle_stress_in_questions(word, ipa):
    assert word_to_ipa(word, question=True) == ipa


def test_question_particle_ignored_outside_questions():
    assert word_to_ipa("gəlirsənmi") == "ɟælirsænmˈi"


def test_default_lexicon_overrides_rules():
    assert word_to_ipa("bakı") == "bˈakɯ"
    assert word_to_ipa("və") == "væ"  # unstressed clitic


def test_custom_lexicon():
    assert word_to_ipa("salam", lexicon={"salam": "sˈalam"}) == "sˈalam"


def test_text_keeps_punctuation_as_tokens():
    from az_g2p import text_to_ipa
    assert text_to_ipa("salam, dünya!") == "saɫˈam , dynjˈa !"


def test_text_question_context_reaches_words_before_commas():
    from az_g2p import text_to_ipa
    assert text_to_ipa("siz gəlirsənmi? bəli.") == "sˈiz ɟælirsˈænmi ? bælˈi ."
    assert text_to_ipa("gəlirsənmi, dostum?") == "ɟælirsˈænmi , dostˈum ?"
    assert text_to_ipa("gəlirsənmi. bəli?") == "ɟælirsænmˈi . bælˈi ?"


def test_text_hyphen_separates_words():
    from az_g2p import text_to_ipa
    assert text_to_ipa("nyu-york") == "njˈu jˈork"


def test_text_rejects_unsupported_characters():
    from az_g2p import text_to_ipa
    with pytest.raises(ValueError, match="'3'"):
        text_to_ipa("3 alma")


def _corpus_pairs():
    import csv
    from pathlib import Path
    from az_g2p import VOWELS, text_to_ipa, load_lexicon
    from az_text import normalize_text
    rows = csv.DictReader(open(Path(__file__).resolve().parents[1] / "metadata.csv", encoding="utf-8"))
    lex = load_lexicon()
    for r in rows:
        norm = normalize_text(r["text"])
        words = norm.replace(",", " , ").replace(".", " . ").replace("!", " ! ").replace("?", " ? ").split()
        ipa = text_to_ipa(norm).split()
        yield words, ipa, lex, VOWELS


def test_corpus_every_word_keeps_its_vowels_and_one_stress():
    ipa_vowels = "aæeiɯoœuy"
    for words, ipa, lex, VOWELS in _corpus_pairs():
        assert len(words) == len(ipa), (words, ipa)
        for w, p in zip(words, ipa):
            assert sum(c in VOWELS for c in w) == sum(c in ipa_vowels for c in p), (w, p)
            if w not in lex and any(c in VOWELS for c in w):
                assert p.count("ˈ") == 1, (w, p)


def test_corpus_uses_only_styletts_symbols():
    from az_text import unknown_symbols
    for _, ipa, _, _ in _corpus_pairs():
        assert not unknown_symbols(" ".join(ipa)), ipa


def test_phonemize_texts_defaults_to_rules_and_keeps_espeak():
    from az_text import phonemize_texts
    assert phonemize_texts(["Ağac"]) == ["aɣˈadʒ"]
    assert phonemize_texts(["Ağac"], g2p="espeak") == ["ˈaːdʒ"]
    with pytest.raises(ValueError):
        phonemize_texts(["ağac"], g2p="nope")


def test_lexicon_file_uses_only_styletts_symbols():
    from az_g2p import load_lexicon
    from az_text import unknown_symbols
    lex = load_lexicon()
    assert lex and all(not unknown_symbols(ipa) for ipa in lex.values())
