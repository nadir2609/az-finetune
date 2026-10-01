import pytest

from assistant.text_norm import number_to_words, to_spoken


@pytest.mark.parametrize("n, words", [
    (0, "sıfır"), (7, "yeddi"), (10, "on"), (19, "on doqquz"), (40, "qırx"), (100, "yüz"),
    (150, "yüz əlli"), (999, "doqquz yüz doxsan doqquz"), (1000, "min"), (1500, "min beş yüz"),
    (2026, "iki min iyirmi altı"), (21000, "iyirmi bir min"), (100000, "yüz min"),
    (1_000_000, "bir milyon"), (2_000_300, "iki milyon üç yüz"), (3_000_000_000, "üç milyard"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_number_to_words_rejects_out_of_range():
    with pytest.raises(ValueError):
        number_to_words(10 ** 12)
    with pytest.raises(ValueError):
        number_to_words(-1)


@pytest.mark.parametrize("text, spoken", [
    ("Qiymət 150 manatdır.", "qiymət yüz əlli manatdır."),
    ("Saat 18:30-da görüşək.", "saat on səkkiz otuzda görüşək."),
    ("Saat 9:05", "saat doqquz sıfır beş"),
    ("3,5 faiz", "üç vergül beş faiz"),
    ("3.5 faiz", "üç vergül beş faiz"),
    ("Endirim 50%-dir.", "endirim əlli faizdir."),
    ("20 AZN", "iyirmi manat"),
    ("$5", "beş dollar"),
    ("100₼", "yüz manat"),
    ("7 €", "yeddi avro"),
    ("5-ci mərtəbə", "beşinci mərtəbə"),
    ("3-cü", "üçüncü"),
    ("6-cı", "altıncı"),
    ("10-cu", "onuncu"),
    ("2-ci", "ikinci"),
    ("30-da", "otuzda"),
    ("25-də", "iyirmi beşdə"),
    ("1 500 nəfər", "min beş yüz nəfər"),
    ("1.500 nəfər", "min beş yüz nəfər"),
    ("1234567890123", "bir iki üç dörd beş altı yeddi səkkiz doqquz sıfır bir iki üç"),
    ("5kq", "beş kq"),
])
def test_numbers_become_words(text, spoken):
    assert to_spoken(text) == spoken


@pytest.mark.parametrize("text, spoken", [
    ("**Salam!** Bu [sayt](https://example.az) yaxşıdır 😀", "salam! bu sayt yaxşıdır"),
    ("Bu (yəni o) yaxşıdır.", "bu, yəni o, yaxşıdır."),
    ("- birinci\n- ikinci", "birinci, ikinci"),
    ("Salam — necəsən?", "salam, necəsən?"),
    ("Bəli!!!", "bəli!"),
    ("WiFi café", "vifi cafe"),
    ("İSTİ ILIQ", "isti ılıq"),
])
def test_markup_and_letters(text, spoken):
    assert to_spoken(text) == spoken


@pytest.mark.parametrize("text", ["😀👍", "...", "", "   ", "***", "— —"])
def test_nothing_speakable_gives_empty_string(text):
    assert to_spoken(text) == ""


@pytest.mark.parametrize("text", [
    "Tarix: 01.10.2026", "+994 50 123 45 67", "E-mail: test@example.com", "x² + y² = z²",
    "Привет, как дела?", "C++ və C# dilləri", "100%", "a/b\\c", "“Sitat” — dedi.", "1.2.3.4",
    "5-ci, 6-cı və 10-cu", "№ 5", "3 × 4 = 12", "naïve façade", "₼", "18:30:45", "١٢٣", "sosial-iqtisadi",
    "-5 dərəcə", "5-", "10-15 dəqiqə", "ǝ Ә", "a\tb\r\nc", "½ ¾", "𝟓",
])
def test_output_never_breaks_the_g2p(text):
    from az_g2p import text_to_ipa
    from az_text import normalize_text
    text_to_ipa(normalize_text(to_spoken(text)))  # must not raise
