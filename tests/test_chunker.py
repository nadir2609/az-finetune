from assistant.chunker import SentenceChunker


def feed_all(chunker, deltas):
    out = []
    for d in deltas:
        out += chunker.feed(d)
    return out


def test_sentence_is_emitted_once_its_end_is_followed_by_whitespace():
    c = SentenceChunker(min_chars=10, max_chars=180)
    assert c.feed("Bu gün hava çox gözəldir.") == []  # the stream could still continue with "5"
    assert c.feed(" Sabah") == ["Bu gün hava çox gözəldir."]
    assert c.flush() == ["Sabah"]


def test_boundary_split_across_deltas():
    c = SentenceChunker(min_chars=10, max_chars=180)
    assert feed_all(c, ["Bu gün hava", " çox gözəl", "dir", "! ", "Sabah da."]) == ["Bu gün hava çox gözəldir!"]
    assert c.flush() == ["Sabah da."]


def test_decimal_point_is_not_a_boundary():
    c = SentenceChunker(min_chars=5, max_chars=180)
    assert c.feed("Qiymət 3.5 manatdır. Sonra") == ["Qiymət 3.5 manatdır."]


def test_short_sentence_is_merged_with_the_next():
    c = SentenceChunker(min_chars=25, max_chars=180)
    assert c.feed("Bəli. Sizə kömək edə bilərəm. ") == ["Bəli. Sizə kömək edə bilərəm."]


def test_question_mark_is_kept():
    c = SentenceChunker(min_chars=5, max_chars=180)
    assert c.feed("Siz gələcəksinizmi? ") == ["Siz gələcəksinizmi?"]


def test_long_sentence_is_split_at_the_last_comma_before_the_limit():
    c = SentenceChunker(min_chars=5, max_chars=40)
    text = "Birinci hissə burada bitir, ikinci hissə isə burada davam edir. "
    assert c.feed(text) == ["Birinci hissə burada bitir,", "ikinci hissə isə burada davam edir."]


def test_long_text_without_commas_is_split_at_a_space():
    c = SentenceChunker(min_chars=5, max_chars=20)
    chunks = c.feed("bir iki üç dörd beş altı yeddi səkkiz doqquz on") + c.flush()
    assert all(len(x) <= 20 for x in chunks)
    assert " ".join(chunks) == "bir iki üç dörd beş altı yeddi səkkiz doqquz on"


def test_unbroken_text_longer_than_the_limit_is_hard_cut():
    c = SentenceChunker(min_chars=5, max_chars=10)
    chunks = c.feed("a" * 25) + c.flush()
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]


def test_flush_returns_short_remainder_and_empties_the_buffer():
    c = SentenceChunker()
    assert c.feed("Bəli.") == []
    assert c.flush() == ["Bəli."]
    assert c.flush() == []


def test_lines_without_punctuation_come_out_at_flush():
    c = SentenceChunker(min_chars=5, max_chars=180)
    assert c.feed("birinci sətir\nikinci sətir") == []
    assert c.flush() == ["birinci sətir\nikinci sətir"]
