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


def never_waits(chunks, rtf):
    """True if every chunk can be synthesized (rtf s per s of audio, one at a time, text length standing in
    for duration) before the chunks ahead of it have finished playing."""
    return all(rtf * sum(map(len, chunks[:n + 1])) <= rtf * len(chunks[0]) + sum(map(len, chunks[:n]))
               for n in range(1, len(chunks)))


REPLY = ("Azərbaycanın paytaxtı Bakıdır. İçərişəhəri, Alov Qüllələrini, Dənizkənarı Bulvarı və Fəvvarələr "
         "Meydanını görmək olar. Həm də müasir memarlıq nümunələri çoxdur, onlara baxmağa dəyər.")


def test_without_rtf_a_long_sentence_after_a_short_one_would_leave_a_silence():
    chunks = SentenceChunker(min_chars=25, max_chars=180).feed(REPLY + " ")
    assert len(chunks) == 3 and not never_waits(chunks, 0.6)


def test_with_rtf_chunks_are_sized_to_be_ready_before_the_audio_ahead_of_them_ends():
    c = SentenceChunker(min_chars=25, max_chars=180, rtf=0.6)
    chunks = c.feed(REPLY + " ") + c.flush()
    assert chunks[0] == "Azərbaycanın paytaxtı Bakıdır."  # the first sentence is never cut: it sets the latency
    assert " ".join(chunks) == REPLY
    assert never_waits(chunks, 0.6)


def test_with_rtf_the_cuts_prefer_commas_and_leave_no_tiny_pieces():
    c = SentenceChunker(min_chars=25, max_chars=180, rtf=0.6)
    chunks = c.feed(REPLY + " ") + c.flush()
    assert chunks[1] == "İçərişəhəri, Alov Qüllələrini,"
    assert all(len(x) >= 12 for x in chunks)


def test_synthesis_slower_than_real_time_keeps_chunks_no_longer_than_the_first():
    c = SentenceChunker(min_chars=25, max_chars=180, rtf=1.5)
    chunks = c.feed(REPLY + " ") + c.flush()
    assert " ".join(chunks) == REPLY and max(map(len, chunks[1:])) <= len(chunks[0])


def test_fast_synthesis_leaves_sentences_whole():
    c = SentenceChunker(min_chars=25, max_chars=180, rtf=0.05)
    assert len(c.feed(REPLY + " ") + c.flush()) == 3
