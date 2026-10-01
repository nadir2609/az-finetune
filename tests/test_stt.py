from types import SimpleNamespace

from assistant.stt import clean_transcript

PROMPT = "Salam, necəsiniz? Bu gün sizə necə kömək edə bilərəm?"


def seg(text, no_speech_prob=0.1):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)


def test_segments_are_joined_and_stripped():
    assert clean_transcript([seg(" Salam."), seg(" Hava necədir? ")]) == "Salam. Hava necədir?"


def test_segments_marked_as_non_speech_are_dropped():
    assert clean_transcript([seg(" Salam.", 0.2), seg(" Altyazı.", 0.9)]) == "Salam."


def test_no_segments_is_empty():
    assert clean_transcript(iter([])) == ""


def test_text_without_letters_is_empty():
    assert clean_transcript([seg(" ... "), seg("?!")]) == ""


def test_an_echo_of_the_initial_prompt_is_empty():
    assert clean_transcript([seg(" salam necəsiniz"), seg("Bu gün sizə necə kömək edə bilərəm")], PROMPT) == ""
    assert clean_transcript([seg(" Salam, necəsiniz?")], PROMPT) == "Salam, necəsiniz?"
