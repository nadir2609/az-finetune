from types import SimpleNamespace

import pytest

from assistant.llm import GeminiLLM, LLMError, LLMRateLimited


class FakeClient:
    """Stands in for genai.Client: each call takes the next scripted reply, a list whose items are
    text deltas (None = a chunk without text) or an exception to raise at that point."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.models = self

    def generate_content_stream(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        for item in self.replies.pop(0):
            if isinstance(item, Exception):
                raise item
            yield SimpleNamespace(text=item)


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def make(client, **kw):
    return GeminiLLM("gemini-test", "Sistem.", client=client, **kw)


def test_stream_yields_deltas_and_records_both_turns():
    llm = make(FakeClient(["Salam", None, "! Necəsən?"]))
    assert list(llm.stream_reply("Salam")) == ["Salam", "! Necəsən?"]
    assert llm.history == [("user", "Salam"), ("model", "Salam! Necəsən?")]


def test_request_carries_model_system_prompt_history_and_thinking_budget():
    client = FakeClient(["Bir."], ["İki."])
    llm = make(client, max_output_tokens=50)
    list(llm.stream_reply("a"))
    list(llm.stream_reply("b"))
    call = client.calls[1]
    assert call["model"] == "gemini-test"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "a"}]},
                                {"role": "model", "parts": [{"text": "Bir."}]},
                                {"role": "user", "parts": [{"text": "b"}]}]
    assert call["config"] == {"system_instruction": "Sistem.", "max_output_tokens": 50,
                              "automatic_function_calling": {"disable": True},
                              "thinking_config": {"thinking_budget": 0}}


def test_thinking_budget_none_omits_the_setting():
    client = FakeClient(["Bir."])
    list(make(client, thinking_budget=None).stream_reply("a"))
    assert "thinking_config" not in client.calls[0]["config"]


def test_history_is_trimmed_to_max_turns():
    llm = make(FakeClient(["1."], ["2."], ["3."]), max_turns=2)
    for q in "abc":
        list(llm.stream_reply(q))
    assert llm.history == [("user", "b"), ("model", "2."), ("user", "c"), ("model", "3.")]


def test_failure_before_any_text_rolls_the_user_turn_back():
    llm = make(FakeClient([ApiError(400)], ["Bəli."]))
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert llm.history == []
    assert list(llm.stream_reply("b")) == ["Bəli."]


def test_http_429_is_rate_limited():
    llm = make(FakeClient([ApiError(429)]))
    with pytest.raises(LLMRateLimited):
        list(llm.stream_reply("a"))


def test_failure_mid_stream_keeps_what_was_said():
    llm = make(FakeClient(["Birinci cümlə. ", ApiError(503)]))
    got = []
    with pytest.raises(LLMError):
        for delta in llm.stream_reply("a"):
            got.append(delta)
    assert got == ["Birinci cümlə. "]
    assert llm.history == [("user", "a"), ("model", "Birinci cümlə. ")]


def test_reply_without_any_text_is_an_error_and_leaves_history_unchanged():
    llm = make(FakeClient([None, ""]))
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert llm.history == []


def test_whitespace_only_reply_is_an_error_and_leaves_history_unchanged():
    llm = make(FakeClient([" ", "  "]))
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert llm.history == []


def test_default_client_has_a_network_timeout(monkeypatch):
    made = []
    monkeypatch.setattr("google.genai.Client", lambda **kw: made.append(kw) or FakeClient())
    GeminiLLM("gemini-test", "Sistem.")
    assert made == [{"http_options": {"timeout": 20000}}]  # milliseconds; without it a stalled call never returns


def test_a_503_before_any_text_is_retried_once():
    client, sleeps = FakeClient([ApiError(503)], ["Bəli."]), []
    llm = make(client, sleep=sleeps.append)
    assert list(llm.stream_reply("a")) == ["Bəli."]
    assert llm.history == [("user", "a"), ("model", "Bəli.")]
    assert len(client.calls) == 2 and sleeps == [1.0]


def test_a_second_503_gives_up():
    client = FakeClient([ApiError(503)], [ApiError(503)], ["Bəli."])
    llm = make(client, sleep=lambda s: None)
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert llm.history == [] and len(client.calls) == 2


def test_a_503_after_some_text_is_not_retried():
    client = FakeClient(["Birinci. ", ApiError(503)], ["Bəli."])
    llm = make(client, sleep=lambda s: None)
    with pytest.raises(LLMError):
        list(llm.stream_reply("a"))
    assert len(client.calls) == 1
