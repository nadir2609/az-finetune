import json
import time
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from assistant.server import BAD_CODE, FULL, IDLE, create_app

BLOCK = bytes(1024)  # 512 samples of 16-bit silence
CFG = {"server": {"max_sessions": 2, "idle_timeout_s": 300},
       "chunker": {"min_chars": 1, "max_chars": 180}, "playback": {"gap_ms": 150}}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def stream_reply(self, user_text):
        self.calls.append(user_text)
        yield "Bəli, hazıram."


class FakeTTS:
    def synthesize(self, text):
        return np.full(4, 0.5, dtype=np.float32)


class FakeSTT:
    def transcribe(self, audio):
        return "Salam"


class FakeSegmenter:
    """Every block is a whole utterance."""

    def feed(self, block):
        return block

    def reset(self):
        pass


class DeafSegmenter(FakeSegmenter):
    """Never hears an utterance (a page left open in a quiet room)."""

    def feed(self, block):
        return None


def make_app(code="secret", segmenter=FakeSegmenter, **server):
    llms = []

    def make_llm():
        llms.append(FakeLLM())
        return llms[-1]

    cfg = {**CFG, "server": {**CFG["server"], **server}}
    app = create_app(FakeSTT(), FakeTTS(), np.zeros(4, dtype=np.float32), cfg, code, make_llm, segmenter,
                     log=lambda line: None)
    return app, llms


def connect(app, code="secret"):
    return TestClient(app).websocket_connect("/ws?code=" + quote(code))


def turn(ws):
    """Send one block of audio; the reply's messages up to reply_end as (kind, payload)."""
    ws.send_bytes(BLOCK)
    out = []
    while True:
        message = ws.receive()
        if message.get("bytes") is not None:
            out.append(("audio", message["bytes"]))
            continue
        data = json.loads(message["text"])
        out.append((data["type"], data))
        if data["type"] == "reply_end":
            return out


def close_code(ws):
    with pytest.raises(WebSocketDisconnect) as e:
        ws.receive_text()
    return e.value.code


def wait_until(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


def test_a_turn_returns_text_audio_and_timings_in_order():
    app, _ = make_app()
    with connect(app) as ws:
        out = turn(ws)
    assert [kind for kind, _ in out] == ["user_text", "audio", "reply_text", "latency", "reply_end"]
    assert out[0][1]["text"] == "Salam" and out[2][1]["text"] == "Bəli, hazıram."
    assert np.frombuffer(out[1][1], dtype="<i2").tolist() == [16383] * 4
    assert "first_audio" in out[3][1]


def test_wrong_or_missing_access_code_is_refused():
    app, llms = make_app()
    with connect(app, "wrong") as ws:
        assert close_code(ws) == BAD_CODE
    with TestClient(app).websocket_connect("/ws") as ws:
        assert close_code(ws) == BAD_CODE
    assert llms == []


def test_access_code_with_non_ascii_letters():
    app, _ = make_app(code="gizli-söz")
    with connect(app, "gizli-soz") as ws:
        assert close_code(ws) == BAD_CODE
    with connect(app, "gizli-söz") as ws:
        assert turn(ws)[0][0] == "user_text"


def test_connection_beyond_the_limit_is_refused():
    app, _ = make_app(max_sessions=1)
    with connect(app) as first:
        turn(first)
        with connect(app) as second:
            assert close_code(second) == FULL


def test_each_session_has_its_own_conversation():
    app, llms = make_app()
    with connect(app) as a, connect(app) as b:
        turn(a)
        turn(b)
    assert [llm.calls for llm in llms] == [["Salam"], ["Salam"]]


def test_closing_the_page_frees_the_session():
    app, _ = make_app(max_sessions=1)
    with connect(app) as ws:
        turn(ws)
    assert wait_until(lambda: not app.state.sessions)
    with connect(app) as ws:
        assert turn(ws)[-1][0] == "reply_end"


def test_a_text_message_from_the_client_is_ignored():
    app, _ = make_app()
    with connect(app) as ws:
        ws.send_text("hello")
        assert turn(ws)[0][0] == "user_text"


def test_a_visitor_who_never_speaks_is_closed_and_the_slot_freed():
    app, _ = make_app(max_sessions=1, idle_timeout_s=0.2)
    with connect(app) as ws:
        assert close_code(ws) == IDLE
    assert wait_until(lambda: not app.state.sessions)


def test_the_page_and_its_scripts_are_served():
    app, _ = make_app()
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200 and "/static/app.js" in page.text
    assert "AudioWorkletNode" in client.get("/static/app.js").text
    assert "registerProcessor" in client.get("/static/mic-worklet.js").text


def test_an_open_page_that_only_streams_silence_is_closed_and_the_slot_freed():
    app, _ = make_app(segmenter=DeafSegmenter, max_sessions=1, idle_timeout_s=0.2)
    with connect(app) as ws:
        for _ in range(20):  # 1 s of audio arriving all the time
            ws.send_bytes(BLOCK)
            time.sleep(0.05)
        assert not app.state.sessions  # closed while the stream was still running
        assert close_code(ws) == IDLE


def test_a_recognised_turn_restarts_the_idle_clock():
    app, _ = make_app(idle_timeout_s=0.4)
    with connect(app) as ws:
        for _ in range(3):  # 0.5 s in all: longer than the timeout, but never 0.4 s without a turn
            time.sleep(0.25)
            assert turn(ws)[-1][0] == "reply_end"
