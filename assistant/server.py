"""Web demo: one WebSocket per visitor, each running the Pipeline against the shared STT and TTS models."""
import asyncio
import hmac
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from assistant.pipeline import Pipeline
from assistant.web_io import Disconnected, IdleTimeout, SocketMic, SocketPlayer

BAD_CODE, FULL, IDLE = 4401, 4429, 4408  # WebSocket close codes the page explains to the visitor
_SEND_TIMEOUT_S = 10
WEB = Path(__file__).parent / "web"


def create_app(stt, tts, fallback_wav, cfg, access_code, make_llm, make_segmenter, rtf=None, log=print):
    """`stt` and `tts` are shared; make_llm() and make_segmenter() are called once per connection."""
    app = FastAPI()
    sessions = app.state.sessions = set()  # only touched on the event loop

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.websocket("/ws")
    async def ws(socket: WebSocket, code: str = ""):
        await socket.accept()  # a close code only reaches the browser on an accepted connection
        if not hmac.compare_digest(code.encode("utf-8"), access_code.encode("utf-8")):
            return await socket.close(BAD_CODE)
        if len(sessions) >= cfg["server"]["max_sessions"]:
            return await socket.close(FULL)

        loop, closed, done, result = asyncio.get_running_loop(), threading.Event(), asyncio.Event(), {}

        def send(make_coro):
            """Called in the session's thread; blocks until the message is handed to the socket."""
            if closed.is_set():
                raise Disconnected("connection closed")
            try:
                asyncio.run_coroutine_threadsafe(make_coro(), loop).result(_SEND_TIMEOUT_S)
            except Exception as e:
                closed.set()
                raise Disconnected(f"{type(e).__name__}: {e}") from e

        def on_event(kind, data):
            if kind == "user_text":
                mic.touch()  # only a recognised turn counts as activity: an open page streams silence
            message = {"type": kind}
            if kind == "latency":
                message.update(data)
            elif data is not None:
                message["text"] = data
            send(lambda: socket.send_json(message))

        mic = SocketMic(cfg["server"]["idle_timeout_s"])
        player = SocketPlayer(lambda data: send(lambda: socket.send_bytes(data)))
        pipeline = Pipeline(make_llm(), tts, player, mic=mic, segmenter=make_segmenter(), stt=stt,
                            chunker=cfg["chunker"], gap_ms=cfg["playback"]["gap_ms"],
                            tail_ms=0,  # the page keeps its microphone muted while it plays the reply
                            fallback_wav=fallback_wav, log=log, rtf=rtf, on_event=on_event)

        def work():
            try:
                pipeline.run()
            except IdleTimeout:
                result["close"] = IDLE
            except Disconnected:
                pass
            except Exception as e:  # one session's crash must not take the others down
                log(f"[session] {type(e).__name__}: {e}")
                result["close"] = 1011
            finally:
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:  # the server is shutting down
                    pass

        async def receive():
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes"):
                    mic.push(message["bytes"])

        sessions.add(socket)
        threading.Thread(target=work, daemon=True).start()
        receiver, finished = asyncio.ensure_future(receive()), asyncio.ensure_future(done.wait())
        try:
            await asyncio.wait({receiver, finished}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            closed.set()
            mic.close()
            sessions.discard(socket)
            receiver.cancel()
            finished.cancel()
            if "close" in result:
                try:
                    await socket.close(result["close"])
                except RuntimeError:  # already closed by the visitor
                    pass

    return app
