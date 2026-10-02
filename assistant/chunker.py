"""Cut a streamed LLM reply into chunks sized for the TTS model."""
import re

_END = re.compile(r"[.!?]+(?=\s)")
MIN_PIECE = 12  # characters: a cut made for timing never leaves a piece shorter than this


class SentenceChunker:
    """`rtf` is the TTS speed (seconds of synthesis per second of audio). Chunks are synthesized one at a time
    while the ones before them play, so with `rtf` set a chunk is kept short enough to be ready when the audio
    ahead of it ends; otherwise a long sentence after a short opening one leaves seconds of silence."""

    def __init__(self, min_chars=25, max_chars=180, rtf=None):
        self.min_chars, self.max_chars, self.rtf = min_chars, max_chars, rtf
        self._buf = ""
        self._first = self._sent = 0  # characters in the first chunk / in all chunks so far

    def feed(self, delta):
        """Add streamed text; returns the chunks that are now complete."""
        self._buf += delta
        out = []
        while True:
            end = self._boundary()
            if end is None:
                if len(self._buf) <= self.max_chars:
                    return out
                end = self._cut(self._buf, self.max_chars)  # no sentence end in sight: cut at a comma or space
            piece, self._buf = self._buf[:end].strip(), self._buf[end:]
            out += self._split_long(piece)

    def flush(self):
        """End of stream: whatever is left, however short."""
        piece, self._buf = self._buf.strip(), ""
        return self._split_long(piece)

    def _boundary(self):
        for m in _END.finditer(self._buf):
            if len(self._buf[:m.end()].strip()) >= self.min_chars:
                return m.end()
        return None

    def _limit(self):
        """Longest next chunk that is synthesized before the chunks already sent have finished playing
        (text length stands in for duration). The first chunk has nothing playing ahead of it."""
        if self.rtf is None or not self._sent:
            return self.max_chars
        ready = self._first + self._sent * max(0.0, 1 - self.rtf) / self.rtf
        return min(self.max_chars, max(self.min_chars, int(ready)))

    def _cut(self, text, limit, min_head=1):
        window = text[:limit]
        i = max(window.rfind(","), window.rfind(";"))
        if i >= min_head:
            return i + 1
        i = window.rfind(" ")
        return i if i >= min_head else limit

    def _split_long(self, piece):
        out = []
        while piece:
            limit, cut = self._limit(), len(piece)
            if limit < self.max_chars:  # a cut made for timing
                if len(piece) > limit and len(piece) >= 2 * MIN_PIECE:
                    cut = self._cut(piece, min(limit, len(piece) - MIN_PIECE), MIN_PIECE)
            elif len(piece) > limit:
                cut = self._cut(piece, limit)
            head, piece = piece[:cut].strip(), piece[cut:].strip()
            if head:
                out.append(head)
                self._first = self._first or len(head)
                self._sent += len(head)
        return out
