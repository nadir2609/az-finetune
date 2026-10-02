"""Cut a streamed LLM reply into chunks sized for the TTS model."""
import re

_END = re.compile(r"[.!?]+(?=\s)")


class SentenceChunker:
    def __init__(self, min_chars=25, max_chars=180):
        self.min_chars, self.max_chars = min_chars, max_chars
        self._buf = ""

    def feed(self, delta):
        """Add streamed text; returns the chunks that are now complete."""
        self._buf += delta
        out = []
        while True:
            end = self._boundary()
            if end is None:
                if len(self._buf) <= self.max_chars:
                    return out
                end = self._cut(self._buf)  # no sentence end in sight: cut at a comma or space
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

    def _cut(self, text):
        window = text[:self.max_chars]
        i = max(window.rfind(","), window.rfind(";"))
        if i > 0:
            return i + 1
        i = window.rfind(" ")
        return i if i > 0 else self.max_chars

    def _split_long(self, piece):
        out = []
        while len(piece) > self.max_chars:
            cut = self._cut(piece)
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        return out + ([piece] if piece else [])
