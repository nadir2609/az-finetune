"""Streaming Gemini chat client that keeps the conversation history."""
import time


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    pass


TIMEOUT_MS = 20000
RETRY_CODES = (500, 502, 503, 504)  # "model is overloaded" and the like: usually gone a moment later
RETRY_DELAY_S = 1.0


class GeminiLLM:
    def __init__(self, model, system_prompt, max_turns=10, max_output_tokens=300, thinking_budget=0, client=None,
                 sleep=time.sleep):
        if client is None:
            from google import genai
            # reads GEMINI_API_KEY; the SDK's default is no timeout, and a stalled call would hang the turn
            client = genai.Client(http_options={"timeout": TIMEOUT_MS})
        self.client, self.model, self.max_turns, self.sleep = client, model, max_turns, sleep
        self.config = {"system_instruction": system_prompt, "max_output_tokens": max_output_tokens,
                       "automatic_function_calling": {"disable": True}}  # no tools; also silences an SDK notice
        if thinking_budget is not None:  # null in the config: for models that reject the setting
            self.config["thinking_config"] = {"thinking_budget": thinking_budget}
        self.history = []  # [(role, text)], role is "user" or "model"

    def stream_reply(self, user_text):
        """Yield the reply's text deltas. Raises LLMError (LLMRateLimited on HTTP 429); a reply with
        no text (or only whitespace) is an LLMError too, and leaves the history as it was."""
        self.history.append(("user", user_text))
        said = []
        try:
            contents = [{"role": role, "parts": [{"text": text}]} for role, text in self.history]
            for attempt in (1, 2):
                try:
                    for chunk in self.client.models.generate_content_stream(
                            model=self.model, contents=contents, config=self.config):
                        if chunk.text:  # None for chunks without text parts
                            said.append(chunk.text)
                            yield chunk.text
                    break
                except Exception as e:  # one retry, only if nothing has been spoken yet
                    if attempt == 2 or said or getattr(e, "code", None) not in RETRY_CODES:
                        raise
                    self.sleep(RETRY_DELAY_S)
            if not "".join(said).strip():
                raise LLMError("empty reply")
        except LLMError:
            raise
        except Exception as e:
            kind = LLMRateLimited if getattr(e, "code", None) == 429 else LLMError
            raise kind(str(e)) from e
        finally:
            if "".join(said).strip():
                self.history.append(("model", "".join(said)))
                self.history = self.history[-2 * self.max_turns:]
            else:
                self.history.pop()
