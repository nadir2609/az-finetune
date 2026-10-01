"""Streaming Gemini chat client that keeps the conversation history."""


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    pass


class GeminiLLM:
    def __init__(self, model, system_prompt, max_turns=10, max_output_tokens=300, thinking_budget=0, client=None):
        if client is None:
            from google import genai
            client = genai.Client()  # reads GEMINI_API_KEY
        self.client, self.model, self.max_turns = client, model, max_turns
        self.config = {"system_instruction": system_prompt, "max_output_tokens": max_output_tokens}
        if thinking_budget is not None:  # null in the config: for models that reject the setting
            self.config["thinking_config"] = {"thinking_budget": thinking_budget}
        self.history = []  # [(role, text)], role is "user" or "model"

    def stream_reply(self, user_text):
        """Yield the reply's text deltas. Raises LLMError (LLMRateLimited on HTTP 429); a reply with
        no text at all is an LLMError too, and leaves the history as it was."""
        self.history.append(("user", user_text))
        said = []
        try:
            contents = [{"role": role, "parts": [{"text": text}]} for role, text in self.history]
            for chunk in self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=self.config):
                if chunk.text:  # None for chunks without text parts
                    said.append(chunk.text)
                    yield chunk.text
            if not said:
                raise LLMError("empty reply")
        except LLMError:
            raise
        except Exception as e:
            kind = LLMRateLimited if getattr(e, "code", None) == 429 else LLMError
            raise kind(str(e)) from e
        finally:
            if said:
                self.history.append(("model", "".join(said)))
                self.history = self.history[-2 * self.max_turns:]
            else:
                self.history.pop()
