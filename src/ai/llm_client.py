"""
src/ai/llm_client.py

One small wrapper around all LLM providers. The rest of InsightIQ only ever calls

    LLMClient().generate(system, messages)

and never knows which provider is behind it. Provider and model come from .env.

Supported: nvidia (NVIDIA NIM, OpenAI-compatible), openai, anthropic.

Self-check (sends one tiny request):   python -m src.ai.llm_client
"""
from __future__ import annotations

import logging
import re
import time

from config.settings import get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_NAMES = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
MAX_RETRIES = 3
BACKOFF_SECONDS = 2  # waits 2s, 4s, 8s between retries

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    """Any problem talking to the LLM. Message is safe to show to the user (never contains keys)."""


def strip_thinking(text: str) -> str:
    """Remove inline <think>...</think> blocks some reasoning models put in their answer."""
    return _THINK_RE.sub("", text).strip()


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in RETRYABLE_STATUS or type(exc).__name__ in RETRYABLE_NAMES


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self.provider = s.llm_provider
        self.model = s.llm_model
        self._timeout = s.llm_timeout_seconds
        self._settings = s
        self._client = None  # created on first use

    # ---------- public ----------
    def generate(self, system: str, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 4096) -> str:
        """Return the model's answer text. `messages` = [{"role": "user"|"assistant", "content": str}, ...]."""
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                text = self._call(system, messages, temperature, max_tokens)
                return self._finalize(text)
            except LLMError:
                raise
            except Exception as exc:  # provider SDK errors
                last_exc = exc
                if _is_retryable(exc) and attempt < MAX_RETRIES:
                    wait = BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning("LLM call failed (%s); retrying in %ss", type(exc).__name__, wait)
                    time.sleep(wait)
                    continue
                break
        raise LLMError(self._friendly(last_exc)) from None

    # ---------- internals ----------
    def _call(self, system, messages, temperature, max_tokens) -> str:
        if self.provider in ("nvidia", "openai"):
            return self._call_openai_style(system, messages, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._call_anthropic(system, messages, temperature, max_tokens)
        raise LLMError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _call_openai_style(self, system, messages, temperature, max_tokens) -> str:
        if self._client is None:
            from openai import OpenAI  # imported lazily so other providers don't need it
            if self.provider == "nvidia":
                key = self._settings.nvidia_api_key.get_secret_value()
                if not key:
                    raise LLMError("NVIDIA_API_KEY is empty. Add it to your .env file.")
                self._client = OpenAI(base_url=self._settings.nvidia_base_url, api_key=key,
                                      timeout=self._timeout, max_retries=0)
            else:
                key = self._settings.openai_api_key.get_secret_value()
                if not key:
                    raise LLMError("OPENAI_API_KEY is empty. Add it to your .env file.")
                self._client = OpenAI(api_key=key, timeout=self._timeout, max_retries=0)

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Only the final answer (`content`). Reasoning models keep their thinking in a
        # separate field, which we deliberately ignore.
        content = resp.choices[0].message.content
        if not content:
            raise LLMError("The model returned an empty answer (it may have spent all tokens "
                           "thinking). Try a larger max_tokens or a different model.")
        return content

    def _call_anthropic(self, system, messages, temperature, max_tokens) -> str:
        if self._client is None:
            import anthropic
            key = self._settings.anthropic_api_key.get_secret_value()
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is empty. Add it to your .env file.")
            self._client = anthropic.Anthropic(api_key=key, timeout=self._timeout, max_retries=0)
        resp = self._client.messages.create(model=self.model, system=system, messages=messages,
                                            temperature=temperature, max_tokens=max_tokens)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if not text:
            raise LLMError("The model returned an empty answer.")
        return text

    @staticmethod
    def _finalize(text: str) -> str:
        cleaned = strip_thinking(text)
        if not cleaned:
            raise LLMError("The model's answer was empty after removing its thinking text.")
        return cleaned

    @staticmethod
    def _friendly(exc: Exception | None) -> str:
        status = getattr(exc, "status_code", None)
        hints = {
            401: "Authentication failed: check your API key in .env.",
            403: "Access denied for this model or key.",
            404: "Model not found: check LLM_MODEL in .env (copy the exact ID from the model page).",
            429: "Rate limit reached even after retries. Wait a minute and try again.",
        }
        if status in hints:
            return f"LLM error ({status}): {hints[status]}"
        return f"LLM request failed ({type(exc).__name__}): {str(exc)[:200]}"


def _self_check() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = LLMClient()
    print(f"Provider: {client.provider}\nModel:    {client.model}")
    start = time.time()
    try:
        answer = client.generate("You are a connectivity test.",
                                 [{"role": "user", "content": "Reply with exactly one word: OK"}],
                                 max_tokens=2048)
    except LLMError as e:
        print(f"\nFAILED: {e}")
        return 1
    print(f"\nModel answered: {answer!r}  ({time.time() - start:.1f}s)\nLLM connection works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())