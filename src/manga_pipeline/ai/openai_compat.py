"""OpenAI-compatible chat-completions backend.

Works with anything that speaks the ``/v1/chat/completions`` schema:
OpenAI itself, OpenRouter, vLLM with the OpenAI server, LM Studio, and
many other inference servers. The user picks the model and (optional)
base URL from the GUI.

Different servers accept different ``response_format`` shapes:

- OpenAI proper / vLLM: ``{"type": "json_object"}`` works.
- LM Studio: only ``{"type": "json_schema", ...}`` or ``{"type": "text"}``
  are accepted; ``json_object`` is rejected with a 400.
- llama.cpp server, KoboldCpp, etc.: usually ignore ``response_format``
  entirely.

We try the formats in order of preference and remember which one this
endpoint accepts so follow-up calls skip the failed attempts.
"""
from __future__ import annotations

import json
from typing import Optional

from .base import (
    BackendUnavailable,
    TranslatorConfig,
    build_system_prompt,
    parse_batch_response,
)


# JSON Schema describing the system-prompt's required output shape. Used
# whenever the server accepts ``response_format=json_schema`` (LM Studio,
# new OpenAI structured-outputs, etc.).
_TRANSLATIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "ko": {"type": "string"},
                        },
                        "required": ["id", "ko"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    },
}


def _looks_like_response_format_error(exc: Exception) -> bool:
    """Heuristic: did the server reject our response_format hint?

    We treat any 400-level error whose message mentions response_format
    or one of its option keywords as "format unsupported, try the next
    one" rather than a real failure.
    """
    msg = str(exc).lower()
    if "response_format" in msg or "response format" in msg:
        return True
    # LM Studio's exact wording: "'response_format.type' must be 'json_schema' or 'text'"
    if "json_schema" in msg or "json_object" in msg:
        return True
    return False


class OpenAICompatTranslator:
    def __init__(self, cfg: TranslatorConfig):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise BackendUnavailable(
                "The 'openai' package is not installed. Run "
                "`pip install openai`."
            ) from e

        # An API key is required by the SDK even when talking to a local
        # vLLM / LM Studio server (some accept any non-empty value); we
        # default to the placeholder used by those servers.
        api_key = cfg.api_key or "sk-no-key"
        kwargs: dict = {"api_key": api_key}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = OpenAI(**kwargs)
        self._model = cfg.model
        self._max_tokens = cfg.max_tokens
        # Remember the first response_format that this endpoint accepted
        # so we don't re-pay the trial-and-error cost on every batch.
        # Values: None = not yet tried, "json_object", "json_schema",
        # "text" (== drop the hint entirely).
        self._chosen_format: Optional[str] = None

    def translate_batch(
        self,
        lines: list[str],
        glossary: str = "",
        style: str = "",
    ) -> list[str]:
        if not lines:
            return []
        out = self._batch_call(lines, glossary, style)
        for i, ko in enumerate(out):
            if not ko:
                out[i] = self._single_call(lines[i], glossary, style)
        return out

    # ---- inference ----

    def _format_options_to_try(self) -> list[Optional[dict]]:
        """Return the list of response_format dicts to attempt this call."""
        if self._chosen_format == "json_object":
            return [{"type": "json_object"}]
        if self._chosen_format == "json_schema":
            return [_TRANSLATIONS_SCHEMA]
        if self._chosen_format == "text":
            return [None]
        # Not yet calibrated — try json_object → json_schema → no hint.
        return [{"type": "json_object"}, _TRANSLATIONS_SCHEMA, None]

    def _format_label(self, fmt: Optional[dict]) -> str:
        if fmt is None:
            return "text"
        return str(fmt.get("type", "text"))

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        common: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_exc: Optional[Exception] = None
        for fmt in self._format_options_to_try():
            kwargs = dict(common)
            if fmt is not None:
                kwargs["response_format"] = fmt
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                if _looks_like_response_format_error(e):
                    last_exc = e
                    continue
                raise
            # Success — remember the working format for next time.
            self._chosen_format = self._format_label(fmt)
            choice = resp.choices[0]
            return choice.message.content or ""

        # Every format we know of was rejected — surface the last error.
        assert last_exc is not None
        raise last_exc

    def _batch_call(
        self, lines: list[str], glossary: str, style: str
    ) -> list[str]:
        system = build_system_prompt(glossary, style)
        payload = [{"id": i, "ja": line} for i, line in enumerate(lines)]
        user = json.dumps(payload, ensure_ascii=False)
        text = self._chat(system, user, self._max_tokens)
        return parse_batch_response(text, len(lines))

    def _single_call(self, line: str, glossary: str, style: str) -> str:
        if not line.strip():
            return ""
        system = build_system_prompt(glossary, style)
        user = json.dumps([{"id": 0, "ja": line}], ensure_ascii=False)
        text = self._chat(system, user, 512)
        out = parse_batch_response(text, 1)
        return out[0] if out and out[0] else line
