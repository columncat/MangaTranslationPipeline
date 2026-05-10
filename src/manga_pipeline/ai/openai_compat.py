"""OpenAI-compatible chat-completions backend.

Works with anything that speaks the ``/v1/chat/completions`` schema:
OpenAI itself, OpenRouter, vLLM with the OpenAI server, LM Studio, and
many other inference servers. The user picks the model and (optional)
base URL from the GUI.

Uses the official ``openai`` Python SDK; if it's not installed we raise
:class:`BackendUnavailable` with an actionable message.
"""
from __future__ import annotations

import json

from .base import (
    BackendUnavailable,
    TranslatorConfig,
    build_system_prompt,
    parse_batch_response,
)


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

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # ``response_format=json_object`` is supported by GPT-4 family
            # and some OSS models; for those that ignore it the parser
            # still copes thanks to the JSON-extraction fallback in base.
            response_format={"type": "json_object"},
        )
        choice = resp.choices[0]
        return choice.message.content or ""

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
