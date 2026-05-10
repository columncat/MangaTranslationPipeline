"""Anthropic (Claude) translator backend.

Module is named ``anthropic_backend`` instead of ``anthropic`` so it
doesn't shadow the official ``anthropic`` PyPI package on local imports.
"""
from __future__ import annotations

import json

from .base import (
    BackendUnavailable,
    TranslatorConfig,
    build_system_prompt,
    parse_batch_response,
)


class AnthropicTranslator:
    def __init__(self, cfg: TranslatorConfig):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise BackendUnavailable(
                "The 'anthropic' package is not installed. Run "
                "`pip install anthropic`."
            ) from e
        if not cfg.api_key:
            raise BackendUnavailable("Anthropic API key is not set.")
        self._client = Anthropic(api_key=cfg.api_key)
        self._model = cfg.model
        self._max_tokens = cfg.max_tokens

    # ---- Translator protocol ----

    def translate_batch(
        self,
        lines: list[str],
        glossary: str = "",
        style: str = "",
    ) -> list[str]:
        if not lines:
            return []
        out = self._batch_call(lines, glossary, style)
        # Single-line fallback for any item the batch missed.
        for i, ko in enumerate(out):
            if not ko:
                out[i] = self._single_call(lines[i], glossary, style)
        return out

    # ---- internals ----

    def _batch_call(
        self, lines: list[str], glossary: str, style: str
    ) -> list[str]:
        system_text = build_system_prompt(glossary, style)
        # Cache the (large, repeated) system prompt to keep follow-up
        # batches cheap; the user payload is small and uncached.
        system = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        payload = [{"id": i, "ja": line} for i, line in enumerate(lines)]
        user = json.dumps(payload, ensure_ascii=False)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return parse_batch_response(text, len(lines))

    def _single_call(self, line: str, glossary: str, style: str) -> str:
        if not line.strip():
            return ""
        system_text = build_system_prompt(glossary, style)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        [{"id": 0, "ja": line}], ensure_ascii=False
                    ),
                }
            ],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        out = parse_batch_response(text, 1)
        return out[0] if out and out[0] else line
