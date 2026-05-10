"""Google Gemini backend.

Uses the new ``google-genai`` SDK (the ``google-generativeai`` package
is the older surface — both ship under the same namespace; we prefer
the unified one). Asks Gemini to return strict JSON via
``response_mime_type='application/json'``.
"""
from __future__ import annotations

import json

from .base import (
    BackendUnavailable,
    TranslatorConfig,
    build_system_prompt,
    parse_batch_response,
)


class GeminiTranslator:
    def __init__(self, cfg: TranslatorConfig):
        try:
            # The unified ``google-genai`` SDK (preferred).
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise BackendUnavailable(
                "The 'google-genai' package is not installed. Run "
                "`pip install google-genai`."
            ) from e
        if not cfg.api_key:
            raise BackendUnavailable("Google AI Studio API key is not set.")
        self._client = genai.Client(api_key=cfg.api_key)
        self._types = genai_types
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

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=cfg,
        )
        return getattr(resp, "text", "") or ""

    def _batch_call(
        self, lines: list[str], glossary: str, style: str
    ) -> list[str]:
        system = build_system_prompt(glossary, style)
        payload = [{"id": i, "ja": line} for i, line in enumerate(lines)]
        user = json.dumps(payload, ensure_ascii=False)
        text = self._generate(system, user, self._max_tokens)
        return parse_batch_response(text, len(lines))

    def _single_call(self, line: str, glossary: str, style: str) -> str:
        if not line.strip():
            return ""
        system = build_system_prompt(glossary, style)
        user = json.dumps([{"id": 0, "ja": line}], ensure_ascii=False)
        text = self._generate(system, user, 512)
        out = parse_batch_response(text, 1)
        return out[0] if out and out[0] else line
