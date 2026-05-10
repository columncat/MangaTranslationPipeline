"""Ollama backend.

Connects to a local Ollama server (default: ``http://localhost:11434``)
via its REST API. The user installs Ollama once, runs
``ollama pull qwen3:8b`` (or any other model) and selects the model
name from the GUI.

We use plain ``requests`` rather than the ``ollama`` Python SDK so
there's no extra optional dependency — ``requests`` is already pulled
in by the rest of the project for weight downloads.
"""
from __future__ import annotations

import json

from .base import (
    BackendUnavailable,
    TranslatorConfig,
    build_system_prompt,
    parse_batch_response,
)


DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaTranslator:
    def __init__(self, cfg: TranslatorConfig):
        try:
            import requests  # noqa: F401 — used in _chat
        except ImportError as e:
            raise BackendUnavailable(
                "The 'requests' package is not installed."
            ) from e
        self._base_url = (cfg.base_url or DEFAULT_BASE_URL).rstrip("/")
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
        import requests

        url = f"{self._base_url}/api/chat"
        body = {
            "model": self._model,
            "stream": False,
            "format": "json",  # tells Ollama to constrain output to JSON
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_predict": max_tokens,
                # Lower temperature for more deterministic translations.
                "temperature": 0.3,
            },
        }
        try:
            r = requests.post(url, json=body, timeout=300)
        except requests.exceptions.ConnectionError as e:
            raise BackendUnavailable(
                f"Could not reach Ollama at {self._base_url}. "
                "Is the server running? Try `ollama serve`."
            ) from e
        if r.status_code == 404:
            raise BackendUnavailable(
                f"Ollama model {self._model!r} not found. "
                f"Pull it first: `ollama pull {self._model}`."
            )
        r.raise_for_status()
        data = r.json()
        # /api/chat returns {"message": {"content": "..."}}
        return (data.get("message") or {}).get("content", "")

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
