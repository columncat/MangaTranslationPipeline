"""Placeholder for the Ollama backend (filled in by v1.1 [6/6])."""
from __future__ import annotations

from .base import BackendUnavailable, TranslatorConfig


class OllamaTranslator:
    def __init__(self, cfg: TranslatorConfig):  # noqa: ARG002
        raise BackendUnavailable("Ollama backend not implemented yet.")

    def translate_batch(self, *_a, **_kw):  # pragma: no cover
        raise BackendUnavailable("Ollama backend not implemented yet.")
