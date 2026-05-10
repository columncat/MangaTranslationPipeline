"""Placeholder for the llama.cpp backend (filled in by v1.1 [6/6])."""
from __future__ import annotations

from .base import BackendUnavailable, TranslatorConfig


class LlamaCppTranslator:
    def __init__(self, cfg: TranslatorConfig):  # noqa: ARG002
        raise BackendUnavailable("llama.cpp backend not implemented yet.")

    def translate_batch(self, *_a, **_kw):  # pragma: no cover
        raise BackendUnavailable("llama.cpp backend not implemented yet.")
