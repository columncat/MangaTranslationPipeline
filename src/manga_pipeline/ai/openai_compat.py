"""Placeholder for the OpenAI-compatible backend.

Filled in by v1.1 [6/6]. Until then, selecting this provider raises
:class:`BackendUnavailable` so the GUI can show a clear message.
"""
from __future__ import annotations

from .base import BackendUnavailable, TranslatorConfig


class OpenAICompatTranslator:
    def __init__(self, cfg: TranslatorConfig):  # noqa: ARG002
        raise BackendUnavailable("OpenAI-compatible backend not implemented yet.")

    def translate_batch(self, *_a, **_kw):  # pragma: no cover - unreachable
        raise BackendUnavailable("OpenAI-compatible backend not implemented yet.")
