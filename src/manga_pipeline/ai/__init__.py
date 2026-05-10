"""AI provider abstraction layer.

The :class:`Translator` protocol describes what a JA→KO translation
backend must offer. Concrete backends live in their own submodules
(``anthropic_backend``, ``openai_compat``, ``gemini``, ``ollama``,
``llamacpp``) and are constructed lazily through :func:`make_translator`
so that optional SDKs are imported only when actually needed.
"""

from .base import (
    AIProvider,
    BackendUnavailable,
    PROVIDERS,
    Translator,
    TranslatorConfig,
    build_system_prompt,
    extract_translation_json,
    make_translator,
)

__all__ = [
    "AIProvider",
    "BackendUnavailable",
    "PROVIDERS",
    "Translator",
    "TranslatorConfig",
    "build_system_prompt",
    "extract_translation_json",
    "make_translator",
]
