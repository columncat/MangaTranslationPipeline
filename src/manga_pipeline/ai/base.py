"""Translator interface + shared helpers.

The pipeline (Step 4) talks only to a :class:`Translator` instance —
``translate_batch`` returns Korean strings aligned positionally with the
input JA strings. Each concrete backend implements this and is selected
at runtime through :func:`make_translator`.

Adding a new backend means:
1. Subclass :class:`Translator`.
2. Register a string id in :data:`AIProvider`.
3. Add a branch to :func:`make_translator`.
4. Optionally extend :class:`TranslatorConfig` with backend-specific
   fields (e.g. base URL, model name, temperature).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

# ---- enum-ish provider id ----

class AIProvider:
    ANTHROPIC = "anthropic"
    OPENAI_COMPAT = "openai_compat"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"


PROVIDERS: tuple[str, ...] = (
    AIProvider.ANTHROPIC,
    AIProvider.OPENAI_COMPAT,
    AIProvider.GEMINI,
    AIProvider.OLLAMA,
    AIProvider.LLAMACPP,
)


# ---- prompt + parser shared by every backend ----

SYSTEM_TEMPLATE = """You are a professional Japanese-to-Korean manga translator.
Translate each line into natural Korean dialogue suitable for the speech bubble.
Preserve tone, honorifics, sound effects, and character voice.
Because Korean and Japanese have very similar sentence structures, translate based on meaning while keeping the sentence structure as close to a literal translation as possible.

Insert newline characters (`\\n`) to fit the text into vertical manga speech bubbles by following the rules below.
1. Keep lines balanced in length, typically 4-6 Korean characters per line, except spacing and punctuation.
2. Break lines at natural spaces (boundaries). Avoid to split a single word in half.
3. Write in 2 lines (single newline character) when text is around 10 characters, 3 lines if under 20 characters.

Do not add commentary, romanization, or notes — only the Korean translation.

Glossary (use these exact translations when the source matches):
{glossary}

Style notes: {style_notes}

Output strictly as JSON: {{"translations":[{{"id":<int>,"ko":"<korean>"}}, ...]}}.
Include every input id exactly once. No prose outside JSON."""


def build_system_prompt(glossary: str, style: str) -> str:
    g = (glossary or "").strip() or "(none)"
    s = (style or "").strip() or "(none)"
    return SYSTEM_TEMPLATE.format(glossary=g, style_notes=s)


def extract_translation_json(text: str) -> Optional[dict]:
    """Best-effort JSON parser.

    LLMs that don't strictly obey "JSON only" often wrap the payload in
    code fences or add a sentence of preamble. Try the literal text first,
    then fall back to the largest ``{...}`` substring.
    """
    text = (text or "").strip()
    # Strip Markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        # Drop a leading "json" language tag if any.
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_batch_response(text: str, n_lines: int) -> list[str]:
    """Convert the raw model output into ``n_lines`` aligned KO strings.

    Missing ids are returned as empty strings — callers usually retry
    them via single-line fallback calls.
    """
    out = [""] * n_lines
    data = extract_translation_json(text)
    if not data or not isinstance(data.get("translations"), list):
        return out
    for item in data["translations"]:
        idx = item.get("id")
        ko = item.get("ko", "")
        if isinstance(idx, int) and 0 <= idx < n_lines:
            out[idx] = str(ko).strip()
    return out


# ---- config / interface ----


@dataclass
class TranslatorConfig:
    """Backend-agnostic translator configuration.

    Each backend uses only the subset of fields that applies to it; the
    rest are ignored. Persisted into ``AppConfig.step4`` (and migrated
    from the v1.0 single-Anthropic schema by :mod:`config`).
    """

    provider: str = AIProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    # Cloud APIs key (Anthropic, OpenAI-compat, Gemini). Ignored for
    # local backends (Ollama, llama.cpp).
    api_key: Optional[str] = None
    # Base URL override:
    # - openai_compat: full base URL up to /v1 (e.g. http://localhost:8000/v1)
    # - ollama: e.g. http://localhost:11434
    # - others: ignored
    base_url: Optional[str] = None
    # Local-model file path (llama.cpp). Ignored elsewhere.
    model_path: Optional[str] = None
    # n_ctx and n_gpu_layers for llama.cpp.
    n_ctx: int = 8192
    n_gpu_layers: int = -1  # -1 = offload all layers to GPU if possible


class Translator(Protocol):
    """Protocol for translation backends."""

    def translate_batch(
        self,
        lines: list[str],
        glossary: str = "",
        style: str = "",
    ) -> list[str]:
        ...


class BackendUnavailable(RuntimeError):
    """Raised when a backend's optional SDK / runtime isn't installed."""


def make_translator(cfg: TranslatorConfig) -> Translator:
    """Construct a translator for ``cfg.provider``.

    Raises :class:`BackendUnavailable` (with an actionable message) when
    the backend's optional dependency hasn't been installed yet.
    """
    p = cfg.provider
    if p == AIProvider.ANTHROPIC:
        from .anthropic_backend import AnthropicTranslator

        return AnthropicTranslator(cfg)
    if p == AIProvider.OPENAI_COMPAT:
        from .openai_compat import OpenAICompatTranslator

        return OpenAICompatTranslator(cfg)
    if p == AIProvider.GEMINI:
        from .gemini import GeminiTranslator

        return GeminiTranslator(cfg)
    if p == AIProvider.OLLAMA:
        from .ollama import OllamaTranslator

        return OllamaTranslator(cfg)
    if p == AIProvider.LLAMACPP:
        from .llamacpp import LlamaCppTranslator

        return LlamaCppTranslator(cfg)
    raise BackendUnavailable(f"unknown provider {p!r}")
