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
    # Auto-managed local backend that downloads Gemma 4 E4B-it Q4_K_M
    # on first use and runs it through llama-cpp-python.
    EMBEDDED = "embedded"


PROVIDERS: tuple[str, ...] = (
    AIProvider.ANTHROPIC,
    AIProvider.OPENAI_COMPAT,
    AIProvider.GEMINI,
    AIProvider.OLLAMA,
    AIProvider.LLAMACPP,
    AIProvider.EMBEDDED,
)


# ---- prompt + parser shared by every backend ----

SYSTEM_TEMPLATE = """You are a professional Japanese-to-Korean manga translator.
Translate each line into natural Korean dialogue suitable for the speech bubble.
Preserve tone, honorifics, sound effects, and character voice.
Because Korean and Japanese have very similar sentence structures, translate based on meaning while keeping the sentence structure as close to a literal translation as possible.

Insert newline characters (`\\n`) to fit the text into vertical manga speech bubbles by following the rules below.
1. Keep lines balanced in length, typically 4-6 Korean characters per line, except spacing and punctuation.
2. Break lines at natural spaces (boundaries). Avoid to split a single word in half.
3. Write in 2 lines (single newline character) when the result is around 10 Korean characters, 3 lines for ~15 characters, 4+ lines beyond that.
4. Very short lines (≤6 characters) stay on a single line — do NOT add a newline.
5. Count Korean characters (e.g. 안녕하세요 = 5 characters), not Japanese.

Do not add commentary, romanization, or notes — only the Korean translation.

Glossary (use these exact translations when the source matches):
{glossary}

Style notes: {style_notes}

Output strictly as JSON: {{"translations":[{{"id":<int>,"ko":"<korean>"}}, ...]}}.
Include every input id exactly once. No prose outside JSON.

Examples (study the line-break placement carefully — these are the
expected outputs given the inputs, not extra inputs to translate):

Input:  [{{"id":0,"ja":"ありがとう"}},{{"id":1,"ja":"今日は本当にお疲れ様でした"}},{{"id":2,"ja":"明日の朝早く必ず連絡してください"}},{{"id":3,"ja":"うん"}}]
Output: {{"translations":[{{"id":0,"ko":"고마워"}},{{"id":1,"ko":"오늘 정말\\n수고하셨어요"}},{{"id":2,"ko":"내일 아침\\n일찍 꼭\\n연락해주세요"}},{{"id":3,"ko":"응"}}]}}

Input:  [{{"id":0,"ja":"何だこいつ"}},{{"id":1,"ja":"そんなはずがないだろう"}},{{"id":2,"ja":"待ってくれ俺の話を聞いてくれよ頼むから"}}]
Output: {{"translations":[{{"id":0,"ko":"뭐야 이 녀석"}},{{"id":1,"ko":"그럴 리가\\n없잖아"}},{{"id":2,"ko":"기다려 줘\\n내 얘기 좀\\n들어줘 제발"}}]}}"""


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
    if p == AIProvider.EMBEDDED:
        from .embedded import EmbeddedTranslator

        return EmbeddedTranslator(cfg)
    raise BackendUnavailable(f"unknown provider {p!r}")
