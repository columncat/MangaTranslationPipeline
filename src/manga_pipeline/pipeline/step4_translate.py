"""Step 4 — translate Japanese OCR text into Korean.

The actual translation work is delegated to a :class:`Translator`
instance built by :func:`manga_pipeline.ai.make_translator`, so the
pipeline doesn't care whether the backend is Anthropic, OpenAI-
compatible, Ollama, or anything else — it just hands over the JA
strings and receives KO strings back in the same order.
"""
from __future__ import annotations

from typing import Optional

from ..ai import (
    AIProvider,
    BackendUnavailable,
    TranslatorConfig,
    make_translator,
)

# Re-exported for tests and any v1.0-era callers that imported the
# helpers from this module directly.
from ..ai.base import (
    build_system_prompt as _build_system_prompt,
    extract_translation_json as _extract_json,
    parse_batch_response,
)
from ..config import Step4Params
from ..models import PageContext, TranslationResult
from ..utils.secrets import get_anthropic_key
from .base import PipelineStep, ProgressCallback, StepResult


def _config_from_params(params: Step4Params) -> TranslatorConfig:
    """Map :class:`Step4Params` → backend-agnostic :class:`TranslatorConfig`.

    The Anthropic API key is sourced from the OS keyring / env var when
    the chosen provider needs one.
    """
    api_key = None
    if params.provider == AIProvider.ANTHROPIC:
        api_key = get_anthropic_key()
    return TranslatorConfig(
        provider=params.provider,
        model=params.model,
        max_tokens=params.max_tokens,
        api_key=api_key,
        base_url=params.base_url,
        model_path=params.model_path,
        n_ctx=params.n_ctx,
        n_gpu_layers=params.n_gpu_layers,
    )


class Step4Translate(PipelineStep):
    name = "step4_translate"

    def __init__(self, api_key: Optional[str] = None):
        # ``api_key`` is kept as a constructor argument for backwards
        # compatibility with v1.0 callers; it's only consulted when the
        # configured provider is Anthropic and no key is found via the
        # keyring fallback.
        self._explicit_api_key = api_key

    def run(
        self,
        ctx: PageContext,
        params: Step4Params,
        progress: Optional[ProgressCallback] = None,
    ) -> StepResult:
        if not ctx.ocr:
            return StepResult(ok=False, message="no OCR results — run Detect/OCR first")

        # Bypass: copy OCR text straight into ko, skipping the API call entirely.
        if params.skip_translation:
            if progress:
                progress(0, len(ctx.ocr), "skip translation — using JA as KO")
            ctx.translations = [
                TranslationResult(bbox=r.bbox, text_ja=r.text_ja, text_ko=r.text_ja)
                for r in ctx.ocr
            ]
            if progress:
                progress(len(ctx.ocr), len(ctx.ocr), "skipped")
            return StepResult(
                ok=True, message=f"{len(ctx.translations)} kept as JA (skip on)"
            )

        cfg = _config_from_params(params)
        # If the user picked Anthropic but the keyring lookup failed,
        # honour the explicit key passed to the step constructor (v1.0).
        if cfg.provider == AIProvider.ANTHROPIC and not cfg.api_key:
            cfg.api_key = self._explicit_api_key
        if cfg.provider == AIProvider.ANTHROPIC and not cfg.api_key:
            return StepResult(ok=False, message="ANTHROPIC_API_KEY not set")

        if progress:
            progress(0, len(ctx.ocr), f"calling {cfg.provider}")

        try:
            translator = make_translator(cfg)
        except BackendUnavailable as e:
            return StepResult(ok=False, message=f"backend unavailable: {e}", error=e)

        lines = [r.text_ja for r in ctx.ocr]
        try:
            ko_lines = translator.translate_batch(
                lines, params.glossary, params.style_notes
            )
        except BackendUnavailable as e:
            return StepResult(ok=False, message=f"backend unavailable: {e}", error=e)
        except Exception as e:  # noqa: BLE001
            return StepResult(ok=False, message=f"translation failed: {e}", error=e)

        ctx.translations = [
            TranslationResult(bbox=r.bbox, text_ja=r.text_ja, text_ko=ko)
            for r, ko in zip(ctx.ocr, ko_lines)
        ]
        if progress:
            progress(len(ctx.ocr), len(ctx.ocr), "translation done")
        return StepResult(ok=True, message=f"{len(ctx.translations)} translated")


# Backwards-compatible re-export so existing tests that import
# ``ClaudeTranslator`` from this module keep working.
def ClaudeTranslator(*args, **kwargs):  # noqa: N802 — keep legacy name
    """Shim: build the new AnthropicTranslator from the v1.0 call signature.

    v1.0 callers used ``ClaudeTranslator(api_key=..., model=..., max_tokens=...)``.
    """
    from ..ai.anthropic_backend import AnthropicTranslator

    cfg = TranslatorConfig(
        provider=AIProvider.ANTHROPIC,
        api_key=kwargs.get("api_key") or (args[0] if args else None),
        model=kwargs.get("model", "claude-sonnet-4-6"),
        max_tokens=kwargs.get("max_tokens", 4096),
    )
    return AnthropicTranslator(cfg)


__all__ = [
    "ClaudeTranslator",
    "Step4Translate",
    "_build_system_prompt",
    "_extract_json",
    "parse_batch_response",
]
