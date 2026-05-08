from __future__ import annotations

import json
import re
from typing import Optional

from ..config import Step4Params
from ..models import OcrResult, PageContext, TranslationResult
from ..utils.secrets import get_anthropic_key
from .base import PipelineStep, ProgressCallback, StepResult

SYSTEM_TEMPLATE = """You are a professional Japanese-to-Korean manga translator.
Translate each line into natural Korean dialogue suitable for the speech bubble.
Preserve tone, honorifics, sound effects, and character voice.
Because Korean and Japanese have very similar sentence structures, translate based on meaning while keeping the sentence structure as close to a literal translation as possible.

Insert newline characters (`\n`) to fit the text into vertical manga speech bubbles by following the rules below.
1. Keep lines balanced in length, typically 4-6 Korean characters per line, except spacing and punctuation.
2. Break lines at natural spaces (boundaries). Avoid to split a single word in half.
3. Write in 2 lines (single newline character) when text is around 10 characters, 3 lines if under 20 characters.

Do not add commentary, romanization, or notes — only the Korean translation.

Glossary (use these exact translations when the source matches):
{glossary}

Style notes: {style_notes}

Output strictly as JSON: {{"translations":[{{"id":<int>,"ko":"<korean>"}}, ...]}}.
Include every input id exactly once. No prose outside JSON."""


def _build_system_prompt(glossary: str, style: str) -> str:
    g = glossary.strip() or "(none)"
    s = style.strip() or "(none)"
    return SYSTEM_TEMPLATE.format(glossary=g, style_notes=s)


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
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


class ClaudeTranslator:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6", max_tokens: int = 4096):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def translate_batch(
        self,
        lines: list[str],
        glossary: str = "",
        style: str = "",
    ) -> list[str]:
        if not lines:
            return []

        system_text = _build_system_prompt(glossary, style)
        system = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        payload = [{"id": i, "ja": line} for i, line in enumerate(lines)]
        user = json.dumps(payload, ensure_ascii=False)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)

        data = _extract_json(text)
        out = [""] * len(lines)
        if data and isinstance(data.get("translations"), list):
            for item in data["translations"]:
                idx = item.get("id")
                ko = item.get("ko", "")
                if isinstance(idx, int) and 0 <= idx < len(lines):
                    out[idx] = str(ko).strip()

        for i, ko in enumerate(out):
            if not ko:
                out[i] = self.translate_one(lines[i], glossary, style)
        return out

    def translate_one(self, line: str, glossary: str = "", style: str = "") -> str:
        if not line.strip():
            return ""
        system_text = _build_system_prompt(glossary, style)
        resp = self.client.messages.create(
            model=self.model,
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
        data = _extract_json(text)
        if data and data.get("translations"):
            ko = data["translations"][0].get("ko", "")
            return str(ko).strip()
        return line


class Step4Translate(PipelineStep):
    name = "step4_translate"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_anthropic_key()

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

        if not self.api_key:
            return StepResult(ok=False, message="ANTHROPIC_API_KEY not set")

        if progress:
            progress(0, len(ctx.ocr), "calling Claude")

        translator = ClaudeTranslator(
            api_key=self.api_key, model=params.model, max_tokens=params.max_tokens
        )

        lines = [r.text_ja for r in ctx.ocr]
        try:
            ko_lines = translator.translate_batch(lines, params.glossary, params.style_notes)
        except Exception as e:  # noqa: BLE001
            return StepResult(ok=False, message=f"translation failed: {e}", error=e)

        ctx.translations = [
            TranslationResult(bbox=r.bbox, text_ja=r.text_ja, text_ko=ko)
            for r, ko in zip(ctx.ocr, ko_lines)
        ]
        if progress:
            progress(len(ctx.ocr), len(ctx.ocr), "translation done")
        return StepResult(ok=True, message=f"{len(ctx.translations)} translated")


__all__ = [
    "ClaudeTranslator",
    "Step4Translate",
    "_build_system_prompt",
    "_extract_json",
]
