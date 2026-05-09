from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import Step1Params, Step5Params
from ..device import auto_device, empty_cuda_cache
from ..ml.inpainter import Inpainter
from ..models import BBox, PageContext, TranslationResult
from ..utils.fonts import find_default_font, load_font
from ..utils.image_io import alpha_composite
from .base import PipelineStep, ProgressCallback, StepResult


def build_inpaint_mask(
    shape_hw: tuple[int, int],
    translations: list[TranslationResult],
    *,
    base_mask: Optional[np.ndarray] = None,
    dilate_px: int = 7,
) -> np.ndarray:
    """Mask for inpainting — limited to the bboxes of currently-kept translations.

    If ``base_mask`` (the precise text mask) is provided, the result is
    ``base_mask AND bbox_union`` so unrelated detections outside translated
    bboxes are spared. Otherwise the bbox rectangles are used directly.
    """
    h, w = shape_hw
    bbox_union = np.zeros((h, w), dtype=np.uint8)
    for tr in translations:
        b = tr.bbox
        cv2.rectangle(bbox_union, (b.x, b.y), (b.x + b.w, b.y + b.h), 255, -1)

    if base_mask is not None and base_mask.shape[:2] == (h, w):
        result = cv2.bitwise_and(base_mask, bbox_union)
    else:
        result = bbox_union

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        result = cv2.dilate(result, k, iterations=1)
    return result


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _block_size(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float,
) -> tuple[int, int]:
    if not lines:
        return 0, 0
    widths = [_measure(draw, ln or " ", font)[0] for ln in lines]
    line_h = font.size
    total_h = int(line_h * line_spacing * (len(lines) - 1)) + line_h
    return max(widths), total_h


def _render_text_block_to_buffer(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    params: Step5Params,
    align: str,
) -> Image.Image:
    """Render multi-line text onto a transparent RGBA image.

    The block's geometric center is the natural anchor — callers paste the
    buffer so its center lands on the desired (x, y) on the final image.
    Padding around the buffer prevents stroke clipping when rotated.
    """
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    if not lines:
        lines = [""]
    line_widths = [_measure(measure, ln or " ", font)[0] for ln in lines]
    block_w = max(line_widths) if line_widths else 0
    line_h = font.size
    block_h = int(line_h * params.line_spacing * (len(lines) - 1)) + line_h

    pad = max(2, int(params.stroke_px) + 2)
    buf = Image.new(
        "RGBA",
        (max(1, block_w + 2 * pad), max(1, block_h + 2 * pad)),
        (0, 0, 0, 0),
    )
    bdraw = ImageDraw.Draw(buf)
    for i, ln in enumerate(lines):
        line_w = line_widths[i]
        if align == "left":
            lx = pad
        elif align == "right":
            lx = pad + (block_w - line_w)
        else:  # center (default)
            lx = pad + (block_w - line_w) // 2
        ly = pad + int(i * line_h * params.line_spacing)
        bdraw.text(
            (lx, ly),
            ln,
            font=font,
            fill=params.fill_rgb,
            stroke_width=params.stroke_px,
            stroke_fill=params.stroke_rgb,
        )
    return buf


class KoreanRenderer:
    def __init__(self, font_path: Optional[Path] = None):
        if font_path is None or not Path(font_path).exists():
            font_path = find_default_font()
        if font_path is None:
            raise RuntimeError(
                "No Korean-capable font found. Place a TTF/OTF in fonts/ or set Step5Params.font_path."
            )
        self.font_path = Path(font_path)

    def render(
        self,
        cleaned_rgb: np.ndarray,
        items: list[TranslationResult],
        params: Step5Params,
    ) -> np.ndarray:
        """Render every translation centered on its bbox.

        Translations are always treated as ``ignore_boundary``: lines split
        only on user-inserted ``\\n`` and the block is centered on the bbox.
        """
        h, w = cleaned_rgb.shape[:2]
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        for item in items:
            if not item.text_ko:
                continue
            font_path = self._resolve_font_path(item.font_path)
            ox = int(getattr(item, "text_offset_x", 0) or 0)
            oy = int(getattr(item, "text_offset_y", 0) or 0)
            align = (getattr(item, "text_align", "center") or "center").lower()
            rotation = int(getattr(item, "text_rotation", 0) or 0)

            pt = (
                int(item.font_pt)
                if (item.font_pt and item.font_pt > 0)
                else int(params.outside_pt)
            )
            font = load_font(font_path, max(6, pt))
            lines = item.text_ko.split("\n")
            cx = item.bbox.x + item.bbox.w // 2 + ox
            cy = item.bbox.y + item.bbox.h // 2 + oy

            buf = _render_text_block_to_buffer(lines, font, params, align)
            if rotation:
                buf = buf.rotate(
                    rotation, expand=True, resample=Image.BICUBIC
                )
            paste_x = int(cx - buf.width / 2)
            paste_y = int(cy - buf.height / 2)
            layer.alpha_composite(buf, (paste_x, paste_y))

        overlay = np.asarray(layer)
        return alpha_composite(cleaned_rgb, overlay)

    def _resolve_font_path(self, override: Optional[str]) -> Path:
        if override:
            p = Path(override)
            if p.exists():
                return p
        return self.font_path


class Step5Render(PipelineStep):
    """Inpaint text regions then render Korean translations on top."""

    name = "step5_render"

    def __init__(self, font_path: Optional[Path] = None, device: Optional[str] = None):
        self._renderer: Optional[KoreanRenderer] = None
        self._font_hint = font_path
        self._inpainter: Optional[Inpainter] = None
        self._device = device or auto_device()

    def _renderer_obj(self, params: Step5Params) -> KoreanRenderer:
        path = Path(params.font_path) if params.font_path else self._font_hint
        if self._renderer is None or (path is not None and Path(self._renderer.font_path) != path):
            self._renderer = KoreanRenderer(font_path=path)
        return self._renderer

    def _inpainter_obj(self) -> Inpainter:
        if self._inpainter is None:
            self._inpainter = Inpainter(device=self._device)
        return self._inpainter

    def run(
        self,
        ctx: PageContext,
        params: Step5Params,
        progress: Optional[ProgressCallback] = None,
    ) -> StepResult:
        if ctx.source is None:
            return StepResult(ok=False, message="source image missing")
        if not ctx.translations:
            return StepResult(ok=False, message="no translations — run Step 4 first")

        try:
            renderer = self._renderer_obj(params)
        except RuntimeError as e:
            return StepResult(ok=False, message=str(e), error=e)

        try:
            inpaint_dilate = getattr(ctx, "_inpaint_dilate_px", None)
            if inpaint_dilate is None:
                inpaint_dilate = Step1Params().inpaint_dilate_px

            if ctx.cleaned is None:
                if progress:
                    progress(0, 2, "inpainting text regions")
                inpaint_mask = build_inpaint_mask(
                    ctx.source.shape[:2],
                    ctx.translations,
                    base_mask=ctx.mask,
                    dilate_px=int(inpaint_dilate),
                )
                ctx.cleaned = self._inpainter_obj().inpaint(
                    ctx.source.copy(), inpaint_mask, dilate_px=0
                )

            if progress:
                progress(1, 2, "rendering Korean text")
            ctx.final = renderer.render(ctx.cleaned, ctx.translations, params)
            if progress:
                progress(2, 2, "done")
            return StepResult(ok=True, message="cleaned + final image generated")

        except Exception as e:  # noqa: BLE001
            empty_cuda_cache()
            return StepResult(ok=False, message=str(e), error=e)
