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

    Per-bbox masks (``TranslationResult.bbox_mask``, an ``(h, w)`` uint8
    array in bbox-local coordinates) take precedence — pixels marked
    non-zero are pasted into the global bbox-union as inpaint targets.
    For translations without a custom mask the full bbox rectangle is
    used (the v1.0 behaviour).

    If ``base_mask`` (the precise text mask) is provided, the result is
    ``base_mask AND bbox_union`` so unrelated detections outside the
    chosen regions are spared. ``dilate_px`` is applied last so the
    inpainter has a small bleed margin around the mask.
    """
    h, w = shape_hw
    bbox_union = np.zeros((h, w), dtype=np.uint8)
    for tr in translations:
        b = tr.bbox
        custom = getattr(tr, "bbox_mask", None)
        if custom is not None and custom.shape[:2] == (b.h, b.w):
            # Composite the per-bbox mask into the page-sized canvas.
            # Clip in case the bbox falls partially out of the image.
            x0, y0 = max(0, b.x), max(0, b.y)
            x1, y1 = min(w, b.x + b.w), min(h, b.y + b.h)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = custom[y0 - b.y : y1 - b.y, x0 - b.x : x1 - b.x]
            # The mask is OR-blended so multiple overlapping bboxes
            # behave like a union.
            patch = bbox_union[y0:y1, x0:x1]
            np.maximum(patch, (sub > 0).astype(np.uint8) * 255, out=patch)
        else:
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
    *,
    fill_rgb: Optional[tuple[int, int, int]] = None,
    stroke_rgb: Optional[tuple[int, int, int]] = None,
    bg_fill_rgb: Optional[tuple[int, int, int]] = None,
    bg_fill_pad: int = 6,
    bg_border_rgb: Optional[tuple[int, int, int]] = None,
    bg_border_px: int = 0,
) -> Image.Image:
    """Render multi-line text onto a transparent RGBA image.

    The block's geometric center is the natural anchor — callers paste the
    buffer so its center lands on the desired (x, y) on the final image.
    Padding around the buffer prevents stroke clipping when rotated.

    ``fill_rgb`` / ``stroke_rgb`` override the Step5Params defaults per
    item. ``bg_fill_rgb`` (when not None) draws an opaque rectangle behind
    the text — the rectangle hugs the text's pixel bbox plus
    ``bg_fill_pad`` pixels of padding on every side.
    ``bg_border_rgb`` + ``bg_border_px`` (when both set) draw a rectangle
    outline around the same area. Border can be enabled independently of
    the fill (e.g. to draw an outline-only frame).

    Sizing strategy: each line's bounding box is measured with the stroke
    included via ``draw.textbbox(...)`` so that artistic fonts whose
    ascenders / descenders / left-bearings exceed the nominal font.size
    don't get clipped on the buffer edges. The buffer is sized to the
    union of all painted ink, plus a safety margin equal to ¼ of the
    font size (covers fonts whose embellishments still escape the
    measured bbox by a pixel or two), plus the backdrop padding and
    border thickness so the panel never gets clipped.
    """
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    if not lines:
        lines = [""]

    fill = fill_rgb if fill_rgb is not None else params.fill_rgb
    stroke = stroke_rgb if stroke_rgb is not None else params.stroke_rgb
    stroke_w = max(0, int(params.stroke_px))

    # Per-line bboxes measured WITH the stroke — Pillow textbbox returns
    # (left, top, right, bottom) where left/top can be non-zero when the
    # font has bearings that extend past the nominal origin.
    line_bboxes = [
        measure.textbbox((0, 0), ln or " ", font=font, stroke_width=stroke_w)
        for ln in lines
    ]
    line_widths = [b[2] - b[0] for b in line_bboxes]
    line_heights = [b[3] - b[1] for b in line_bboxes]
    block_w = max(line_widths) if line_widths else 0

    # Distance between successive line origins. ``font.size * line_spacing``
    # matches the v1.0 layout for the typical case where line height ==
    # font.size; for fonts where the rendered line is taller than that, we
    # extend the gap so successive lines don't overlap.
    nominal_line_h = font.size
    line_advance = max(
        int(nominal_line_h * params.line_spacing),
        max(line_heights) if line_heights else 0,
    )

    # Total ink height: distance from the top of line 0 to the bottom of
    # line N-1 (the last line's full bbox extends below its baseline by
    # `line_heights[-1] - nominal_line_h` for fonts with deep descenders).
    n = len(lines)
    if n > 0:
        block_h = (n - 1) * line_advance + line_heights[-1]
    else:
        block_h = 0

    # Outer padding around the buffer protects the stroke when the buffer
    # is later rotated, AND adds a safety margin proportional to the font
    # size for artistic fonts whose ink can still graze the measured bbox.
    safety_pad = max(2, font.size // 4)
    stroke_pad = max(safety_pad, stroke_w + 2)
    # Backdrop occupies (text bbox) + bg_fill_pad on each side. Border
    # is drawn AROUND the backdrop and grows the buffer by border width.
    has_fill = bg_fill_rgb is not None
    has_border = bg_border_rgb is not None and int(bg_border_px) > 0
    panel_pad = max(0, int(bg_fill_pad)) if (has_fill or has_border) else 0
    border_w = max(0, int(bg_border_px)) if has_border else 0
    pad = stroke_pad + panel_pad + border_w
    buf = Image.new(
        "RGBA",
        (max(1, block_w + 2 * pad), max(1, block_h + 2 * pad)),
        (0, 0, 0, 0),
    )
    bdraw = ImageDraw.Draw(buf)

    if has_fill or has_border:
        # Rectangle hugs the text bbox expanded by panel_pad on each
        # side. Pillow's ``rectangle`` includes the bottom-right pixel,
        # so subtract 1 from the bottom-right corner so the visible
        # outline width matches ``border_w`` exactly.
        rx0 = stroke_pad + border_w - 0
        ry0 = stroke_pad + border_w - 0
        rx1 = stroke_pad + border_w + 2 * panel_pad + block_w - 1
        ry1 = stroke_pad + border_w + 2 * panel_pad + block_h - 1
        # Re-anchor so the text still lands on its expected coordinates:
        # text was placed at (pad + base_x, pad + i * line_advance),
        # i.e. (stroke_pad + panel_pad + border_w + base_x, ...). The
        # rectangle inner edge sits at stroke_pad + border_w, leaving
        # exactly panel_pad of fill before the text starts.
        rx0 = max(0, rx0)
        ry0 = max(0, ry0)
        bdraw.rectangle(
            (rx0, ry0, rx1, ry1),
            fill=(*bg_fill_rgb, 255) if has_fill else None,
            outline=(*bg_border_rgb, 255) if has_border else None,
            width=border_w if has_border else 0,
        )

    for i, ln in enumerate(lines):
        bb = line_bboxes[i]
        line_w = line_widths[i]
        if align == "left":
            base_x = 0
        elif align == "right":
            base_x = block_w - line_w
        else:  # center (default)
            base_x = (block_w - line_w) // 2
        # ``draw.text`` interprets the position as where the bbox's
        # (left, top) corner of the painted glyphs should land — but the
        # bbox we measured starts at (bb[0], bb[1]) when drawn at (0, 0).
        # Subtract those offsets so the painted ink lands exactly where
        # we expect (left = pad + base_x, top = pad + i * line_advance).
        draw_x = pad + base_x - bb[0]
        draw_y = pad + i * line_advance - bb[1]
        bdraw.text(
            (draw_x, draw_y),
            ln,
            font=font,
            fill=fill,
            stroke_width=stroke_w,
            stroke_fill=stroke,
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

            # Per-item colour overrides; fall back to Step5Params defaults.
            item_fill = getattr(item, "fill_rgb", None)
            item_stroke = getattr(item, "stroke_rgb", None)
            bg_enabled = bool(getattr(item, "bg_fill_enabled", False))
            bg_rgb = (
                tuple(getattr(item, "bg_fill_rgb", (255, 255, 255)))
                if bg_enabled
                else None
            )
            bg_pad = int(getattr(item, "bg_fill_pad", 6))
            border_enabled = bool(getattr(item, "bg_border_enabled", False))
            border_rgb = (
                tuple(getattr(item, "bg_border_rgb", (0, 0, 0)))
                if border_enabled
                else None
            )
            border_px = int(getattr(item, "bg_border_px", 0))

            buf = _render_text_block_to_buffer(
                lines,
                font,
                params,
                align,
                fill_rgb=tuple(item_fill) if item_fill is not None else None,
                stroke_rgb=tuple(item_stroke) if item_stroke is not None else None,
                bg_fill_rgb=bg_rgb,
                bg_fill_pad=bg_pad,
                bg_border_rgb=border_rgb,
                bg_border_px=border_px,
            )
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
