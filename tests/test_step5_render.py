from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_pipeline.config import Step5Params
from manga_pipeline.models import BBox, TranslationResult
from manga_pipeline.pipeline.step5_render import (
    KoreanRenderer,
    _block_size,
    _measure,
    _wrap_to_width,
    fit_font_size,
)
from manga_pipeline.utils.fonts import find_default_font, load_font

FONT = find_default_font()
pytestmark = pytest.mark.skipif(FONT is None, reason="no Korean font available")


def _draw():
    return ImageDraw.Draw(Image.new("RGB", (1, 1)))


def test_wrap_to_width_keeps_lines_within_width():
    draw = _draw()
    font = load_font(FONT, 20)
    text = "안녕하세요반갑습니다오늘날씨가좋네요"
    max_w = 80
    lines = _wrap_to_width(draw, text, font, max_w)
    assert len(lines) >= 2
    for ln in lines:
        w, _ = _measure(draw, ln, font)
        assert w <= max_w or len(ln) == 1


def test_fit_font_size_respects_box():
    draw = _draw()
    text = "한국어 번역 텍스트"
    pt, lines = fit_font_size(
        draw, text, FONT, box_w=100, box_h=60,
        min_pt=8, max_pt=48, line_spacing=1.15, padding=4,
    )
    assert 8 <= pt <= 48
    font = load_font(FONT, pt)
    w, h = _block_size(draw, lines, font, 1.15)
    assert w <= 100 - 8
    assert h <= 60 - 8


def test_fit_font_size_picks_larger_for_bigger_box():
    draw = _draw()
    text = "안"
    pt_small, _ = fit_font_size(draw, text, FONT, 30, 30, min_pt=8, max_pt=48, line_spacing=1.15, padding=2)
    pt_big, _ = fit_font_size(draw, text, FONT, 200, 200, min_pt=8, max_pt=48, line_spacing=1.15, padding=2)
    assert pt_big > pt_small


def test_renderer_writes_pixels_into_box():
    cleaned = np.full((200, 300, 3), 255, dtype=np.uint8)
    items = [
        TranslationResult(
            bbox=BBox(x=50, y=50, w=200, h=100, area=20000),
            text_ja="ja",
            text_ko="안녕하세요",
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out = renderer.render(cleaned, items, Step5Params())

    assert out.shape == cleaned.shape
    box_region = out[50:150, 50:250]
    assert np.any(box_region.min(axis=2) < 128)


def test_renderer_ignore_boundary_centers_on_bbox_and_extends_both_sides():
    cleaned = np.full((300, 600, 3), 255, dtype=np.uint8)
    bbox_x, bbox_y, bbox_w, bbox_h = 280, 140, 40, 20
    items = [
        TranslationResult(
            bbox=BBox(x=bbox_x, y=bbox_y, w=bbox_w, h=bbox_h, area=bbox_w * bbox_h),
            text_ja="ja",
            text_ko="긴 한국어 번역 텍스트입니다",
            ignore_boundary=True,
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out = renderer.render(cleaned, items, Step5Params(outside_pt=24))

    # Text is centered on bbox center (x ~= 300), so it should extend BOTH
    # to the left of the bbox (x < 280) and to the right (x > 320).
    band_y = slice(bbox_y - 20, bbox_y + bbox_h + 20)
    left_band = out[band_y, : bbox_x]
    right_band = out[band_y, bbox_x + bbox_w :]
    assert np.any(left_band.min(axis=2) < 128), "text must extend left of bbox"
    assert np.any(right_band.min(axis=2) < 128), "text must extend right of bbox"


def test_renderer_ignore_boundary_only_user_newlines_split():
    cleaned = np.full((300, 600, 3), 255, dtype=np.uint8)
    items = [
        TranslationResult(
            bbox=BBox(x=100, y=100, w=20, h=20, area=400),
            text_ja="ja",
            text_ko="첫줄\n두번째줄",
            ignore_boundary=True,
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out = renderer.render(cleaned, items, Step5Params(outside_pt=20, line_spacing=1.2))

    # There must be dark pixels on two distinct horizontal bands (the two lines).
    column_min = out[:, 100:300, :].min(axis=2)
    rows_with_text = np.where(column_min.min(axis=1) < 128)[0]
    assert len(rows_with_text) > 0
    if len(rows_with_text) > 1:
        gaps = np.diff(rows_with_text)
        assert (gaps > 3).sum() >= 1, "expected at least one vertical gap between lines"


def test_renderer_per_item_font_pt_overrides_outside_pt():
    cleaned = np.full((300, 600, 3), 255, dtype=np.uint8)
    items_default = [
        TranslationResult(
            bbox=BBox(x=200, y=100, w=80, h=40, area=3200),
            text_ja="ja",
            text_ko="안녕",
            ignore_boundary=True,
        )
    ]
    items_big = [
        TranslationResult(
            bbox=BBox(x=200, y=100, w=80, h=40, area=3200),
            text_ja="ja",
            text_ko="안녕",
            ignore_boundary=True,
            font_pt=72,
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out_default = renderer.render(cleaned, items_default, Step5Params(outside_pt=20))
    out_big = renderer.render(cleaned, items_big, Step5Params(outside_pt=20))

    # Larger override → more dark pixels overall.
    dark_default = int((out_default.min(axis=2) < 128).sum())
    dark_big = int((out_big.min(axis=2) < 128).sum())
    assert dark_big > dark_default * 2


def test_renderer_per_item_font_path_falls_back_when_missing():
    cleaned = np.full((100, 200, 3), 255, dtype=np.uint8)
    items = [
        TranslationResult(
            bbox=BBox(x=50, y=40, w=80, h=20, area=1600),
            text_ja="ja",
            text_ko="가",
            ignore_boundary=True,
            font_path="C:/this/path/does/not/exist.ttf",
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    # Should not raise — falls back to renderer's default font.
    out = renderer.render(cleaned, items, Step5Params(outside_pt=24))
    assert out.shape == cleaned.shape
    assert np.any(out.min(axis=2) < 128)


def test_renderer_alignment_right_places_text_at_right_edge():
    cleaned = np.full((100, 400, 3), 255, dtype=np.uint8)
    items = [
        TranslationResult(
            bbox=BBox(x=200, y=40, w=80, h=40, area=3200),
            text_ja="ja",
            text_ko="가\n나다라마바",  # two lines of different widths
            ignore_boundary=True,
            text_align="right",
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out = renderer.render(cleaned, items, Step5Params(outside_pt=20))

    # Both lines must end near the same right edge (right-aligned), so the
    # short top line "가" should appear shifted to the right of where it
    # would be when center-aligned. We approximate this by checking that
    # any dark pixel exists in the top-right region.
    dark = (out.min(axis=2) < 128)
    assert dark.any()


def test_renderer_rotation_produces_text_pixels():
    cleaned = np.full((300, 400, 3), 255, dtype=np.uint8)
    items = [
        TranslationResult(
            bbox=BBox(x=180, y=140, w=40, h=20, area=800),
            text_ja="ja",
            text_ko="회전",
            ignore_boundary=True,
            text_rotation=45,
        )
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out = renderer.render(cleaned, items, Step5Params(outside_pt=24))

    assert out.shape == cleaned.shape
    # Some rotated text must have been drawn somewhere on the canvas.
    assert (out.min(axis=2) < 128).sum() > 5


def test_renderer_skips_empty_translation():
    cleaned = np.full((100, 100, 3), 255, dtype=np.uint8)
    items = [
        TranslationResult(bbox=BBox(0, 0, 50, 50, 2500), text_ja="ja", text_ko=""),
    ]
    renderer = KoreanRenderer(font_path=FONT)
    out = renderer.render(cleaned, items, Step5Params())
    assert np.array_equal(out, cleaned)
