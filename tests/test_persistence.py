from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_pipeline import persistence
from manga_pipeline.models import BBox, OcrResult, PageContext, TranslationResult


def _write_source(tmp_path: Path) -> Path:
    arr = np.full((40, 60, 3), 200, dtype=np.uint8)
    p = tmp_path / "page.png"
    Image.fromarray(arr).save(p)
    return p


def test_save_load_roundtrip_full(tmp_path):
    src = _write_source(tmp_path)
    bboxes = [BBox(5, 5, 20, 10, 200), BBox(30, 12, 25, 15, 375)]
    ctx = PageContext(
        source=np.asarray(Image.open(src).convert("RGB")),
        mask=np.full((40, 60), 255, dtype=np.uint8),
        cleaned=np.full((40, 60, 3), 250, dtype=np.uint8),
        bboxes=bboxes,
        ocr=[OcrResult(bbox=bboxes[0], text_ja="こんにちは")],
        translations=[
            TranslationResult(
                bbox=bboxes[0],
                text_ja="こんにちは",
                text_ko="안녕",
                ignore_boundary=True,
                font_pt=42,
                font_path="C:/Windows/Fonts/malgun.ttf",
            ),
            TranslationResult(
                bbox=bboxes[1],
                text_ja="さようなら",
                text_ko="잘 가",
            ),
        ],
        final=np.full((40, 60, 3), 220, dtype=np.uint8),
    )

    persistence.save_context(ctx, src)
    assert persistence.has_metadata(src)
    assert (tmp_path / "metadata" / "page.json").exists()
    assert (tmp_path / "metadata" / "page_mask.png").exists()
    assert (tmp_path / "metadata" / "page_cleaned.png").exists()
    assert (tmp_path / "metadata" / "page_final.png").exists()

    loaded = persistence.load_context(src)
    assert loaded is not None
    assert len(loaded.bboxes) == 2
    assert loaded.bboxes[0].x == 5 and loaded.bboxes[1].w == 25
    assert len(loaded.ocr) == 1
    assert loaded.ocr[0].text_ja == "こんにちは"
    # OCR/translation must reference the SAME bbox object as ctx.bboxes
    assert loaded.ocr[0].bbox is loaded.bboxes[0]
    assert loaded.translations[0].bbox is loaded.bboxes[0]
    assert loaded.translations[0].text_ko == "안녕"
    assert loaded.translations[0].ignore_boundary is True
    assert loaded.translations[0].font_pt == 42
    assert loaded.translations[1].font_pt is None
    assert loaded.mask is not None and loaded.mask.shape == (40, 60)
    assert loaded.cleaned is not None and loaded.cleaned.shape == (40, 60, 3)
    assert loaded.final is not None


def test_load_returns_none_when_no_metadata(tmp_path):
    src = _write_source(tmp_path)
    assert persistence.load_context(src) is None
    assert not persistence.has_metadata(src)


def test_save_minimal_context(tmp_path):
    """Detect-only state (no OCR/translations/cleaned/final) must round-trip."""
    src = _write_source(tmp_path)
    bboxes = [BBox(0, 0, 10, 10, 100)]
    ctx = PageContext(
        source=np.asarray(Image.open(src).convert("RGB")),
        mask=np.zeros((40, 60), dtype=np.uint8),
        bboxes=bboxes,
    )
    persistence.save_context(ctx, src)
    loaded = persistence.load_context(src)
    assert loaded is not None
    assert len(loaded.bboxes) == 1
    assert loaded.ocr == []
    assert loaded.translations == []
    assert loaded.cleaned is None
    assert loaded.final is None


def test_text_offset_roundtrip(tmp_path):
    src = _write_source(tmp_path)
    bbox = BBox(0, 0, 10, 10, 100)
    ctx = PageContext(
        source=np.asarray(Image.open(src).convert("RGB")),
        bboxes=[bbox],
        translations=[
            TranslationResult(
                bbox=bbox,
                text_ja="ja",
                text_ko="ko",
                text_offset_x=12,
                text_offset_y=-7,
            )
        ],
    )
    persistence.save_context(ctx, src)
    loaded = persistence.load_context(src)
    assert loaded is not None
    assert loaded.translations[0].text_offset_x == 12
    assert loaded.translations[0].text_offset_y == -7


def test_align_and_rotation_roundtrip(tmp_path):
    src = _write_source(tmp_path)
    bbox = BBox(0, 0, 10, 10, 100)
    ctx = PageContext(
        source=np.asarray(Image.open(src).convert("RGB")),
        bboxes=[bbox],
        translations=[
            TranslationResult(
                bbox=bbox,
                text_ja="ja",
                text_ko="ko",
                text_align="right",
                text_rotation=-30,
            )
        ],
    )
    persistence.save_context(ctx, src)
    loaded = persistence.load_context(src)
    assert loaded is not None
    assert loaded.translations[0].text_align == "right"
    assert loaded.translations[0].text_rotation == -30


def test_save_drops_orphan_ocr(tmp_path):
    """OCR/translation entries whose bbox isn't in ctx.bboxes are silently skipped."""
    src = _write_source(tmp_path)
    a = BBox(0, 0, 10, 10, 100)
    b = BBox(20, 20, 5, 5, 25)
    ctx = PageContext(
        source=np.asarray(Image.open(src).convert("RGB")),
        bboxes=[a],  # only one in the canonical list
        ocr=[OcrResult(bbox=a, text_ja="a"), OcrResult(bbox=b, text_ja="orphan")],
    )
    persistence.save_context(ctx, src)
    loaded = persistence.load_context(src)
    assert loaded is not None
    assert len(loaded.ocr) == 1
    assert loaded.ocr[0].text_ja == "a"
