from __future__ import annotations

import numpy as np

from manga_pipeline.pipeline.step2_bboxes import extract_bboxes


def _blank(h: int = 400, w: int = 400) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _draw_rect(mask: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    mask[y : y + h, x : x + w] = 255


def test_returns_one_box_for_one_rect():
    mask = _blank()
    _draw_rect(mask, 50, 50, 80, 30)
    bboxes = extract_bboxes(mask, kernel_w=3, kernel_h=3, min_area=100)
    assert len(bboxes) == 1
    b = bboxes[0]
    assert b.x == 50 and b.y == 50
    assert b.w == 80 and b.h == 30


def test_separate_rects_become_separate_boxes():
    mask = _blank()
    _draw_rect(mask, 30, 30, 40, 40)
    _draw_rect(mask, 200, 200, 40, 40)
    _draw_rect(mask, 30, 300, 40, 40)
    bboxes = extract_bboxes(mask, kernel_w=3, kernel_h=3, min_area=100)
    assert len(bboxes) == 3


def test_morphology_merges_close_glyphs():
    mask = _blank()
    for x in range(50, 251, 30):
        _draw_rect(mask, x, 100, 20, 20)
    bboxes = extract_bboxes(mask, kernel_w=25, kernel_h=5, min_area=100)
    assert len(bboxes) == 1
    assert bboxes[0].w >= 200


def test_min_area_filters_noise():
    mask = _blank()
    _draw_rect(mask, 10, 10, 3, 3)
    _draw_rect(mask, 100, 100, 60, 60)
    bboxes = extract_bboxes(mask, kernel_w=1, kernel_h=1, min_area=200)
    assert len(bboxes) == 1
    assert bboxes[0].x == 100


def test_max_area_filters_huge_blob():
    mask = np.full((100, 100), 255, dtype=np.uint8)
    bboxes = extract_bboxes(mask, kernel_w=1, kernel_h=1, min_area=10, max_area_ratio=0.5)
    assert len(bboxes) == 0


def test_empty_mask_returns_empty():
    assert extract_bboxes(_blank(), kernel_w=3, kernel_h=3) == []


def test_invalid_input_raises():
    import pytest

    with pytest.raises(ValueError):
        extract_bboxes(np.zeros((10, 10, 3), dtype=np.uint8))


def test_sort_order_top_to_bottom():
    mask = _blank()
    _draw_rect(mask, 100, 200, 30, 30)
    _draw_rect(mask, 100, 50, 30, 30)
    _draw_rect(mask, 200, 50, 30, 30)
    bboxes = extract_bboxes(mask, kernel_w=3, kernel_h=3, min_area=100)
    assert [b.y for b in bboxes] == [50, 50, 200]
    assert bboxes[0].x < bboxes[1].x
