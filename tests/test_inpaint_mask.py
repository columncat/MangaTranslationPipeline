from __future__ import annotations

import numpy as np

from manga_pipeline.models import BBox, TranslationResult
from manga_pipeline.pipeline.step5_render import build_inpaint_mask


def _tr(x: int, y: int, w: int, h: int) -> TranslationResult:
    return TranslationResult(bbox=BBox(x, y, w, h, w * h), text_ja="", text_ko="")


def test_inpaint_mask_only_inside_translated_bboxes():
    base = np.full((100, 100), 255, dtype=np.uint8)
    items = [_tr(10, 10, 20, 20)]
    out = build_inpaint_mask((100, 100), items, base_mask=base, dilate_px=0)

    assert out[15, 15] == 255
    assert out[80, 80] == 0


def test_inpaint_mask_excludes_deleted_bbox_region():
    base = np.full((100, 100), 255, dtype=np.uint8)
    kept = [_tr(10, 10, 20, 20)]
    out = build_inpaint_mask((100, 100), kept, base_mask=base, dilate_px=0)

    deleted_region = out[60:80, 60:80]
    assert int(deleted_region.sum()) == 0


def test_inpaint_mask_empty_translations_yields_empty_mask():
    base = np.full((50, 50), 255, dtype=np.uint8)
    out = build_inpaint_mask((50, 50), [], base_mask=base, dilate_px=0)
    assert int(out.sum()) == 0


def test_inpaint_mask_falls_back_to_rectangles_without_base():
    items = [_tr(5, 5, 10, 10)]
    out = build_inpaint_mask((30, 30), items, base_mask=None, dilate_px=0)
    assert out[7, 7] == 255
    assert out[20, 20] == 0


def test_inpaint_mask_dilation_grows_region():
    items = [_tr(20, 20, 10, 10)]
    out0 = build_inpaint_mask((60, 60), items, base_mask=None, dilate_px=0)
    out5 = build_inpaint_mask((60, 60), items, base_mask=None, dilate_px=5)
    assert int(out5.sum()) > int(out0.sum())
