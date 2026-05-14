"""Down/up-scale a PageContext at the queue boundary.

The pipeline (Steps 1, 3, 5-inpaint) runs on whatever is in
``ctx.source``. To make morphology kernels behave consistently across
manga page sizes, the work-queue lets the user choose a scale factor
that is applied to every queued image *before* the pipeline runs and
reverted *after* every pipeline step finishes saving — so on disk the
metadata and final PNG always use the original coordinate system.

The pure functions here are used by ``QueueScaler`` in main_window;
tested in isolation so the scaling math is independently auditable.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..models import BBox, PageContext


def _round_int(v: float) -> int:
    return int(round(v))


def scaled_dims(orig_w: int, orig_h: int, scale: float) -> tuple[int, int]:
    """Compute (w, h) at ``scale``, clamping to at least 1×1."""
    return max(1, _round_int(orig_w * scale)), max(1, _round_int(orig_h * scale))


def _resize_image(arr: np.ndarray, new_w: int, new_h: int, interp: int) -> np.ndarray:
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


def _scale_bbox_inplace(b: BBox, scale: float) -> None:
    """Multiply the bbox's coords by ``scale`` and update its area."""
    b.x = _round_int(b.x * scale)
    b.y = _round_int(b.y * scale)
    b.w = max(1, _round_int(b.w * scale))
    b.h = max(1, _round_int(b.h * scale))
    b.area = b.w * b.h


def downscale_context(ctx: PageContext, scale: float) -> None:
    """Mutate ``ctx`` in-place to live in the scaled coordinate system.

    Source uses INTER_AREA (best for downscale), masks INTER_NEAREST
    (preserve binary edges), cleaned image INTER_AREA. Bboxes and any
    per-bbox mask are scaled positionally.
    """
    if scale == 1.0:
        return
    h, w = ctx.source.shape[:2]
    new_w, new_h = scaled_dims(w, h, scale)
    ctx.source = _resize_image(ctx.source, new_w, new_h, cv2.INTER_AREA)
    if ctx.mask is not None:
        ctx.mask = _resize_image(ctx.mask, new_w, new_h, cv2.INTER_NEAREST)
    if ctx.cleaned is not None:
        ctx.cleaned = _resize_image(ctx.cleaned, new_w, new_h, cv2.INTER_AREA)
    for b in ctx.bboxes:
        _scale_bbox_inplace(b, scale)
    # Per-bbox masks are bbox-local; recompute their size from the now-scaled
    # bbox dimensions.
    for tr in ctx.translations:
        bm = getattr(tr, "bbox_mask", None)
        if bm is None:
            continue
        target_h = max(1, tr.bbox.h)
        target_w = max(1, tr.bbox.w)
        if bm.shape[:2] != (target_h, target_w):
            tr.bbox_mask = _resize_image(bm, target_w, target_h, cv2.INTER_NEAREST)


def upscale_context_to_original(
    ctx: PageContext,
    original_source: np.ndarray,
    inv_scale: float,
) -> None:
    """Mutate ``ctx`` in-place back to the ``original_source`` size.

    ``inv_scale`` should be ``1 / downscale_factor`` (e.g. 2.0 if the
    queue used 0.5). Cleaned / final get LANCZOS4 (best upscale
    quality) and the mask gets nearest-neighbour. Bboxes scale up.
    The caller is expected to swap ``ctx.source = original_source``
    *before* calling this; the function trusts ``original_source`` for
    the target dimensions.
    """
    if inv_scale == 1.0:
        return
    oh, ow = original_source.shape[:2]
    if ctx.mask is not None:
        ctx.mask = _resize_image(ctx.mask, ow, oh, cv2.INTER_NEAREST)
    if ctx.cleaned is not None:
        ctx.cleaned = _resize_image(ctx.cleaned, ow, oh, cv2.INTER_LANCZOS4)
    if ctx.final is not None:
        ctx.final = _resize_image(ctx.final, ow, oh, cv2.INTER_LANCZOS4)
    for b in ctx.bboxes:
        _scale_bbox_inplace(b, inv_scale)
        # Clamp to image bounds so rounding doesn't push a bbox off the
        # canvas.
        b.x = max(0, min(b.x, ow - 1))
        b.y = max(0, min(b.y, oh - 1))
        b.w = max(1, min(b.w, ow - b.x))
        b.h = max(1, min(b.h, oh - b.y))
        b.area = b.w * b.h
    for tr in ctx.translations:
        bm = getattr(tr, "bbox_mask", None)
        if bm is None:
            continue
        target_h = max(1, tr.bbox.h)
        target_w = max(1, tr.bbox.w)
        if bm.shape[:2] != (target_h, target_w):
            tr.bbox_mask = _resize_image(bm, target_w, target_h, cv2.INTER_NEAREST)
