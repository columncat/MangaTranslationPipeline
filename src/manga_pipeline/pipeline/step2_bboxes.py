from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import Step2Params
from ..models import BBox, PageContext
from .base import PipelineStep, ProgressCallback, StepResult


def extract_bboxes(
    mask: np.ndarray,
    *,
    kernel_w: int = 15,
    kernel_h: int = 15,
    iterations: int = 2,
    min_area: int = 200,
    max_area_ratio: float = 0.5,
) -> list[BBox]:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D binary image")

    binary = (mask > 0).astype(np.uint8) * 255

    kw = max(1, int(kernel_w))
    kh = max(1, int(kernel_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iterations)))

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    image_area = mask.size
    max_area = int(max_area_ratio * image_area)

    out: list[BBox] = []
    for i in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[i, :5])
        if area < min_area or area > max_area:
            continue
        out.append(BBox(x=x, y=y, w=w, h=h, area=area))

    out.sort(key=lambda b: (b.y, b.x))
    return out


class Step2BBoxes(PipelineStep):
    name = "step2_bboxes"

    def run(
        self,
        ctx: PageContext,
        params: Step2Params,
        progress: Optional[ProgressCallback] = None,
    ) -> StepResult:
        if ctx.mask is None:
            return StepResult(ok=False, message="Step 1 mask is missing")

        if progress:
            progress(0, 1, "extracting bounding boxes")

        ctx.bboxes = extract_bboxes(
            ctx.mask,
            kernel_w=params.kernel_w,
            kernel_h=params.kernel_h,
            iterations=params.iterations,
            min_area=params.min_area,
            max_area_ratio=params.max_area_ratio,
        )

        if progress:
            progress(1, 1, f"{len(ctx.bboxes)} bboxes")

        return StepResult(ok=True, message=f"{len(ctx.bboxes)} bounding boxes")
