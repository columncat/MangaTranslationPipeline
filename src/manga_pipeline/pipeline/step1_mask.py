from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import Step1Params
from ..device import auto_device, empty_cuda_cache
from ..ml.text_detector import TextDetector
from ..models import PageContext
from .base import PipelineStep, ProgressCallback, StepResult


class Step1Mask(PipelineStep):
    """Step 1 — detect the text mask only. Inpainting moved to Step 5."""

    name = "step1_mask"

    def __init__(self, device: Optional[str] = None):
        self.device = device or auto_device()
        self._detector: Optional[TextDetector] = None

    def _detector_obj(self) -> TextDetector:
        if self._detector is None:
            self._detector = TextDetector(device=self.device)
        return self._detector

    def run(
        self,
        ctx: PageContext,
        params: Step1Params,
        progress: Optional[ProgressCallback] = None,
    ) -> StepResult:
        if ctx.source is None:
            return StepResult(ok=False, message="source image missing")

        try:
            if progress:
                progress(0, 1, "detecting text mask")
            mask = self._detector_obj().detect(ctx.source, threshold=params.mask_threshold)

            if params.mask_dilate_px > 0:
                import cv2

                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (params.mask_dilate_px * 2 + 1,) * 2
                )
                mask = cv2.dilate(mask, k, iterations=1)
            ctx.mask = mask
            ctx.cleaned = None
            ctx.final = None

            if progress:
                progress(1, 1, "mask done")
            return StepResult(ok=True, message="text mask generated")

        except Exception as e:  # noqa: BLE001
            empty_cuda_cache()
            return StepResult(ok=False, message=str(e), error=e)

    def warmup_async(self) -> None:
        self._detector_obj()


def make_synthetic_mask(img_rgb: np.ndarray, threshold: int = 80) -> np.ndarray:
    """Fallback test-only mask: dark pixels become text."""
    import cv2

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask
