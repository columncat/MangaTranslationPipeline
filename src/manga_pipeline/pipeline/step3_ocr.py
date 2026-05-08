from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..models import OcrResult, PageContext
from ..paths import CACHE_DIR
from .base import PipelineStep, ProgressCallback, StepResult


class MangaOcrEngine:
    """Thin wrapper over ``manga_ocr.MangaOcr``.

    Requires ``transformers < 5`` — version 5.x removed image support from
    ``AutoFeatureExtractor`` which manga-ocr 0.1.10 still uses.
    """

    def __init__(self, force_cpu: bool = False):
        self._mocr = None
        self.force_cpu = force_cpu

    def _load(self) -> None:
        if self._mocr is not None:
            return
        from manga_ocr import MangaOcr

        self._mocr = MangaOcr(force_cpu=self.force_cpu)

    def ocr_image(self, pil_img: Image.Image) -> str:
        self._load()
        assert self._mocr is not None
        return (self._mocr(pil_img) or "").strip()


class Step3Ocr(PipelineStep):
    """Run manga-ocr on each bbox crop of the ORIGINAL (with-text) image."""

    name = "step3_ocr"
    DEBUG_DIR = CACHE_DIR / "ocr_crops"

    def __init__(self, save_debug_crops: bool = True):
        self.engine = MangaOcrEngine()
        self.save_debug_crops = save_debug_crops

    def _prepare_debug_dir(self) -> Optional[Path]:
        if not self.save_debug_crops:
            return None
        self.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        for old in self.DEBUG_DIR.glob("*.png"):
            try:
                old.unlink()
            except OSError:
                pass
        return self.DEBUG_DIR

    def run(
        self,
        ctx: PageContext,
        params: object = None,
        progress: Optional[ProgressCallback] = None,
    ) -> StepResult:
        if not ctx.bboxes:
            return StepResult(ok=False, message="no bounding boxes — run Step 2 first")
        if ctx.source is None:
            return StepResult(ok=False, message="source image missing")

        # Force the OCR source to be a writable, contiguous RGB uint8 array.
        # ctx.source comes from PIL via np.asarray() which is read-only and may
        # also be a view shared with other PIL images.
        ocr_source = np.ascontiguousarray(ctx.source)
        if ocr_source.ndim != 3 or ocr_source.shape[2] != 3:
            return StepResult(
                ok=False,
                message=f"source must be HxWx3 RGB, got shape {ocr_source.shape}",
            )
        if ocr_source.dtype != np.uint8:
            ocr_source = ocr_source.astype(np.uint8)

        h, w = ocr_source.shape[:2]
        debug_dir = self._prepare_debug_dir()

        results: list[OcrResult] = []
        total = len(ctx.bboxes)

        for i, bbox in enumerate(ctx.bboxes):
            if progress:
                progress(i, total, f"OCR {i + 1}/{total}")

            x0 = max(0, int(bbox.x))
            y0 = max(0, int(bbox.y))
            x1 = min(w, int(bbox.x + bbox.w))
            y1 = min(h, int(bbox.y + bbox.h))
            if x1 <= x0 or y1 <= y0:
                if progress:
                    progress(i + 1, total, f"#{i}: invalid bbox, skipped")
                continue

            crop = np.ascontiguousarray(ocr_source[y0:y1, x0:x1])
            pil = Image.fromarray(crop, mode="RGB")

            if debug_dir is not None:
                pil.save(debug_dir / f"bbox_{i:03d}_x{x0}y{y0}_w{x1 - x0}h{y1 - y0}.png")

            try:
                text = self.engine.ocr_image(pil)
            except Exception as e:  # noqa: BLE001
                text = ""
                if progress:
                    progress(i + 1, total, f"#{i}: OCR error: {e}")
                continue

            if text:
                results.append(OcrResult(bbox=bbox, text_ja=text))
                if progress:
                    snippet = text if len(text) <= 30 else text[:27] + "..."
                    progress(i + 1, total, f"#{i}: {snippet}")
            elif progress:
                progress(i + 1, total, f"#{i}: (empty)")

        ctx.ocr = results
        msg = f"{len(results)}/{total} non-empty"
        if debug_dir is not None:
            msg += f" — debug crops in {debug_dir}"
        if progress:
            progress(total, total, msg)
        return StepResult(ok=True, message=msg)
