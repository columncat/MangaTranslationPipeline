from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..paths import CTD_WEIGHTS, VENDOR_DIR
from .weights import ensure_ctd_weights


class TextDetector:
    """Wraps the vendored comic-text-detector model.

    The vendor repo (https://github.com/dmMaze/comic-text-detector) must be
    cloned to ``vendor/comic_text_detector/`` so its modules are importable.
    Weights are auto-downloaded via :func:`ensure_ctd_weights`.
    """

    def __init__(self, device: str = "cuda", input_size: int = 1024):
        self.device = device
        self.input_size = input_size
        self._model = None
        self._mask_thresh: Optional[float] = None
        self._weights_path: Optional[Path] = None

    def _load(self, mask_thresh: float) -> None:
        if self._model is not None and self._mask_thresh == mask_thresh:
            return

        ctd_root = VENDOR_DIR / "comic_text_detector"
        if not ctd_root.exists():
            raise RuntimeError(
                f"comic-text-detector vendor missing. Clone it to {ctd_root}:\n"
                "  git clone https://github.com/dmMaze/comic-text-detector.git "
                f"{ctd_root}"
            )

        if str(ctd_root) not in sys.path:
            sys.path.insert(0, str(ctd_root))

        from inference import TextDetector as _CTD

        self._weights_path = ensure_ctd_weights()
        self._model = _CTD(
            model_path=str(self._weights_path),
            input_size=self.input_size,
            device=self.device,
            act="leaky",
            mask_thresh=mask_thresh,
        )
        self._mask_thresh = mask_thresh

    def detect(self, img_rgb: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Return a uint8 binary mask (0/255) of text regions, same HxW as input."""
        self._load(threshold)
        assert self._model is not None

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _mask, mask_refined, _blk_list = self._model(
            bgr, refine_mode=0, keep_undetected_mask=True
        )

        mask = mask_refined if mask_refined is not None else _mask
        if mask.ndim == 3:
            mask = mask[..., 0]
        binary = (mask > 0).astype(np.uint8) * 255

        if binary.shape[:2] != img_rgb.shape[:2]:
            binary = cv2.resize(
                binary, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return binary
