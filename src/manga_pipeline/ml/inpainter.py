from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class Inpainter:
    """Wraps simple-lama-inpainting for text removal."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._lama = None

    def _load(self) -> None:
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama

        self._lama = SimpleLama(device=self.device)

    def inpaint(self, img_rgb: np.ndarray, mask: np.ndarray, dilate_px: int = 7) -> np.ndarray:
        """Remove text from ``img_rgb`` using LaMa, returning a clean RGB image."""
        self._load()
        assert self._lama is not None

        if mask.ndim == 3:
            mask = mask[..., 0]
        binary = (mask > 0).astype(np.uint8) * 255

        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
            binary = cv2.dilate(binary, k, iterations=1)

        pil_img = Image.fromarray(img_rgb, mode="RGB")
        pil_mask = Image.fromarray(binary, mode="L")
        result = self._lama(pil_img, pil_mask)
        return np.asarray(result.convert("RGB"))
