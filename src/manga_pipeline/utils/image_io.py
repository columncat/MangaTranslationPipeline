from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def save_image(arr: np.ndarray, path: str | Path) -> None:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)


def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr, mode="RGB")


def from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def alpha_composite(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    base = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")
    over = Image.fromarray(overlay_rgba, mode="RGBA")
    out = Image.alpha_composite(base, over).convert("RGB")
    return np.asarray(out)
