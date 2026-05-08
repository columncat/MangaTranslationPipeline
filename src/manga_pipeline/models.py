from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y : self.y2, self.x : self.x2]


@dataclass
class OcrResult:
    bbox: BBox
    text_ja: str


@dataclass
class TranslationResult:
    bbox: BBox
    text_ja: str
    text_ko: str
    # Default True: text is centered on the bbox center at a fixed font size
    # and only breaks on user-inserted (or Claude-inserted) "\n" characters.
    # Toggle off in the Edit Translation dialog to fall back to bbox-fit.
    ignore_boundary: bool = True
    # Per-dialogue overrides (None → use Step5Params defaults).
    font_path: Optional[str] = None  # absolute path to a TTF/OTF/TTC file
    font_pt: Optional[int] = None    # fixed font size in points
    # Pixel offset from the bbox center where the text block is rendered.
    # Used for ``Move text`` edits in the Translate tab; does not affect the
    # inpainting mask.
    text_offset_x: int = 0
    text_offset_y: int = 0
    # Per-line horizontal alignment of the rendered text:
    # ``"left" | "center" | "right"``.
    text_align: str = "center"
    # Counter-clockwise rotation in degrees applied to the rendered text
    # block around the bbox center (after ``text_offset``).
    text_rotation: int = 0


@dataclass
class PageContext:
    source: np.ndarray
    mask: Optional[np.ndarray] = None
    cleaned: Optional[np.ndarray] = None
    bboxes: list[BBox] = field(default_factory=list)
    ocr: list[OcrResult] = field(default_factory=list)
    translations: list[TranslationResult] = field(default_factory=list)
    final: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        return int(self.source.shape[0])

    @property
    def width(self) -> int:
        return int(self.source.shape[1])
