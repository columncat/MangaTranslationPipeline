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
    # Optional per-bbox inpainting mask in bbox-local coordinates
    # (shape == (bbox.h, bbox.w), uint8, 0 = keep original, 255 = inpaint).
    # When None, Step 5 falls back to using the full bbox rectangle.
    bbox_mask: Optional["np.ndarray"] = field(default=None, repr=False)
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
    # Per-dialogue colour overrides. ``None`` falls back to Step5Params
    # defaults (black fill, white stroke). Stored as RGB triples.
    fill_rgb: Optional[tuple[int, int, int]] = None
    stroke_rgb: Optional[tuple[int, int, int]] = None
    # Optional ellipse drawn behind the rendered text block as a backdrop
    # (for white text on busy art). Disabled by default.
    bg_fill_enabled: bool = False
    bg_fill_rgb: tuple[int, int, int] = (255, 255, 255)
    # Padding (px) added around the text bounding box before fitting the
    # ellipse to it. Larger values produce a roomier oval.
    bg_fill_pad: int = 6


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
