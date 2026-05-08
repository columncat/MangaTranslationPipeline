from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import ImageFont

from ..paths import FONTS_DIR

DEFAULT_FONT_CANDIDATES = [
    "malgun.ttf",
    "malgunbd.ttf",
    "NanumGothic.ttf",
    "NanumGothicBold.ttf",
    "AppleSDGothicNeo.ttc",
    "NotoSansKR-Regular.otf",
    "NotoSansKR-Bold.otf",
]

WINDOWS_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
]


def _has_korean_glyph(font_path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), 16)
        mask = font.getmask("가")
        return mask.getbbox() is not None
    except Exception:
        return False


def find_default_font() -> Optional[Path]:
    bundled = list(FONTS_DIR.glob("*.[ot]t[fc]"))
    for p in bundled:
        if _has_korean_glyph(p):
            return p

    for d in WINDOWS_FONT_DIRS:
        if not d.exists():
            continue
        for name in DEFAULT_FONT_CANDIDATES:
            p = d / name
            if p.exists() and _has_korean_glyph(p):
                return p
        for p in d.glob("*.[ot]t[fc]"):
            if _has_korean_glyph(p):
                return p

    return None


def load_font(path: Path | str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def supports_korean(path: Path | str) -> bool:
    return _has_korean_glyph(Path(path))
