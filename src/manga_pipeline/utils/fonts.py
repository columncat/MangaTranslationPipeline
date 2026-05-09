from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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

FONT_GLOBS = ("*.ttf", "*.otf", "*.ttc", "*.TTF", "*.OTF", "*.TTC")


def _has_korean_glyph(font_path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), 16)
        mask = font.getmask("가")
        return mask.getbbox() is not None
    except Exception:
        return False


def _iter_font_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return ()
    seen: set[str] = set()
    out: list[Path] = []
    for pattern in FONT_GLOBS:
        for p in folder.glob(pattern):
            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def list_bundled_fonts() -> list[Path]:
    """Return absolute paths of every font file inside the project's fonts/ folder.

    No Korean-glyph filter — the user is responsible for what they put there.
    Sorted by file name (case-insensitive) for stable UI ordering.
    """
    return sorted(_iter_font_files(FONTS_DIR), key=lambda p: p.name.lower())


def find_default_font() -> Optional[Path]:
    bundled = list(_iter_font_files(FONTS_DIR))
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
