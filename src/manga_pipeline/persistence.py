"""Save and reload :class:`PageContext` snapshots next to source images.

Layout: for ``<dir>/<name>.<ext>`` we store::

    <dir>/metadata/<name>.json       # bboxes, OCR, translations, params
    <dir>/metadata/<name>_mask.png   # uint8 binary text mask (HxW)
    <dir>/metadata/<name>_cleaned.png  # inpainted RGB image
    <dir>/metadata/<name>_final.png    # final composited RGB image

JSON references bboxes by *index* so OCR/translation entries can be matched
back to bbox objects after a reload (preserving the ``is`` identity used
elsewhere for cascade-delete logic).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .models import BBox, OcrResult, PageContext, TranslationResult
from .utils.image_io import load_rgb

METADATA_VERSION = 1
METADATA_SUBDIR = "metadata"


def metadata_dir(source_path: Path) -> Path:
    return Path(source_path).parent / METADATA_SUBDIR


def _paths_for(source_path: Path) -> dict[str, Path]:
    d = metadata_dir(source_path)
    stem = Path(source_path).stem
    return {
        "json": d / f"{stem}.json",
        "mask": d / f"{stem}_mask.png",
        "cleaned": d / f"{stem}_cleaned.png",
        "final": d / f"{stem}_final.png",
    }


def has_metadata(source_path: Path) -> bool:
    return _paths_for(source_path)["json"].exists()


def save_context(ctx: PageContext, source_path: Path) -> Path:
    """Persist whatever is present on ``ctx``. Returns the JSON path."""
    paths = _paths_for(source_path)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)

    # bbox identity → index, so OCR/translation rows are decoupled from the
    # in-memory ordering (and from any stale references).
    bbox_idx: dict[int, int] = {id(b): i for i, b in enumerate(ctx.bboxes)}

    payload = {
        "version": METADATA_VERSION,
        "source": Path(source_path).name,
        "bboxes": [
            {
                "x": int(b.x),
                "y": int(b.y),
                "w": int(b.w),
                "h": int(b.h),
                "area": int(b.area),
            }
            for b in ctx.bboxes
        ],
        "ocr": [
            {
                "bbox_idx": bbox_idx.get(id(r.bbox), -1),
                "text_ja": r.text_ja,
            }
            for r in ctx.ocr
            if id(r.bbox) in bbox_idx
        ],
        "translations": [
            {
                "bbox_idx": bbox_idx.get(id(t.bbox), -1),
                "text_ja": t.text_ja,
                "text_ko": t.text_ko,
                "ignore_boundary": bool(t.ignore_boundary),
                "font_path": t.font_path,
                "font_pt": t.font_pt,
                "text_offset_x": int(t.text_offset_x or 0),
                "text_offset_y": int(t.text_offset_y or 0),
                "text_align": getattr(t, "text_align", "center") or "center",
                "text_rotation": int(getattr(t, "text_rotation", 0) or 0),
            }
            for t in ctx.translations
            if id(t.bbox) in bbox_idx
        ],
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if ctx.mask is not None:
        Image.fromarray(_as_uint8(ctx.mask)).save(paths["mask"])
    if ctx.cleaned is not None:
        Image.fromarray(_as_uint8(ctx.cleaned)).save(paths["cleaned"])
    if ctx.final is not None:
        Image.fromarray(_as_uint8(ctx.final)).save(paths["final"])

    return paths["json"]


def load_context(source_path: Path) -> Optional[PageContext]:
    """Reconstruct a :class:`PageContext` from saved metadata.

    Returns ``None`` when no JSON is present. The source image is always
    reloaded from disk; ``mask``/``cleaned``/``final`` are loaded if their
    sidecar PNGs exist.
    """
    paths = _paths_for(source_path)
    if not paths["json"].exists():
        return None

    source = load_rgb(source_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))

    bboxes = [
        BBox(
            x=int(b["x"]),
            y=int(b["y"]),
            w=int(b["w"]),
            h=int(b["h"]),
            area=int(b.get("area", b["w"] * b["h"])),
        )
        for b in payload.get("bboxes", [])
    ]

    ocr: list[OcrResult] = []
    for o in payload.get("ocr", []):
        idx = int(o.get("bbox_idx", -1))
        if 0 <= idx < len(bboxes):
            ocr.append(OcrResult(bbox=bboxes[idx], text_ja=str(o.get("text_ja", ""))))

    translations: list[TranslationResult] = []
    for t in payload.get("translations", []):
        idx = int(t.get("bbox_idx", -1))
        if 0 <= idx < len(bboxes):
            translations.append(
                TranslationResult(
                    bbox=bboxes[idx],
                    text_ja=str(t.get("text_ja", "")),
                    text_ko=str(t.get("text_ko", "")),
                    ignore_boundary=bool(t.get("ignore_boundary", True)),
                    font_path=t.get("font_path"),
                    font_pt=t.get("font_pt"),
                    text_offset_x=int(t.get("text_offset_x", 0) or 0),
                    text_offset_y=int(t.get("text_offset_y", 0) or 0),
                    text_align=str(t.get("text_align", "center") or "center"),
                    text_rotation=int(t.get("text_rotation", 0) or 0),
                )
            )

    return PageContext(
        source=source,
        mask=_load_image(paths["mask"], "L"),
        cleaned=_load_image(paths["cleaned"], "RGB"),
        bboxes=bboxes,
        ocr=ocr,
        translations=translations,
        final=_load_image(paths["final"], "RGB"),
    )


def _load_image(path: Path, mode: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.asarray(Image.open(path).convert(mode))
    except (OSError, ValueError):
        return None


def _as_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)
