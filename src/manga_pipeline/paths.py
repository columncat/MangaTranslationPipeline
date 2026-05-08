from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FONTS_DIR = PROJECT_ROOT / "fonts"
SAMPLES_DIR = PROJECT_ROOT / "samples"
VENDOR_DIR = PROJECT_ROOT / "vendor"
CACHE_DIR = MODELS_DIR / "cache"

CTD_WEIGHTS = MODELS_DIR / "comictextdetector.pt"


def ensure_dirs() -> None:
    for p in (MODELS_DIR, FONTS_DIR, SAMPLES_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)
