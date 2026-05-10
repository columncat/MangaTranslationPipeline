from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FONTS_DIR = PROJECT_ROOT / "fonts"
SAMPLES_DIR = PROJECT_ROOT / "samples"
VENDOR_DIR = PROJECT_ROOT / "vendor"
CACHE_DIR = MODELS_DIR / "cache"

CTD_WEIGHTS = MODELS_DIR / "comictextdetector.pt"

# Embedded translation LLM (Gemma 4 E4B-it Q4_K_M, ~5 GB).
EMBEDDED_LLM_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"
EMBEDDED_LLM_WEIGHTS = MODELS_DIR / EMBEDDED_LLM_FILENAME


def ensure_dirs() -> None:
    for p in (MODELS_DIR, FONTS_DIR, SAMPLES_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)
