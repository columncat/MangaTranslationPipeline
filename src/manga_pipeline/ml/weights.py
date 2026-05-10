from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import requests

from ..paths import CTD_WEIGHTS, EMBEDDED_LLM_FILENAME, EMBEDDED_LLM_WEIGHTS, ensure_dirs

CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.3/comictextdetector.pt"
)
CTD_SHA256: Optional[str] = None

# unsloth's GGUF mirror — high download count, actively maintained.
# Q4_K_M trades a small quality hit for a ~5 GB file size and good
# tokens/sec on consumer GPUs.
EMBEDDED_LLM_URL = (
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/"
    + EMBEDDED_LLM_FILENAME
)
EMBEDDED_LLM_SHA256: Optional[str] = None

ProgressFn = Callable[[int, int], None]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> Path:
    ensure_dirs()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_sha256 is None or _sha256(dest) == expected_sha256:
            return dest
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

    if expected_sha256 and _sha256(tmp) != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}")

    tmp.replace(dest)
    return dest


def ensure_ctd_weights(progress: Optional[ProgressFn] = None) -> Path:
    return download_file(CTD_URL, CTD_WEIGHTS, expected_sha256=CTD_SHA256, progress=progress)


def ctd_weights_present() -> bool:
    return CTD_WEIGHTS.exists() and CTD_WEIGHTS.stat().st_size > 0


def ensure_embedded_llm_weights(
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Download the embedded translation LLM (Gemma 4 E4B-it Q4_K_M).

    Lazy: skipped if the file already exists. The expected size on
    disk is around 5 GB so callers should usually present a progress
    dialog before invoking this on a foreground thread.
    """
    return download_file(
        EMBEDDED_LLM_URL,
        EMBEDDED_LLM_WEIGHTS,
        expected_sha256=EMBEDDED_LLM_SHA256,
        progress=progress,
    )


def embedded_llm_weights_present() -> bool:
    # A few hundred MB threshold guards against an aborted partial
    # download being mistaken for a complete file.
    return (
        EMBEDDED_LLM_WEIGHTS.exists()
        and EMBEDDED_LLM_WEIGHTS.stat().st_size > 1 << 28  # > 256 MB
    )


def ensure_lama_weights(progress: Optional[ProgressFn] = None) -> None:
    """Trigger LaMa weight download by instantiating the model.

    ``simple_lama_inpainting.SimpleLama`` downloads its weights on first
    construction; there is no public progress hook, so we just call it and
    rely on a coarse "in progress" indicator at the UI level.
    """
    from simple_lama_inpainting import SimpleLama  # lazy import

    if progress:
        progress(0, 0)
    _ = SimpleLama(device="cpu")
    if progress:
        progress(1, 1)


def ensure_manga_ocr_weights(progress: Optional[ProgressFn] = None) -> None:
    """Trigger manga-ocr (HuggingFace) weight download.

    HuggingFace handles caching itself; constructing :class:`MangaOcr` is
    enough to fetch the weights into the local cache. No fine-grained
    progress is available without monkey-patching ``hf_hub_download``.
    """
    from manga_ocr import MangaOcr  # lazy import

    if progress:
        progress(0, 0)
    _ = MangaOcr()
    if progress:
        progress(1, 1)
