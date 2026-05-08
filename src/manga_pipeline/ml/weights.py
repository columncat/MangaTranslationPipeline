from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import requests

from ..paths import CTD_WEIGHTS, ensure_dirs

CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.3/comictextdetector.pt"
)
CTD_SHA256: Optional[str] = None

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
