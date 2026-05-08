from __future__ import annotations

import os
from typing import Optional

SERVICE = "manga_pipeline"
KEY_ANTHROPIC = "ANTHROPIC_API_KEY"


def _keyring():
    try:
        import keyring  # type: ignore

        return keyring
    except ImportError:
        return None


def get_anthropic_key() -> Optional[str]:
    env = os.environ.get(KEY_ANTHROPIC)
    if env:
        return env
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, KEY_ANTHROPIC)
    except Exception:
        return None


def set_anthropic_key(value: str) -> bool:
    kr = _keyring()
    if kr is None:
        os.environ[KEY_ANTHROPIC] = value
        return False
    try:
        kr.set_password(SERVICE, KEY_ANTHROPIC, value)
        return True
    except Exception:
        os.environ[KEY_ANTHROPIC] = value
        return False


def delete_anthropic_key() -> None:
    kr = _keyring()
    if kr is None:
        os.environ.pop(KEY_ANTHROPIC, None)
        return
    try:
        kr.delete_password(SERVICE, KEY_ANTHROPIC)
    except Exception:
        pass
    os.environ.pop(KEY_ANTHROPIC, None)
