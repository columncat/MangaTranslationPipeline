"""OS keyring access for cloud-API credentials.

Each provider stores its key under a separate keyring entry so the user
can rotate them independently. The legacy ``ANTHROPIC_API_KEY`` entry
is preserved as a fallback so v1.0 keyrings keep working.

Falls back to a process-local environment variable if ``keyring`` isn't
installed or the platform backend can't write.
"""
from __future__ import annotations

import os
from typing import Optional

SERVICE = "manga_pipeline"

# Legacy v1.0 keyring entry; still consulted on read for back-compat.
KEY_ANTHROPIC = "ANTHROPIC_API_KEY"

# Map provider id → (keyring entry name, env-var name).
PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    "anthropic": (KEY_ANTHROPIC, "ANTHROPIC_API_KEY"),
    "openai_compat": ("OPENAI_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GOOGLE_API_KEY", "GOOGLE_API_KEY"),
}


def _keyring():
    try:
        import keyring  # type: ignore

        return keyring
    except ImportError:
        return None


def _entries_for(provider: str) -> tuple[str, str]:
    return PROVIDER_KEYS.get(provider, (KEY_ANTHROPIC, "ANTHROPIC_API_KEY"))


def get_api_key(provider: str = "anthropic") -> Optional[str]:
    entry, env_name = _entries_for(provider)
    env = os.environ.get(env_name)
    if env:
        return env
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, entry)
    except Exception:  # noqa: BLE001
        return None


def set_api_key(value: str, provider: str = "anthropic") -> bool:
    entry, env_name = _entries_for(provider)
    kr = _keyring()
    if kr is None:
        os.environ[env_name] = value
        return False
    try:
        kr.set_password(SERVICE, entry, value)
        return True
    except Exception:  # noqa: BLE001
        os.environ[env_name] = value
        return False


def delete_api_key(provider: str = "anthropic") -> None:
    entry, env_name = _entries_for(provider)
    kr = _keyring()
    if kr is None:
        os.environ.pop(env_name, None)
        return
    try:
        kr.delete_password(SERVICE, entry)
    except Exception:  # noqa: BLE001
        pass
    os.environ.pop(env_name, None)


# ---- v1.0 compatibility shims ----

def get_anthropic_key() -> Optional[str]:
    return get_api_key("anthropic")


def set_anthropic_key(value: str) -> bool:
    return set_api_key(value, "anthropic")


def delete_anthropic_key() -> None:
    delete_api_key("anthropic")
