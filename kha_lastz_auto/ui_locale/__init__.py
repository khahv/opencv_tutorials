"""
UI string maps (English / Vietnamese). Import as ``ui_locale`` to avoid clashing
with the stdlib ``locale`` module.
"""

from __future__ import annotations

from typing import Any, Dict

from . import en, vi

_LOCALES: Dict[str, Dict[str, str]] = {
    "en": en.MESSAGES,
    "vi": vi.MESSAGES,
}


def normalize_language_code(code: Any) -> str:
    """Return 'en' or 'vi' for storage and lookup."""
    if code is None:
        return "en"
    c = str(code).strip().lower()
    if c.startswith("vi"):
        return "vi"
    return "en"


def get_messages(lang: Any) -> Dict[str, str]:
    """Return the message dict for *lang* (falls back to English)."""
    key = normalize_language_code(lang)
    return _LOCALES.get(key, _LOCALES["en"])
