"""Warnings and console text that survive a console with a narrow encoding."""

from __future__ import annotations

import sys
import warnings
from typing import Any, TextIO


def _safe_text(value: Any, stream: TextIO) -> str:
    """Return ``str(value)``, escaping the characters ``stream`` cannot encode."""
    text = str(value)
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    return text


def safe_warn(
    message: Any,
    category: type[Warning] = UserWarning,
    stacklevel: int = 1,
) -> None:
    """Issue a warning and copy it to the run report, if one is configured."""
    from ._logging import record_python_warning

    record_python_warning(message, category)
    warnings.warn(
        _safe_text(message, sys.stderr),
        category=category,
        stacklevel=stacklevel + 1,
    )
