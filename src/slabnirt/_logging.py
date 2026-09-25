"""Package logging with default console output and optional UTF-8 reports."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ._console import _safe_text

_LOGGER = logging.getLogger("slabnirt")
_CONFIGURATION_LOCK = threading.RLock()
_CONFIGURED = False


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "slabnirt_report_only", False)


class _DynamicConsoleHandler(logging.Handler):
    """Write to the current stdout, escaping characters the console cannot encode."""

    terminator = "\n"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = sys.stdout
            message = self.format(record)
            stream.write(_safe_text(message, stream))
            stream.write(_safe_text(self.terminator, stream))
            stream.flush()
        except Exception:
            self.handleError(record)


def _resolve_level(level: int | str) -> int:
    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), None)
        if not isinstance(resolved, int):
            raise ValueError(f"level: unknown logging level {level!r}.")
        return resolved
    if not isinstance(level, int):
        raise TypeError("level must be an integer or a standard logging-level name.")
    return level


def _mark_owned(handler: logging.Handler) -> logging.Handler:
    handler.slabnirt_owned = True
    return handler


def configure_logging(
    *,
    console: bool = True,
    report_file: str | Path | None = None,
    level: int | str = logging.INFO,
) -> None:
    """Configure the package's progress messages and the optional run report.

    A later call closes and replaces the handlers of the earlier one, so each
    run can have its own report without duplicate messages. Importing
    :mod:`slabnirt` does not call this function or create a file; without a
    call, the first package message or warning sets up console output only.
    Package messages are not passed on to the root logger.

    Parameters
    ----------
    console : bool, default True
        Print the messages to standard output.
    report_file : str or Path, optional
        Text file (UTF-8) that receives the messages and the package's
        warnings, each line with a UTC time stamp and the level. An existing
        file is overwritten; a missing folder is created.
    level : int or str, default logging.INFO
        Lowest level written, as an integer or a name such as 'WARNING'.

    Raises
    ------
    ValueError
        For an unknown level name.
    TypeError
        For a level that is neither an integer nor a string.
    """

    global _CONFIGURED

    resolved_level = _resolve_level(level)
    new_handlers: list[logging.Handler] = []

    if console:
        console_handler = _mark_owned(_DynamicConsoleHandler())
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.addFilter(_ConsoleFilter())
        new_handlers.append(console_handler)

    if report_file is not None:
        report_path = Path(report_file).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _mark_owned(
            logging.FileHandler(
                report_path,
                mode="w",
                encoding="utf-8",
            )
        )
        file_handler.setLevel(resolved_level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)
        new_handlers.append(file_handler)

    with _CONFIGURATION_LOCK:
        old_handlers = [
            handler for handler in _LOGGER.handlers if getattr(handler, "slabnirt_owned", False)
        ]
        for handler in old_handlers:
            _LOGGER.removeHandler(handler)
            handler.close()

        _LOGGER.setLevel(resolved_level)
        _LOGGER.propagate = False
        for handler in new_handlers:
            _LOGGER.addHandler(handler)
        _CONFIGURED = True


def _ensure_configured() -> None:
    if _CONFIGURED:
        return
    with _CONFIGURATION_LOCK:
        if not _CONFIGURED:
            configure_logging()


def log_info(message: Any, *args: Any) -> None:
    """Log a progress message through the package logger."""
    _ensure_configured()
    _LOGGER.info(message, *args)


def log_warning(message: Any, *args: Any) -> None:
    """Log a warning message to the console and the run report."""
    _ensure_configured()
    _LOGGER.warning(message, *args)


def record_python_warning(message: Any, category: type[Warning]) -> None:
    """Record a Python warning in reports without duplicating console output."""
    _ensure_configured()
    _LOGGER.warning(
        "%s: %s",
        category.__name__,
        message,
        extra={"slabnirt_report_only": True},
    )
