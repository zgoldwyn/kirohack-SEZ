"""Structured logging configuration for the Coordinator.

Configures Python's ``logging`` module with a structured format that
includes timestamp, level, logger name, and event type.  Call
``setup_logging()`` once at application startup (before any log
messages are emitted).

Requirements: 11.1, 11.3
"""

from __future__ import annotations

import logging
import sys

# Default log format — structured with timestamp, level, and logger name.
# Individual log calls add event-specific context via the message and extra
# fields.
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a structured console handler.

    This sets up a single ``StreamHandler`` writing to *stderr* with the
    project-wide format.  It is safe to call more than once — duplicate
    handlers are avoided by clearing existing handlers first.

    Args:
        level: The minimum log level.  Defaults to ``logging.INFO``.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicates when
    # ``setup_logging`` is called more than once (e.g. in tests).
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
