"""Supabase client initialization and database query helpers.

Provides a singleton Supabase client and thin helper functions for
common database operations (insert, select, update with filters).
Wraps Supabase errors into consistent application exceptions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from supabase import Client, create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DatabaseError(Exception):
    """Raised when a database operation fails."""


class RecordNotFoundError(DatabaseError):
    """Raised when an expected record is not found."""


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: Client | None = None


def get_client() -> Client:
    """Return the singleton Supabase client, creating it on first call."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
            )
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def insert(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a single row and return the inserted record.

    Raises DatabaseError on failure.
    """
    try:
        response = get_client().table(table).insert(data).execute()
        if response.data:
            return response.data[0]
        raise DatabaseError(f"Insert into '{table}' returned no data")
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Insert into '{table}' failed: {exc}") from exc


def select(
    table: str,
    columns: str = "*",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select rows from a table with optional equality filters.

    Returns a (possibly empty) list of matching records.
    """
    try:
        query = get_client().table(table).select(columns)
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        response = query.execute()
        return response.data or []
    except Exception as exc:
        raise DatabaseError(f"Select from '{table}' failed: {exc}") from exc


def select_one(
    table: str,
    columns: str = "*",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select exactly one row. Raises RecordNotFoundError if none found."""
    rows = select(table, columns, filters)
    if not rows:
        raise RecordNotFoundError(
            f"No record found in '{table}' matching {filters}"
        )
    return rows[0]


def update(
    table: str,
    data: dict[str, Any],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Update rows matching the given equality filters.

    Returns the list of updated records.
    """
    try:
        query = get_client().table(table).update(data)
        for col, val in filters.items():
            query = query.eq(col, val)
        response = query.execute()
        return response.data or []
    except Exception as exc:
        raise DatabaseError(f"Update on '{table}' failed: {exc}") from exc
