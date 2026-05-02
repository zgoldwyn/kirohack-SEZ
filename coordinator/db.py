"""Supabase client initialization and database query helpers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


class RecordNotFoundError(DatabaseError):
    """Raised when a query returns no matching records."""


class DuplicateRecordError(DatabaseError):
    """Raised when an insert violates a uniqueness constraint."""


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Client | None = None


def get_client() -> Client:
    """Return the Supabase client, creating it on first call."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise DatabaseError(
                "Missing Supabase credentials",
                detail="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set",
            )
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def insert(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a single row and return the created record.

    Raises:
        DuplicateRecordError: if a uniqueness constraint is violated.
        DatabaseError: on any other Supabase/Postgres error.
    """
    try:
        response = get_client().table(table).insert(data).execute()
    except Exception as exc:
        _handle_exception(exc)

    if not response.data:
        raise DatabaseError(f"Insert into '{table}' returned no data")
    return response.data[0]


def select(
    table: str,
    columns: str = "*",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select rows from *table*, optionally filtered by equality conditions.

    Returns an empty list when no rows match.
    """
    try:
        query = get_client().table(table).select(columns)
        for col, val in (filters or {}).items():
            query = query.eq(col, val)
        response = query.execute()
    except Exception as exc:
        _handle_exception(exc)

    return response.data or []


def select_one(
    table: str,
    columns: str = "*",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select exactly one row.  Raises *RecordNotFoundError* if none match."""
    rows = select(table, columns, filters)
    if not rows:
        raise RecordNotFoundError(
            f"No record found in '{table}'",
            detail=filters,
        )
    return rows[0]


def update(
    table: str,
    data: dict[str, Any],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Update rows matching *filters* and return the updated records.

    Raises:
        DatabaseError: on any Supabase/Postgres error.
    """
    try:
        query = get_client().table(table).update(data)
        for col, val in filters.items():
            query = query.eq(col, val)
        response = query.execute()
    except Exception as exc:
        _handle_exception(exc)

    return response.data or []


def update_one(
    table: str,
    data: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Update exactly one row.  Raises *RecordNotFoundError* if none match."""
    rows = update(table, data, filters)
    if not rows:
        raise RecordNotFoundError(
            f"No record updated in '{table}'",
            detail=filters,
        )
    return rows[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _handle_exception(exc: Exception) -> None:
    """Translate Supabase / PostgREST errors into application exceptions."""
    msg = str(exc)
    # PostgREST surfaces unique-violation as code 23505
    if "23505" in msg or "duplicate key" in msg.lower():
        raise DuplicateRecordError("Duplicate record", detail=msg) from exc
    raise DatabaseError(f"Database operation failed: {msg}", detail=msg) from exc
