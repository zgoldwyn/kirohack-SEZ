"""Supabase PostgREST client via httpx.

Uses httpx to call the Supabase PostgREST API directly, bypassing
supabase-py's JWT key format validation (which rejects the newer
sb_secret_ / sb_publishable_ key formats).

Provides the same interface as before: insert, select, select_one,
update — so all callers continue to work without modification.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DatabaseError(Exception):
    """Raised when a database operation fails."""


class RecordNotFoundError(DatabaseError):
    """Raised when an expected record is not found."""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

_http_client: httpx.Client | None = None


def _get_base_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_URL environment variable is not set")
    return url.rstrip("/")


def _get_key() -> str:
    key = os.getenv("SUPABASE_KEY")
    if not key:
        raise RuntimeError("SUPABASE_KEY environment variable is not set")
    return key


def _get_http_client() -> httpx.Client:
    """Return a singleton httpx client with auth headers pre-configured."""
    global _http_client
    if _http_client is None:
        key = _get_key()
        _http_client = httpx.Client(
            base_url=f"{_get_base_url()}/rest/v1",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )
    return _http_client


def get_client():
    """Backward-compat shim for storage.py. Returns the httpx client."""
    return _get_http_client()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def insert(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a single row and return the inserted record.

    Raises DatabaseError on failure.
    """
    try:
        resp = _get_http_client().post(
            f"/{table}",
            json=data,
            headers={"Prefer": "return=representation"},
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            return rows[0]
        raise DatabaseError(f"Insert into '{table}' returned no data")
    except DatabaseError:
        raise
    except httpx.HTTPStatusError as exc:
        raise DatabaseError(
            f"Insert into '{table}' failed: {exc.response.status_code} {exc.response.text}"
        ) from exc
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
        params: dict[str, str] = {"select": columns}
        if filters:
            for col, val in filters.items():
                params[col] = f"eq.{val}"
        resp = _get_http_client().get(f"/{table}", params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise DatabaseError(
            f"Select from '{table}' failed: {exc.response.status_code} {exc.response.text}"
        ) from exc
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
        params: dict[str, str] = {}
        for col, val in filters.items():
            params[col] = f"eq.{val}"
        resp = _get_http_client().patch(
            f"/{table}",
            json=data,
            params=params,
            headers={"Prefer": "return=representation"},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise DatabaseError(
            f"Update on '{table}' failed: {exc.response.status_code} {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise DatabaseError(f"Update on '{table}' failed: {exc}") from exc


def delete(
    table: str,
    filters: dict[str, Any],
) -> None:
    """Delete rows matching the given equality filters."""
    try:
        params: dict[str, str] = {}
        for col, val in filters.items():
            params[col] = f"eq.{val}"
        resp = _get_http_client().delete(f"/{table}", params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DatabaseError(
            f"Delete from '{table}' failed: {exc.response.status_code} {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise DatabaseError(f"Delete from '{table}' failed: {exc}") from exc
