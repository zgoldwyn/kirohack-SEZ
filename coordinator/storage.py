"""Supabase Storage helpers for binary blob operations.

Provides a unified interface for uploading, downloading, listing, and
deleting binary blobs in Supabase Storage buckets.  Used by
``param_server.py`` (model parameters / checkpoints) and
``aggregator.py`` (gradient payloads).

All functions communicate with the Supabase Storage HTTP API via *httpx*.

Requirements: 7.1, 13.1, 13.2
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from coordinator.db import DatabaseError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket name constants
# ---------------------------------------------------------------------------

CHECKPOINTS_BUCKET = "checkpoints"
PARAMETERS_BUCKET = "parameters"
GRADIENTS_BUCKET = "gradients"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StorageError(DatabaseError):
    """Raised when a Supabase Storage operation fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_storage_config() -> tuple[str, str]:
    """Return ``(supabase_url, supabase_key)`` from environment variables.

    Raises :class:`StorageError` if either variable is missing or empty.
    """
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    if not supabase_url or not key:
        raise StorageError(
            "SUPABASE_URL and SUPABASE_KEY must be set for storage operations"
        )
    return supabase_url, key


def _auth_headers(key: str) -> dict[str, str]:
    """Standard auth headers for Supabase Storage requests."""
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_blob(bucket: str, path: str, data: bytes) -> None:
    """Upload binary data to Supabase Storage, overwriting if it exists.

    Uses an *upsert* approach: first attempts a POST (create); if the
    object already exists (any non-success response), falls back to a
    PUT (update).

    Parameters
    ----------
    bucket:
        The storage bucket name (e.g. ``"parameters"``, ``"gradients"``).
    path:
        The object path within the bucket (e.g. ``"{job_id}/current.pt"``).
    data:
        The raw bytes to upload.

    Raises
    ------
    StorageError
        If both create and update attempts fail.
    """
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)
    headers["Content-Type"] = "application/octet-stream"

    object_url = f"{supabase_url}/storage/v1/object/{bucket}/{path}"

    try:
        # Try create first
        resp = httpx.post(
            object_url,
            headers=headers,
            content=data,
            timeout=60.0,
        )
        if resp.status_code in (200, 201):
            return

        # If the object already exists, update it
        resp = httpx.put(
            object_url,
            headers=headers,
            content=data,
            timeout=60.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise StorageError(
            f"Failed to upload blob to {bucket}/{path}: "
            f"{exc.response.status_code} {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise StorageError(
            f"Failed to upload blob to {bucket}/{path}: {exc}"
        ) from exc


def download_blob(bucket: str, path: str) -> bytes:
    """Download binary data from Supabase Storage.

    Parameters
    ----------
    bucket:
        The storage bucket name.
    path:
        The object path within the bucket.

    Returns
    -------
    bytes
        The raw bytes of the stored object.

    Raises
    ------
    StorageError
        If the download fails.
    """
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)

    object_url = f"{supabase_url}/storage/v1/object/{bucket}/{path}"

    try:
        resp = httpx.get(
            object_url,
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPStatusError as exc:
        raise StorageError(
            f"Failed to download blob from {bucket}/{path}: "
            f"{exc.response.status_code} {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise StorageError(
            f"Failed to download blob from {bucket}/{path}: {exc}"
        ) from exc


def delete_blob(bucket: str, path: str) -> None:
    """Delete a single blob from Supabase Storage.

    Uses the Supabase Storage ``/object/{bucket}`` DELETE endpoint which
    accepts a JSON body with a list of prefixes to remove.

    This is a best-effort operation: if the blob does not exist (404) the
    call succeeds silently.

    Parameters
    ----------
    bucket:
        The storage bucket name.
    path:
        The object path within the bucket.

    Raises
    ------
    StorageError
        If the delete fails with an unexpected status code.
    """
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)
    headers["Content-Type"] = "application/json"

    url = f"{supabase_url}/storage/v1/object/{bucket}"

    try:
        resp = httpx.delete(
            url,
            headers=headers,
            json={"prefixes": [path]},
            timeout=30.0,
        )
        # 200, 201, 204, or 404 are all acceptable
        if resp.status_code not in (200, 201, 204, 404):
            logger.warning(
                "Failed to delete blob %s/%s: %s %s",
                bucket,
                path,
                resp.status_code,
                resp.text,
            )
    except Exception as exc:
        raise StorageError(
            f"Failed to delete blob from {bucket}/{path}: {exc}"
        ) from exc


def list_blobs(bucket: str, prefix: str) -> list[str]:
    """List blob names under a prefix in a bucket.

    Parameters
    ----------
    bucket:
        The storage bucket name.
    prefix:
        The path prefix to list under (e.g. ``"{job_id}/round_1/"``).

    Returns
    -------
    list[str]
        A list of full object paths (prefix + name) for each blob found.
        Returns an empty list on failure or if no blobs match.
    """
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)
    headers["Content-Type"] = "application/json"

    url = f"{supabase_url}/storage/v1/object/list/{bucket}"

    try:
        resp = httpx.post(
            url,
            headers=headers,
            json={"prefix": prefix, "limit": 1000},
            timeout=30.0,
        )
    except Exception as exc:
        logger.warning(
            "Failed to list blobs under %s/%s: %s", bucket, prefix, exc
        )
        return []

    if resp.status_code != 200:
        logger.warning(
            "Failed to list blobs under %s/%s: %s %s",
            bucket,
            prefix,
            resp.status_code,
            resp.text,
        )
        return []

    items = resp.json()
    if not isinstance(items, list):
        return []

    return [
        f"{prefix}{item['name']}"
        for item in items
        if isinstance(item, dict) and "name" in item
    ]
