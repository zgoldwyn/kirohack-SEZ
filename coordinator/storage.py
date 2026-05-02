"""Supabase Storage helpers for checkpoint upload URL generation.

Uses httpx to call the Supabase Storage API directly.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from coordinator.db import DatabaseError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINTS_BUCKET = "checkpoints"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StorageError(DatabaseError):
    """Raised when a Supabase Storage operation fails."""


# ---------------------------------------------------------------------------
# Signed upload URL generation
# ---------------------------------------------------------------------------


def generate_signed_upload_url(job_id: str, task_id: str) -> dict[str, Any]:
    """Generate a signed upload URL for a task checkpoint.

    The URL follows the path convention ``{job_id}/{task_id}/final.pt`` inside
    the ``checkpoints`` bucket.

    Returns:
        A dict with ``signed_url`` and ``path`` keys.

    Raises:
        StorageError: if the signed URL could not be created.
    """
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")

    if not supabase_url or not key:
        raise StorageError(
            "SUPABASE_URL and SUPABASE_KEY must be set for storage operations"
        )

    storage_path = f"{job_id}/{task_id}/final.pt"
    api_url = f"{supabase_url}/storage/v1/object/upload/sign/{CHECKPOINTS_BUCKET}/{storage_path}"

    try:
        resp = httpx.post(
            api_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise StorageError(
            f"Failed to create signed upload URL for '{storage_path}': {exc}"
        ) from exc

    signed_url = data.get("url") or data.get("signedURL") or data.get("signed_url")
    if not signed_url:
        raise StorageError(
            f"Unexpected response when creating signed upload URL: {data}"
        )

    # The Supabase Storage API returns a relative path like
    # /object/upload/sign/bucket/path?token=...
    # We need to prepend the full storage API base URL
    if signed_url.startswith("/"):
        signed_url = f"{supabase_url}/storage/v1{signed_url}"

    return {
        "signed_url": signed_url,
        "token": data.get("token"),
        "path": storage_path,
    }
