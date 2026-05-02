"""Supabase Storage helpers for checkpoint upload URL generation."""

from __future__ import annotations

from typing import Any

from coordinator.db import DatabaseError, get_client

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


def create_signed_upload_url(job_id: str, task_id: str) -> dict[str, Any]:
    """Generate a signed upload URL for a task checkpoint.

    The URL follows the path convention ``{job_id}/{task_id}/final.pt`` inside
    the ``checkpoints`` bucket.  Signed upload URLs are valid for 2 hours
    (Supabase default) and allow the holder to upload a file without further
    authentication.

    Workers use these URLs to upload checkpoint files directly to Supabase
    Storage — they never hold privileged Supabase credentials.

    Returns:
        A dict with ``signed_url`` (the upload URL) and ``path`` (the storage
        path) keys.  The ``token`` field from Supabase is included as well so
        callers can use ``upload_to_signed_url`` if needed.

    Raises:
        StorageError: if the signed URL could not be created.
    """
    storage_path = f"{job_id}/{task_id}/final.pt"

    try:
        response = (
            get_client()
            .storage
            .from_(CHECKPOINTS_BUCKET)
            .create_signed_upload_url(storage_path)
        )
    except Exception as exc:
        raise StorageError(
            f"Failed to create signed upload URL for '{storage_path}'",
            detail=str(exc),
        ) from exc

    # The response is a dict with 'signed_url' and 'token' (among others).
    if not response or "signed_url" not in (response if isinstance(response, dict) else {}):
        # Depending on the supabase-py version the shape may vary; handle
        # both dict and object-with-attributes.
        signed_url = getattr(response, "signed_url", None)
        token = getattr(response, "token", None)
        if signed_url is None:
            raise StorageError(
                f"Unexpected response when creating signed upload URL for '{storage_path}'",
                detail=str(response),
            )
        return {
            "signed_url": signed_url,
            "token": token,
            "path": storage_path,
        }

    return {
        "signed_url": response["signed_url"],
        "token": response.get("token"),
        "path": storage_path,
    }
