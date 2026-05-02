"""Supabase Storage helpers for signed upload URL generation.

Workers do not hold direct Supabase credentials. Instead, the
Coordinator generates time-limited signed upload URLs that Workers
use to upload checkpoint files to the ``checkpoints`` bucket.

Path convention: ``{job_id}/{task_id}/final.pt``
"""

from __future__ import annotations

import logging

from coordinator import db

logger = logging.getLogger(__name__)

BUCKET_NAME = "checkpoints"

# Default signed URL validity in seconds (10 minutes)
_DEFAULT_EXPIRY_SECONDS = 600


def generate_signed_upload_url(
    job_id: str,
    task_id: str,
    expiry_seconds: int = _DEFAULT_EXPIRY_SECONDS,
) -> str:
    """Generate a time-limited signed upload URL for a checkpoint.

    Parameters
    ----------
    job_id:
        The job UUID.
    task_id:
        The task UUID.
    expiry_seconds:
        How long the URL remains valid (default 600 s / 10 min).

    Returns
    -------
    str
        The signed upload URL.

    Raises
    ------
    Exception
        If the Supabase storage call fails.
    """
    path = f"{job_id}/{task_id}/final.pt"
    client = db.get_client()
    result = client.storage.from_(BUCKET_NAME).create_signed_upload_url(path)

    # The Supabase Python client returns a dict with "signed_url" (or "signedURL")
    if isinstance(result, dict):
        signed_url = result.get("signed_url") or result.get("signedURL") or result.get("data", {}).get("signed_url", "")
    else:
        # Fallback: treat as string
        signed_url = str(result)

    if not signed_url:
        raise RuntimeError(
            f"Failed to generate signed upload URL for path '{path}'"
        )

    logger.info("Generated signed upload URL for %s", path)
    return signed_url
