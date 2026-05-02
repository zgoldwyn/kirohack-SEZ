"""Checkpoint upload via signed URL.

Uploads final checkpoint files to Supabase Storage using
Coordinator-issued signed upload URLs.  The Worker never holds
direct Supabase credentials — all storage access goes through
time-limited signed URLs obtained from the Coordinator API.

Typical flow
------------
1. Training completes → checkpoint saved to a local temp file.
2. ``upload_checkpoint`` requests a signed URL from the Coordinator
   (via ``reporter.request_upload_url``).
3. The file is uploaded with a plain HTTP ``PUT`` to the signed URL.
4. On success the storage path is returned so the caller can pass it
   to ``reporter.complete_task``.
5. On failure the task is reported as failed to the Coordinator and
   the error is re-raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from worker.reporter import Reporter

logger = logging.getLogger(__name__)

# Default timeout for checkpoint uploads (large files may need more time)
_DEFAULT_UPLOAD_TIMEOUT = 120.0  # seconds


class CheckpointUploadError(Exception):
    """Raised when a checkpoint upload fails."""


@dataclass
class StorageClient:
    """Handles checkpoint uploads to Supabase Storage via signed URLs.

    Parameters
    ----------
    reporter:
        The :class:`~worker.reporter.Reporter` instance used to request
        signed upload URLs and to report task failures.
    upload_timeout:
        Timeout in seconds for the HTTP PUT upload request.
    """

    reporter: Reporter
    upload_timeout: float = _DEFAULT_UPLOAD_TIMEOUT
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (or lazily create) an ``httpx.AsyncClient`` for uploads."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.upload_timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def upload_checkpoint(
        self,
        task_id: str,
        local_path: str | Path,
    ) -> str:
        """Upload a checkpoint file and return the storage path.

        Steps
        -----
        1. Request a signed upload URL from the Coordinator.
        2. Read the local file and ``PUT`` it to the signed URL.
        3. Return the storage path (``{job_id}/{task_id}/final.pt``).

        On any failure the task is reported as failed to the Coordinator
        via ``reporter.fail_task`` and a :class:`CheckpointUploadError`
        is raised.

        Parameters
        ----------
        task_id:
            The task ID whose checkpoint is being uploaded.
        local_path:
            Path to the local checkpoint file (e.g. a ``.pt`` file).

        Returns
        -------
        str
            The signed URL that was used for the upload.  The caller
            typically passes the Coordinator-known storage path (returned
            alongside the URL) to ``reporter.complete_task``, but the
            signed URL itself confirms the upload target.

        Raises
        ------
        CheckpointUploadError
            If the upload fails for any reason (network, HTTP error,
            file not found, etc.).  The task will already have been
            reported as failed to the Coordinator.
        """
        local_path = Path(local_path)

        # --- Validate local file exists -----------------------------------
        if not local_path.is_file():
            error_msg = f"Checkpoint file not found: {local_path}"
            logger.error(error_msg)
            await self._report_failure(task_id, error_msg)
            raise CheckpointUploadError(error_msg)

        # --- Request signed upload URL from Coordinator -------------------
        try:
            signed_url = await self.reporter.request_upload_url(task_id)
        except Exception as exc:
            error_msg = (
                f"Failed to obtain signed upload URL for task {task_id}: {exc}"
            )
            logger.error(error_msg)
            # Don't report failure here — if we can't reach the Coordinator
            # to get a URL, reporting failure will likely also fail.  The
            # caller should handle this at a higher level.
            raise CheckpointUploadError(error_msg) from exc

        # --- Upload the file via HTTP PUT ---------------------------------
        try:
            file_data = local_path.read_bytes()
            logger.info(
                "Uploading checkpoint for task %s (%d bytes) to signed URL",
                task_id,
                len(file_data),
            )

            client = await self._get_client()
            response = await client.put(
                signed_url,
                content=file_data,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()

            logger.info(
                "Checkpoint upload succeeded for task %s (HTTP %d)",
                task_id,
                response.status_code,
            )
            return signed_url

        except Exception as exc:
            error_msg = (
                f"Checkpoint upload failed for task {task_id}: {exc}"
            )
            logger.error(error_msg)
            await self._report_failure(task_id, error_msg)
            raise CheckpointUploadError(error_msg) from exc

    async def _report_failure(self, task_id: str, error_message: str) -> None:
        """Best-effort report of task failure to the Coordinator."""
        try:
            await self.reporter.fail_task(task_id, error_message=error_message)
            logger.info("Reported upload failure for task %s to Coordinator", task_id)
        except Exception as report_exc:
            # If we can't even report the failure, log it and move on.
            logger.error(
                "Failed to report upload failure for task %s: %s",
                task_id,
                report_exc,
            )
