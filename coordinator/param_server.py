"""Parameter Server module — Global Model state management.

Manages the lifecycle of the Global_Model parameters stored in Supabase
Storage.  The Coordinator uses this module to:

- Initialize a fresh model from a job configuration and upload its
  ``state_dict`` to storage.
- Serve the current parameters to Workers each training round.
- Persist updated parameters after gradient aggregation.
- Snapshot checkpoints at specific training rounds.

All serialization uses PyTorch's ``torch.save`` / ``torch.load`` with the
``state_dict`` convention.  Binary blobs are held in memory via
``io.BytesIO`` — no temporary files are created.

Storage is accessed through the Supabase Storage HTTP API using *httpx*,
following the same pattern as ``coordinator/storage.py``.

Requirements: 4.2, 7.1, 7.2, 7.5, 13.4, 13.5
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import torch

from coordinator.db import DatabaseError, insert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAMETERS_BUCKET = "parameters"
CHECKPOINTS_BUCKET = "checkpoints"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParamServerError(DatabaseError):
    """Raised when a parameter server operation fails."""


# ---------------------------------------------------------------------------
# Internal helpers — Supabase Storage HTTP API
# ---------------------------------------------------------------------------


def _get_storage_config() -> tuple[str, str]:
    """Return ``(supabase_url, supabase_key)`` from environment variables.

    Raises :class:`ParamServerError` if either variable is missing.
    """
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    if not supabase_url or not key:
        raise ParamServerError(
            "SUPABASE_URL and SUPABASE_KEY must be set for storage operations"
        )
    return supabase_url, key


def _auth_headers(key: str) -> dict[str, str]:
    """Standard auth headers for Supabase Storage requests."""
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }


def _upload_blob(bucket: str, path: str, data: bytes) -> None:
    """Upload a binary blob to Supabase Storage, overwriting if it exists.

    Uses an *upsert* approach: first attempts a POST (create); if the
    object already exists (HTTP 409 / 400 with "Duplicate"), falls back
    to a PUT (update).

    Raises :class:`ParamServerError` on failure.
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
        raise ParamServerError(
            f"Failed to upload blob to {bucket}/{path}: "
            f"{exc.response.status_code} {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise ParamServerError(
            f"Failed to upload blob to {bucket}/{path}: {exc}"
        ) from exc


def _download_blob(bucket: str, path: str) -> bytes:
    """Download a binary blob from Supabase Storage.

    Returns the raw bytes of the stored object.

    Raises :class:`ParamServerError` on failure.
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
        raise ParamServerError(
            f"Failed to download blob from {bucket}/{path}: "
            f"{exc.response.status_code} {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise ParamServerError(
            f"Failed to download blob from {bucket}/{path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_state_dict(state_dict: dict[str, Any]) -> bytes:
    """Serialize a PyTorch ``state_dict`` to bytes using ``torch.save``.

    Uses an in-memory ``BytesIO`` buffer to avoid temporary files.
    """
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _deserialize_state_dict(data: bytes) -> dict[str, Any]:
    """Deserialize bytes into a PyTorch ``state_dict`` using ``torch.load``.

    Uses an in-memory ``BytesIO`` buffer and ``weights_only=True`` for
    safety.
    """
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_model(job_id: str, job_config: Any) -> str:
    """Create an initial model from *job_config* and upload its parameters.

    Uses ``worker.models.build_model()`` to construct the model, then
    serializes its ``state_dict`` and uploads to Supabase Storage at
    ``parameters/{job_id}/current.pt``.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    job_config:
        A ``JobConfig`` (or compatible object) with ``dataset_name``,
        ``model_type``, and ``hyperparameters`` attributes.

    Returns
    -------
    str
        The storage path where the initial parameters were uploaded
        (``parameters/{job_id}/current.pt``).

    Raises
    ------
    ParamServerError
        If model creation or upload fails.
    """
    # Import here to avoid circular dependency at module level — the
    # worker package is a peer, not a dependency of the coordinator.
    from worker.models import build_model

    try:
        model = build_model(
            dataset_name=job_config.dataset_name,
            model_type=job_config.model_type,
            hidden_layers=list(job_config.hyperparameters.hidden_layers),
            activation=job_config.hyperparameters.activation,
        )
    except Exception as exc:
        raise ParamServerError(
            f"Failed to build initial model for job {job_id}: {exc}"
        ) from exc

    state_dict = model.state_dict()
    data = _serialize_state_dict(state_dict)

    storage_path = f"{job_id}/current.pt"
    _upload_blob(PARAMETERS_BUCKET, storage_path, data)

    logger.info(
        "Initialized global model for job %s (%d bytes, %d parameters)",
        job_id,
        len(data),
        sum(p.numel() for p in model.parameters()),
    )

    return f"{PARAMETERS_BUCKET}/{storage_path}"


def get_parameters(job_id: str) -> bytes:
    """Download the current Global_Model parameters for *job_id*.

    Parameters
    ----------
    job_id:
        The database UUID of the job.

    Returns
    -------
    bytes
        The raw serialized ``state_dict`` bytes (``torch.save`` format).

    Raises
    ------
    ParamServerError
        If the download fails.
    """
    storage_path = f"{job_id}/current.pt"
    data = _download_blob(PARAMETERS_BUCKET, storage_path)

    logger.debug(
        "Downloaded parameters for job %s (%d bytes)", job_id, len(data)
    )
    return data


def update_parameters(job_id: str, new_state_dict: dict[str, Any]) -> None:
    """Serialize and upload an updated ``state_dict`` for *job_id*.

    Overwrites the existing parameters at
    ``parameters/{job_id}/current.pt``.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    new_state_dict:
        The updated model ``state_dict`` to persist.

    Raises
    ------
    ParamServerError
        If serialization or upload fails.
    """
    data = _serialize_state_dict(new_state_dict)
    storage_path = f"{job_id}/current.pt"
    _upload_blob(PARAMETERS_BUCKET, storage_path, data)

    logger.info(
        "Updated global model parameters for job %s (%d bytes)",
        job_id,
        len(data),
    )


def store_checkpoint(job_id: str, round_number: int) -> str:
    """Copy current parameters to a checkpoint and record the artifact.

    Downloads the current parameters from
    ``parameters/{job_id}/current.pt``, re-uploads them to
    ``checkpoints/{job_id}/round_{round_number}.pt``, and inserts an
    artifact record in the database.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    round_number:
        The training round this checkpoint corresponds to.

    Returns
    -------
    str
        The storage path of the checkpoint
        (``checkpoints/{job_id}/round_{N}.pt``).

    Raises
    ------
    ParamServerError
        If download, upload, or database insert fails.
    """
    # Download current parameters
    data = _download_blob(PARAMETERS_BUCKET, f"{job_id}/current.pt")

    # Upload as checkpoint
    checkpoint_path = f"{job_id}/round_{round_number}.pt"
    _upload_blob(CHECKPOINTS_BUCKET, checkpoint_path, data)

    # Insert artifact record
    now = datetime.now(timezone.utc).isoformat()
    try:
        insert(
            "artifacts",
            {
                "job_id": job_id,
                "task_id": None,
                "node_id": None,
                "artifact_type": "checkpoint",
                "storage_path": f"{CHECKPOINTS_BUCKET}/{checkpoint_path}",
                "round_number": round_number,
                "size_bytes": len(data),
                "created_at": now,
            },
        )
    except Exception as exc:
        raise ParamServerError(
            f"Failed to insert artifact record for checkpoint "
            f"{checkpoint_path}: {exc}"
        ) from exc

    logger.info(
        "Stored checkpoint for job %s round %d (%d bytes) at %s",
        job_id,
        round_number,
        len(data),
        checkpoint_path,
    )

    return f"{CHECKPOINTS_BUCKET}/{checkpoint_path}"
