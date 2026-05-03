"""Gradient aggregation, job completion, and job failure logic.

Implements the core aggregation loop for collaborative distributed
training.  After all active Workers submit their gradient updates for a
training round, this module:

1. Downloads each Worker's gradient tensors from Supabase Storage.
2. Computes the element-wise mean of all gradient ``state_dict`` tensors.
3. Applies an SGD step to the current Global_Model parameters.
4. Uploads the updated parameters via the parameter server.
5. Records per-round global and per-worker metrics.
6. Advances the job to the next round (or completes/fails it).

Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5, 14.3
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import torch

from coordinator import db
from coordinator.constants import (
    JobStatus,
    TaskStatus,
    TrainingRoundStatus,
)
from coordinator import barrier as barrier_mod
from coordinator import param_server

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage bucket for gradient payloads
# ---------------------------------------------------------------------------

GRADIENTS_BUCKET = "gradients"


# ---------------------------------------------------------------------------
# Internal helpers — Supabase Storage HTTP API for gradients
# ---------------------------------------------------------------------------


def _get_storage_config() -> tuple[str, str]:
    """Return ``(supabase_url, supabase_key)`` from environment variables."""
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    if not supabase_url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set for storage operations"
        )
    return supabase_url, key


def _auth_headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }


def _download_blob(bucket: str, path: str) -> bytes:
    """Download a binary blob from Supabase Storage."""
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)
    object_url = f"{supabase_url}/storage/v1/object/{bucket}/{path}"

    resp = httpx.get(object_url, headers=headers, timeout=60.0)
    resp.raise_for_status()
    return resp.content


def _delete_blob(bucket: str, path: str) -> None:
    """Delete a single blob from Supabase Storage.

    Uses the Supabase Storage ``/object/{bucket}`` DELETE endpoint which
    accepts a JSON body with a list of prefixes to remove.
    """
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)
    headers["Content-Type"] = "application/json"

    url = f"{supabase_url}/storage/v1/object/{bucket}"
    resp = httpx.delete(url, headers=headers, json={"prefixes": [path]}, timeout=30.0)
    # 200 or 404 are both acceptable (file may already be gone)
    if resp.status_code not in (200, 201, 204, 404):
        logger.warning(
            "Failed to delete blob %s/%s: %s %s",
            bucket,
            path,
            resp.status_code,
            resp.text,
        )


def _list_blobs(bucket: str, prefix: str) -> list[str]:
    """List blob names under *prefix* in *bucket*.

    Returns a list of object names (relative to the bucket root).
    """
    supabase_url, key = _get_storage_config()
    headers = _auth_headers(key)
    headers["Content-Type"] = "application/json"

    url = f"{supabase_url}/storage/v1/object/list/{bucket}"
    resp = httpx.post(
        url,
        headers=headers,
        json={"prefix": prefix, "limit": 1000},
        timeout=30.0,
    )
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
        f"{prefix}{item['name']}" if prefix and not prefix.endswith("/")
        else f"{prefix}{item['name']}"
        for item in items
        if isinstance(item, dict) and "name" in item
    ]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _deserialize_state_dict(data: bytes) -> dict[str, Any]:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_round(job_id: str, round_number: int) -> None:
    """Aggregate gradients for a completed training round.

    This is the core aggregation routine called when the synchronization
    barrier is met (all active Workers have submitted their gradient
    updates for *round_number*).

    Steps
    -----
    1. Mark the round as ``aggregating``.
    2. Load all gradient submissions for the round from the database.
    3. Download each gradient tensor payload from Supabase Storage.
    4. Compute the element-wise mean of all gradient ``state_dict`` tensors.
    5. Load the current Global_Model parameters.
    6. Apply an SGD step: ``param -= lr * mean_grad``.
    7. Upload the updated parameters.
    8. Store per-worker local metrics and per-round global metrics.
    9. Mark the round as ``completed``.
    10. Advance the job's ``current_round``.
    11. Create the next round record (or complete the job if done).
    12. Clean up gradient storage for this round.
    """

    # --- 1. Mark round as aggregating ---
    round_records = db.select(
        "training_rounds",
        filters={"job_id": job_id, "round_number": round_number},
    )
    if round_records:
        db.update(
            "training_rounds",
            {"status": TrainingRoundStatus.AGGREGATING.value},
            filters={"id": round_records[0]["id"]},
        )

    # --- 2. Load gradient submissions ---
    submissions = db.select(
        "gradient_submissions",
        filters={"job_id": job_id, "round_number": round_number},
    )

    if not submissions:
        logger.warning(
            "event=aggregation_skipped | job_id=%s | round=%d | reason=no_submissions",
            job_id,
            round_number,
        )
        return

    # --- 3. Download gradient tensors ---
    gradient_dicts: list[dict[str, torch.Tensor]] = []
    for sub in submissions:
        gradient_path = sub.get("gradient_path", "")
        if not gradient_path:
            logger.warning(
                "event=missing_gradient_path | job_id=%s | round=%d | submission_id=%s",
                job_id,
                round_number,
                sub.get("id"),
            )
            continue
        try:
            data = _download_blob(GRADIENTS_BUCKET, gradient_path)
            grad_dict = _deserialize_state_dict(data)
            gradient_dicts.append(grad_dict)
        except Exception:
            logger.exception(
                "event=gradient_download_failed | job_id=%s | round=%d | path=%s",
                job_id,
                round_number,
                gradient_path,
            )

    if not gradient_dicts:
        logger.error(
            "event=aggregation_failed | job_id=%s | round=%d | reason=no_valid_gradients",
            job_id,
            round_number,
        )
        return

    # --- 4. Compute element-wise mean of gradients ---
    mean_grads: dict[str, torch.Tensor] = {}
    num_grads = len(gradient_dicts)
    keys = gradient_dicts[0].keys()

    for key in keys:
        stacked = torch.stack([g[key].float() for g in gradient_dicts])
        mean_grads[key] = stacked.mean(dim=0)

    # --- 5. Load current Global_Model parameters ---
    param_bytes = param_server.get_parameters(job_id)
    current_params = _deserialize_state_dict(param_bytes)

    # --- 6. Get learning rate from job config ---
    job_rows = db.select("jobs", filters={"id": job_id})
    if not job_rows:
        logger.error("event=job_not_found | job_id=%s", job_id)
        return
    job = job_rows[0]

    hyperparams = job.get("hyperparameters", {})
    learning_rate = float(hyperparams.get("learning_rate", 0.001))
    total_rounds = int(
        job.get("total_rounds") or hyperparams.get("epochs", 10)
    )

    # --- 7. Apply SGD step: new_params[key] = old_params[key] - lr * mean_grads[key] ---
    new_params: dict[str, torch.Tensor] = {}
    for key in current_params:
        if key in mean_grads:
            new_params[key] = current_params[key].float() - learning_rate * mean_grads[key]
        else:
            # Parameter not in gradients (e.g. batch norm running stats) — keep as-is
            new_params[key] = current_params[key]

    # --- 8. Upload updated parameters ---
    param_server.update_parameters(job_id, new_params)

    # --- 9. Store metrics ---
    _store_round_metrics(job_id, round_number, submissions, round_records)

    # --- 10. Mark round as completed ---
    now = datetime.now(timezone.utc).isoformat()
    if round_records:
        db.update(
            "training_rounds",
            {
                "status": TrainingRoundStatus.COMPLETED.value,
                "completed_at": now,
            },
            filters={"id": round_records[0]["id"]},
        )

    # --- 11. Advance job's current_round ---
    next_round = round_number + 1
    db.update(
        "jobs",
        {"current_round": next_round},
        filters={"id": job_id},
    )

    # --- 12. Create next round or complete the job ---
    if next_round >= total_rounds:
        complete_job(job_id)
    else:
        active_workers = barrier_mod.get_active_workers(job_id)
        if active_workers:
            barrier_mod.create_round(job_id, next_round, len(active_workers))
        else:
            # All workers gone — fail the job
            check_job_failure(job_id)

    # --- 13. Clean up gradient storage for this round ---
    _cleanup_round_gradients(job_id, round_number, submissions)

    logger.info(
        "event=round_aggregated | job_id=%s | round=%d | num_gradients=%d | lr=%s",
        job_id,
        round_number,
        num_grads,
        learning_rate,
    )


def complete_job(job_id: str) -> None:
    """Mark a job as completed and store the final checkpoint.

    Called when all configured training rounds have been aggregated.

    - Stores the final Global_Model checkpoint via ``param_server``.
    - Builds ``aggregated_metrics`` JSON from all round metrics.
    - Updates the job record with status "completed".
    """
    now = datetime.now(timezone.utc).isoformat()

    # Determine the last completed round number for the checkpoint
    job_rows = db.select("jobs", filters={"id": job_id})
    if not job_rows:
        logger.error("event=complete_job_not_found | job_id=%s", job_id)
        return
    job = job_rows[0]

    hyperparams = job.get("hyperparameters", {})
    total_rounds = int(
        job.get("total_rounds") or hyperparams.get("epochs", 10)
    )
    final_round = total_rounds - 1

    # Store final checkpoint
    checkpoint_path: str | None = None
    try:
        checkpoint_path = param_server.store_checkpoint(job_id, final_round)
    except Exception:
        logger.exception(
            "event=checkpoint_store_failed | job_id=%s | round=%d",
            job_id,
            final_round,
        )

    # Build aggregated_metrics from all completed training_rounds
    aggregated_metrics = _build_aggregated_metrics(job_id)

    # Update job record
    update_data: dict[str, Any] = {
        "status": JobStatus.COMPLETED.value,
        "aggregated_metrics": aggregated_metrics,
        "completed_at": now,
    }
    if checkpoint_path:
        update_data["global_model_path"] = checkpoint_path

    db.update("jobs", update_data, filters={"id": job_id})

    # Mark all remaining active tasks as completed
    tasks = db.select("tasks", filters={"job_id": job_id})
    for task in tasks:
        if task.get("status") in (
            TaskStatus.ASSIGNED.value,
            TaskStatus.RUNNING.value,
        ):
            db.update(
                "tasks",
                {"status": TaskStatus.COMPLETED.value, "completed_at": now},
                filters={"id": task["id"]},
            )

    logger.info(
        "event=job_completed | job_id=%s | total_rounds=%d | checkpoint=%s",
        job_id,
        total_rounds,
        checkpoint_path,
    )


def check_job_failure(job_id: str) -> None:
    """Check if a job should be marked as "failed".

    A job is marked "failed" when:
    - At least one task has status "failed", AND
    - No tasks remain with status "queued", "assigned", or "running"

    When marking the job as failed:
    - An ``error_summary`` is built containing per-task error messages.
    - If any training rounds were completed, a partial checkpoint is
      stored via ``param_server.store_checkpoint()``.
    """
    all_tasks = db.select("tasks", filters={"job_id": job_id})
    if not all_tasks:
        return

    active_statuses = {
        TaskStatus.QUEUED.value,
        TaskStatus.ASSIGNED.value,
        TaskStatus.RUNNING.value,
    }
    has_failed = False
    has_active = False
    error_details: list[dict[str, Any]] = []

    for task in all_tasks:
        task_status = task.get("status")
        if task_status == TaskStatus.FAILED.value:
            has_failed = True
            error_details.append({
                "task_id": task["id"],
                "shard_index": task.get("shard_index"),
                "node_id": task.get("node_id"),
                "error_message": task.get("error_message", "Unknown error"),
            })
        elif task_status in active_statuses:
            has_active = True

    if not has_failed or has_active:
        return

    now = datetime.now(timezone.utc).isoformat()

    # Attempt to store a partial checkpoint if any rounds completed
    checkpoint_path: str | None = None
    completed_rounds = db.select(
        "training_rounds",
        filters={"job_id": job_id, "status": TrainingRoundStatus.COMPLETED.value},
    )
    if completed_rounds:
        last_completed = max(
            completed_rounds, key=lambda r: r.get("round_number", 0)
        )
        try:
            checkpoint_path = param_server.store_checkpoint(
                job_id, last_completed["round_number"]
            )
        except Exception:
            logger.exception(
                "event=partial_checkpoint_failed | job_id=%s",
                job_id,
            )

    update_data: dict[str, Any] = {
        "status": JobStatus.FAILED.value,
        "error_summary": {"failed_tasks": error_details},
        "completed_at": now,
    }
    if checkpoint_path:
        update_data["global_model_path"] = checkpoint_path

    db.update("jobs", update_data, filters={"id": job_id})

    logger.info(
        "event=job_failed | job_id=%s | failed_task_count=%d | partial_checkpoint=%s",
        job_id,
        len(error_details),
        checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _store_round_metrics(
    job_id: str,
    round_number: int,
    submissions: list[dict[str, Any]],
    round_records: list[dict[str, Any]],
) -> None:
    """Store per-worker local metrics and per-round global metrics.

    - Inserts a ``worker_local`` metric row for each submission that
      reported loss/accuracy.
    - Computes the mean loss and accuracy across all workers.
    - Inserts a ``global_aggregated`` metric row.
    - Updates the ``training_rounds`` record with global metrics.
    """
    now = datetime.now(timezone.utc).isoformat()
    losses: list[float] = []
    accuracies: list[float] = []

    for sub in submissions:
        local_loss = sub.get("local_loss")
        local_accuracy = sub.get("local_accuracy")

        # Convert from Decimal/str if needed
        if local_loss is not None:
            local_loss = float(local_loss)
            losses.append(local_loss)
        if local_accuracy is not None:
            local_accuracy = float(local_accuracy)
            accuracies.append(local_accuracy)

        # Store per-worker metric
        try:
            db.insert(
                "metrics",
                {
                    "job_id": job_id,
                    "task_id": sub.get("task_id"),
                    "node_id": sub.get("node_id"),
                    "round_number": round_number,
                    "metric_type": "worker_local",
                    "loss": local_loss,
                    "accuracy": local_accuracy,
                    "created_at": now,
                },
            )
        except Exception:
            logger.exception(
                "event=worker_metric_insert_failed | job_id=%s | round=%d | task_id=%s",
                job_id,
                round_number,
                sub.get("task_id"),
            )

    # Compute global metrics
    global_loss = sum(losses) / len(losses) if losses else None
    global_accuracy = sum(accuracies) / len(accuracies) if accuracies else None

    # Store global aggregated metric
    try:
        db.insert(
            "metrics",
            {
                "job_id": job_id,
                "task_id": None,
                "node_id": None,
                "round_number": round_number,
                "metric_type": "global_aggregated",
                "loss": global_loss,
                "accuracy": global_accuracy,
                "created_at": now,
            },
        )
    except Exception:
        logger.exception(
            "event=global_metric_insert_failed | job_id=%s | round=%d",
            job_id,
            round_number,
        )

    # Update training_rounds record with global metrics
    if round_records:
        try:
            db.update(
                "training_rounds",
                {
                    "global_loss": global_loss,
                    "global_accuracy": global_accuracy,
                },
                filters={"id": round_records[0]["id"]},
            )
        except Exception:
            logger.exception(
                "event=round_metrics_update_failed | job_id=%s | round=%d",
                job_id,
                round_number,
            )


def _build_aggregated_metrics(job_id: str) -> dict[str, Any]:
    """Build the ``aggregated_metrics`` JSON for a completed job.

    Collects per-round global metrics from the ``training_rounds`` table
    and per-worker breakdowns from the ``metrics`` table.

    Returns a dict with:
    - ``per_round``: list of {round_number, global_loss, global_accuracy}
    - ``mean_loss``: mean of all rounds' global losses
    - ``mean_accuracy``: mean of all rounds' global accuracies
    - ``per_worker``: list of per-worker metric summaries
    """
    # Per-round global metrics
    round_records = db.select("training_rounds", filters={"job_id": job_id})
    completed_rounds = [
        r for r in round_records
        if r.get("status") == TrainingRoundStatus.COMPLETED.value
    ]
    completed_rounds.sort(key=lambda r: r.get("round_number", 0))

    per_round: list[dict[str, Any]] = []
    all_losses: list[float] = []
    all_accuracies: list[float] = []

    for rnd in completed_rounds:
        gl = rnd.get("global_loss")
        ga = rnd.get("global_accuracy")
        if gl is not None:
            gl = float(gl)
            all_losses.append(gl)
        if ga is not None:
            ga = float(ga)
            all_accuracies.append(ga)
        per_round.append({
            "round_number": rnd.get("round_number"),
            "global_loss": gl,
            "global_accuracy": ga,
        })

    # Per-worker breakdown from worker_local metrics
    worker_metrics = db.select("metrics", filters={"job_id": job_id})
    worker_local = [
        m for m in worker_metrics if m.get("metric_type") == "worker_local"
    ]

    # Group by task_id for per-worker summary
    per_worker_map: dict[str, list[dict[str, Any]]] = {}
    for m in worker_local:
        tid = m.get("task_id", "unknown")
        per_worker_map.setdefault(tid, []).append(m)

    per_worker: list[dict[str, Any]] = []
    for tid, metrics_list in per_worker_map.items():
        # Use the last round's metrics as the "final" for this worker
        metrics_list.sort(key=lambda x: x.get("round_number", 0))
        final = metrics_list[-1] if metrics_list else {}
        per_worker.append({
            "task_id": tid,
            "node_id": final.get("node_id"),
            "final_loss": float(final["loss"]) if final.get("loss") is not None else None,
            "final_accuracy": float(final["accuracy"]) if final.get("accuracy") is not None else None,
            "rounds_participated": len(metrics_list),
        })

    return {
        "per_round": per_round,
        "mean_loss": sum(all_losses) / len(all_losses) if all_losses else None,
        "mean_accuracy": sum(all_accuracies) / len(all_accuracies) if all_accuracies else None,
        "per_worker": per_worker,
    }


def _cleanup_round_gradients(
    job_id: str,
    round_number: int,
    submissions: list[dict[str, Any]],
) -> None:
    """Delete gradient blobs from storage after a round is aggregated.

    Removes each individual gradient file that was submitted for this
    round, then attempts to remove the round directory.
    """
    for sub in submissions:
        gradient_path = sub.get("gradient_path", "")
        if gradient_path:
            try:
                _delete_blob(GRADIENTS_BUCKET, gradient_path)
            except Exception:
                logger.warning(
                    "event=gradient_cleanup_failed | job_id=%s | round=%d | path=%s",
                    job_id,
                    round_number,
                    gradient_path,
                    exc_info=True,
                )

    logger.debug(
        "event=gradient_cleanup_done | job_id=%s | round=%d | count=%d",
        job_id,
        round_number,
        len(submissions),
    )
