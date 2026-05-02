"""Metrics aggregation and job completion/failure logic.

When all tasks in a job are completed, computes aggregated metrics
(mean loss, mean accuracy, per-node breakdown) using per-epoch metrics
from the metrics table as the source of truth.

Also provides a helper to check whether a job should be marked as
"failed" when some tasks have failed and none remain active.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from coordinator import db
from coordinator.constants import JobStatus, TaskStatus

logger = logging.getLogger(__name__)


def aggregate_job_metrics(job_id: str) -> None:
    """Compute aggregated metrics for a completed job and update the job record.

    For each completed task, the "final" metric is the metrics row with
    the largest ``epoch`` number for that task.  We then compute:

    - ``mean_loss``: arithmetic mean of all tasks' final epoch losses
    - ``mean_accuracy``: arithmetic mean of all tasks' final epoch accuracies
    - ``per_node``: list of per-task breakdowns (node_id, loss, accuracy)

    The job record is updated with status "completed", the aggregated
    metrics JSON, and ``completed_at``.
    """
    # Fetch all completed tasks for this job
    tasks = db.select("tasks", filters={"job_id": job_id, "status": TaskStatus.COMPLETED.value})
    if not tasks:
        logger.warning("event=aggregation_skipped | job_id=%s | reason=no_completed_tasks", job_id)
        return

    per_node_breakdown: list[dict[str, Any]] = []
    losses: list[float] = []
    accuracies: list[float] = []

    for task in tasks:
        task_id = task["id"]
        node_id = task.get("node_id")

        # Get all metrics for this task, ordered by epoch descending
        task_metrics = db.select("metrics", filters={"task_id": task_id})

        if not task_metrics:
            # No metrics reported — include with None values
            per_node_breakdown.append({
                "node_id": node_id,
                "task_id": task_id,
                "loss": None,
                "accuracy": None,
            })
            continue

        # Find the metric row with the largest epoch
        final_metric = max(task_metrics, key=lambda m: m.get("epoch", 0))

        loss = final_metric.get("loss")
        accuracy = final_metric.get("accuracy")

        # Convert to float if not None (Supabase may return Decimal/str)
        if loss is not None:
            loss = float(loss)
            losses.append(loss)
        if accuracy is not None:
            accuracy = float(accuracy)
            accuracies.append(accuracy)

        per_node_breakdown.append({
            "node_id": node_id,
            "task_id": task_id,
            "loss": loss,
            "accuracy": accuracy,
        })

    aggregated: dict[str, Any] = {
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "per_node": per_node_breakdown,
    }

    now = datetime.now(timezone.utc).isoformat()
    db.update(
        "jobs",
        {
            "status": JobStatus.COMPLETED.value,
            "aggregated_metrics": aggregated,
            "completed_at": now,
        },
        filters={"id": job_id},
    )
    logger.info(
        "event=job_completed | job_id=%s | mean_loss=%s | mean_accuracy=%s | task_count=%d",
        job_id,
        aggregated.get("mean_loss"),
        aggregated.get("mean_accuracy"),
        len(tasks),
    )


def check_job_failure(job_id: str) -> None:
    """Check if a job should be marked as "failed".

    A job is marked "failed" when:
    - At least one task has status "failed", AND
    - No tasks remain with status "queued", "assigned", or "running"

    When marking the job as failed, an ``error_summary`` is built
    containing per-task error messages.
    """
    all_tasks = db.select("tasks", filters={"job_id": job_id})
    if not all_tasks:
        return

    active_statuses = {TaskStatus.QUEUED.value, TaskStatus.ASSIGNED.value, TaskStatus.RUNNING.value}
    has_failed = False
    has_active = False
    error_details: list[dict[str, Any]] = []

    for task in all_tasks:
        status = task.get("status")
        if status == TaskStatus.FAILED.value:
            has_failed = True
            error_details.append({
                "task_id": task["id"],
                "shard_index": task.get("shard_index"),
                "error_message": task.get("error_message", "Unknown error"),
            })
        elif status in active_statuses:
            has_active = True

    if has_failed and not has_active:
        now = datetime.now(timezone.utc).isoformat()
        db.update(
            "jobs",
            {
                "status": JobStatus.FAILED.value,
                "error_summary": {"failed_tasks": error_details},
                "completed_at": now,
            },
            filters={"id": job_id},
        )
        logger.info(
            "event=job_failed | job_id=%s | failed_task_count=%d",
            job_id,
            len(error_details),
        )
