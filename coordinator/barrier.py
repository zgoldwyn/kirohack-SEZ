"""Synchronization Barrier module — manages per-round barrier logic.

Enforces the synchronization barrier for collaborative distributed
training.  Each training round, the Coordinator waits for **all** active
Workers to submit their gradient updates before proceeding to
aggregation.  This module tracks submissions, checks barrier completion,
and handles worker removal (adjusting the expected count).

Key concepts
------------
- **Active workers** for a job are tasks whose status is ``assigned`` or
  ``running`` (i.e. not ``queued``, ``completed``, or ``failed``).
- A **training round** record in the ``training_rounds`` table tracks
  how many workers are expected (``active_worker_count``) and how many
  have submitted (``submitted_count``).
- The barrier is **met** when ``submitted_count >= active_worker_count``.

Requirements: 5.2, 5.5, 6.5, 14.2
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from coordinator import db
from coordinator.constants import TaskStatus, TrainingRoundStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_active_workers(job_id: str) -> set[str]:
    """Return the set of task IDs that are still active for *job_id*.

    A task is considered active if its status is ``assigned`` or
    ``running``.

    Parameters
    ----------
    job_id:
        The database UUID of the job.

    Returns
    -------
    set[str]
        Set of task IDs for active workers.
    """
    all_tasks = db.select("tasks", filters={"job_id": job_id})

    active_statuses = {TaskStatus.ASSIGNED.value, TaskStatus.RUNNING.value}
    active_task_ids: set[str] = set()

    for task in all_tasks:
        if task.get("status") in active_statuses:
            active_task_ids.add(task["id"])

    logger.debug(
        "event=get_active_workers | job_id=%s | active_count=%d",
        job_id,
        len(active_task_ids),
    )
    return active_task_ids


def create_round(
    job_id: str,
    round_number: int,
    active_worker_count: int,
) -> dict[str, Any]:
    """Create a new ``training_rounds`` record for the given round.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    round_number:
        The 0-indexed training round number.
    active_worker_count:
        The number of active workers expected to submit gradients.

    Returns
    -------
    dict
        The inserted ``training_rounds`` record.
    """
    now = datetime.now(timezone.utc).isoformat()

    record = db.insert(
        "training_rounds",
        {
            "job_id": job_id,
            "round_number": round_number,
            "status": TrainingRoundStatus.IN_PROGRESS.value,
            "active_worker_count": active_worker_count,
            "submitted_count": 0,
            "started_at": now,
            "created_at": now,
        },
    )

    logger.info(
        "event=round_created | job_id=%s | round_number=%d | active_worker_count=%d",
        job_id,
        round_number,
        active_worker_count,
    )
    return record


def record_submission(
    job_id: str,
    round_number: int,
    task_id: str,
    node_id: str | None,
) -> dict[str, Any]:
    """Record a gradient submission and increment the round's counter.

    Inserts a row into ``gradient_submissions`` and increments
    ``submitted_count`` on the matching ``training_rounds`` record.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    round_number:
        The training round this submission is for.
    task_id:
        The database UUID of the submitting task.
    node_id:
        The database UUID of the submitting node (may be ``None``).

    Returns
    -------
    dict
        The inserted ``gradient_submissions`` record.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Insert the gradient submission record
    submission = db.insert(
        "gradient_submissions",
        {
            "job_id": job_id,
            "task_id": task_id,
            "node_id": node_id,
            "round_number": round_number,
            "gradient_path": "",  # Caller sets the actual path separately
            "local_loss": None,
            "local_accuracy": None,
            "created_at": now,
        },
    )

    # Increment submitted_count on the training_rounds record.
    # PostgREST doesn't support atomic increment, so we read-then-write.
    round_records = db.select(
        "training_rounds",
        filters={"job_id": job_id, "round_number": round_number},
    )
    if round_records:
        current_count = round_records[0].get("submitted_count", 0)
        db.update(
            "training_rounds",
            {"submitted_count": current_count + 1},
            filters={"id": round_records[0]["id"]},
        )

    logger.info(
        "event=gradient_submitted | job_id=%s | round_number=%d | task_id=%s | node_id=%s",
        job_id,
        round_number,
        task_id,
        node_id,
    )
    return submission


def check_barrier(job_id: str, round_number: int) -> bool:
    """Check whether the synchronization barrier is met for a round.

    The barrier is met when ``submitted_count >= active_worker_count``
    in the ``training_rounds`` record for the given job and round.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    round_number:
        The training round to check.

    Returns
    -------
    bool
        ``True`` if the barrier is met (all active workers have
        submitted), ``False`` otherwise.
    """
    round_records = db.select(
        "training_rounds",
        filters={"job_id": job_id, "round_number": round_number},
    )

    if not round_records:
        logger.warning(
            "event=barrier_check_no_round | job_id=%s | round_number=%d",
            job_id,
            round_number,
        )
        return False

    record = round_records[0]
    submitted = record.get("submitted_count", 0)
    expected = record.get("active_worker_count", 0)

    is_met = submitted >= expected

    logger.debug(
        "event=barrier_check | job_id=%s | round_number=%d | submitted=%d | expected=%d | met=%s",
        job_id,
        round_number,
        submitted,
        expected,
        is_met,
    )
    return is_met


def remove_worker(job_id: str, task_id: str) -> None:
    """Remove a worker from the active set for a job.

    Marks the task as ``failed`` (if not already) and decrements
    ``active_worker_count`` on the **current** (latest ``in_progress``)
    ``training_rounds`` record for the job.

    Parameters
    ----------
    job_id:
        The database UUID of the job.
    task_id:
        The database UUID of the task to remove.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Mark the task as failed if it isn't already
    task_rows = db.select("tasks", filters={"id": task_id})
    if task_rows:
        task = task_rows[0]
        if task.get("status") not in (
            TaskStatus.FAILED.value,
            TaskStatus.COMPLETED.value,
        ):
            db.update(
                "tasks",
                {
                    "status": TaskStatus.FAILED.value,
                    "error_message": "worker removed from active set",
                    "completed_at": now,
                },
                filters={"id": task_id},
            )

    # Find the current in-progress round for this job and decrement
    # active_worker_count.  If there is no in_progress round, look for
    # the latest round by round_number.
    round_records = db.select(
        "training_rounds",
        filters={"job_id": job_id},
    )

    if not round_records:
        logger.warning(
            "event=remove_worker_no_rounds | job_id=%s | task_id=%s",
            job_id,
            task_id,
        )
        return

    # Prefer the in_progress round; fall back to the one with the
    # highest round_number.
    in_progress = [
        r
        for r in round_records
        if r.get("status") == TrainingRoundStatus.IN_PROGRESS.value
    ]
    target_round = (
        in_progress[0]
        if in_progress
        else max(round_records, key=lambda r: r.get("round_number", 0))
    )

    current_active = target_round.get("active_worker_count", 0)
    new_active = max(current_active - 1, 0)

    db.update(
        "training_rounds",
        {"active_worker_count": new_active},
        filters={"id": target_round["id"]},
    )

    logger.info(
        "event=worker_removed | job_id=%s | task_id=%s | round_number=%d | "
        "active_worker_count=%d->%d",
        job_id,
        task_id,
        target_round.get("round_number", -1),
        current_active,
        new_active,
    )
