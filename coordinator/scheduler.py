"""Task scheduling: creation from job config and pull-based assignment.

This module handles:
- Creating task records when a job is submitted (task creation).
- (Future) Task polling and assignment to idle workers.
"""

from __future__ import annotations

import logging
from typing import Any

from coordinator import db
from coordinator.config_parser import generate_task_configs, parse_job_config
from coordinator.constants import JobStatus, TaskStatus
from coordinator.models import JobConfig

logger = logging.getLogger(__name__)


def create_tasks_for_job(job_id: str, job_config: JobConfig) -> list[dict[str, Any]]:
    """Create task records in the database for a newly submitted job.

    Inserts ``job_config.shard_count`` task rows, each with status "queued",
    a unique shard_index in {0, ..., N-1}, and a serialised task_config
    payload.

    Args:
        job_id: Database UUID of the parent job.
        job_config: Validated job configuration.

    Returns:
        List of inserted task records.
    """
    # We need to insert tasks first to get their IDs, then generate configs.
    # Strategy: insert bare tasks, collect IDs, generate configs, then update.
    created_tasks: list[dict[str, Any]] = []

    for shard_index in range(job_config.shard_count):
        task_record = db.insert(
            "tasks",
            {
                "job_id": job_id,
                "shard_index": shard_index,
                "status": TaskStatus.QUEUED.value,
                "task_config": {},  # placeholder — will be updated below
            },
        )
        created_tasks.append(task_record)

    # Now generate full task configs with the real task IDs
    task_ids = [t["id"] for t in created_tasks]
    task_configs = generate_task_configs(job_config, job_id, task_ids)

    # Update each task with its full config payload
    for task_record, task_cfg in zip(created_tasks, task_configs):
        db.update(
            "tasks",
            {"task_config": task_cfg.model_dump()},
            filters={"id": task_record["id"]},
        )

    logger.info(
        "Created %d tasks for job %s",
        job_config.shard_count,
        job_id,
    )
    return created_tasks
