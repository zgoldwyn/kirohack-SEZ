"""Task scheduling: creation from job config and pull-based assignment.

This module handles:
- Creating task records when a job is submitted (task creation).
- Task polling and assignment to idle workers (pull-based).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from coordinator import db
from coordinator.config_parser import (
    generate_task_configs,
    get_resource_requirements,
    parse_job_config,
)
from coordinator.constants import JobStatus, NodeStatus, TaskStatus
from coordinator.models import JobConfig, TaskPollResponse

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
        "event=tasks_created | job_id=%s | task_count=%d",
        job_id,
        job_config.shard_count,
    )
    return created_tasks


def poll_task(node: dict) -> TaskPollResponse:
    """Find and assign an eligible queued task to the polling node.

    Pull-based assignment logic:
    1. Verify the node is idle (not already busy).
    2. Fetch all queued tasks.
    3. For each queued task, check resource eligibility:
       - Node's ``ram_mb`` >= task's minimum RAM requirement
       - If the task requires GPU, node must have a non-null ``gpu_model``
    4. Assign the first eligible task: update task status to "assigned",
       set ``node_id`` and ``assigned_at``.
    5. Update node status to "busy".
    6. If this is the first task assigned for the parent job, update job
       status to "running" and set ``started_at``.
    7. Return a ``TaskPollResponse`` with the task config, or an empty
       response if no eligible task is found.

    Args:
        node: The authenticated node record dict (from ``get_current_node``).

    Returns:
        A ``TaskPollResponse`` — populated if a task was assigned, empty
        (all fields ``None``) if no eligible task is available.
    """
    # 1. Check if the node is idle
    if node.get("status") != NodeStatus.IDLE.value:
        return TaskPollResponse()

    node_ram = node.get("ram_mb", 0)
    node_gpu = node.get("gpu_model")

    # 2. Fetch all queued tasks
    queued_tasks = db.select("tasks", filters={"status": TaskStatus.QUEUED.value})
    if not queued_tasks:
        return TaskPollResponse()

    # 3. Find the first eligible task based on resource requirements
    eligible_task: dict | None = None
    for task in queued_tasks:
        task_config = task.get("task_config", {})
        model_type = task_config.get("model_type")

        if not model_type:
            # If task_config doesn't have model_type, look it up from the job
            job_rows = db.select("jobs", columns="model_type", filters={"id": task["job_id"]})
            if job_rows:
                model_type = job_rows[0].get("model_type")

        if not model_type:
            # Can't determine resource requirements — skip this task
            continue

        try:
            reqs = get_resource_requirements(model_type)
        except ValueError:
            # Unknown model type — skip
            continue

        # Check RAM
        if node_ram < reqs.min_ram_mb:
            continue

        # Check GPU
        if reqs.gpu_required and not node_gpu:
            continue

        eligible_task = task
        break

    if eligible_task is None:
        return TaskPollResponse()

    # 4. Assign the task to this node
    now = datetime.now(timezone.utc).isoformat()
    task_id = eligible_task["id"]
    job_id = eligible_task["job_id"]

    db.update(
        "tasks",
        {
            "status": TaskStatus.ASSIGNED.value,
            "node_id": node["id"],
            "assigned_at": now,
        },
        filters={"id": task_id},
    )

    # 5. Update node status to busy
    db.update(
        "nodes",
        {"status": NodeStatus.BUSY.value},
        filters={"id": node["id"]},
    )

    # 6. Check if this is the first task assigned for the job
    #    (job should transition from "queued" to "running")
    job_rows = db.select("jobs", filters={"id": job_id})
    if job_rows and job_rows[0].get("status") == JobStatus.QUEUED.value:
        db.update(
            "jobs",
            {"status": JobStatus.RUNNING.value, "started_at": now},
            filters={"id": job_id},
        )

    # 7. Build and return the response from the task config
    task_config = eligible_task.get("task_config", {})

    logger.info(
        "event=task_assigned | task_id=%s | job_id=%s | shard_index=%d | node_id=%s",
        task_id,
        job_id,
        eligible_task.get("shard_index", -1),
        node.get("node_id", node["id"]),
    )

    return TaskPollResponse(
        task_id=task_id,
        job_id=job_id,
        dataset_name=task_config.get("dataset_name"),
        model_type=task_config.get("model_type"),
        hyperparameters=task_config.get("hyperparameters"),
        shard_index=task_config.get("shard_index"),
        shard_count=task_config.get("shard_count"),
        total_rounds=task_config.get("total_rounds"),
    )
