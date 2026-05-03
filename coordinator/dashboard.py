"""Dashboard-facing read endpoints (unauthenticated, local/demo only).

These endpoints are intended for local development and demo use only.
They expose node status, job progress, training round convergence data,
per-worker contribution status, and artifact metadata without requiring
authentication.

Endpoints:
    GET /api/nodes                  — List all nodes
    GET /api/jobs                   — List all jobs (with round progress)
    GET /api/jobs/{id}              — Job detail with tasks, round progress,
                                      per-worker contribution status, and
                                      per-round global metrics
    GET /api/jobs/{id}/results      — Per-round metrics history + final
                                      Global_Model checkpoint path
    GET /api/jobs/{id}/artifacts    — Artifacts for a job
    GET /api/monitoring/summary     — System-wide counts with per-job round
                                      progress for running jobs
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from coordinator.db import DatabaseError, RecordNotFoundError, delete, select, select_one

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _db_error_to_http(exc: DatabaseError) -> HTTPException:
    """Convert a DatabaseError into an appropriate HTTPException."""
    return HTTPException(status_code=503, detail={"error": "Service Unavailable", "detail": str(exc)})


# ---------------------------------------------------------------------------
# Node endpoints
# ---------------------------------------------------------------------------


@router.get("/api/nodes", summary="List all registered nodes")
async def list_nodes() -> list[dict[str, Any]]:
    """Return all registered nodes with status, hardware info, and last heartbeat.

    Requirements: 2.4, 9.1
    """
    try:
        nodes = select(
            "nodes",
            columns=(
                "id, node_id, hostname, cpu_cores, gpu_model, vram_mb, "
                "ram_mb, disk_mb, os, python_version, pytorch_version, "
                "status, last_heartbeat, created_at"
            ),
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    return nodes


# ---------------------------------------------------------------------------
# Node detail endpoint
# ---------------------------------------------------------------------------


@router.get("/api/nodes/{node_id}", summary="Get node detail with task history")
async def get_node(node_id: str) -> dict[str, Any]:
    """Return node detail including hardware info and tasks assigned to this node."""
    try:
        node = select_one(
            "nodes",
            columns=(
                "id, node_id, hostname, cpu_cores, gpu_model, vram_mb, "
                "ram_mb, disk_mb, os, python_version, pytorch_version, "
                "status, last_heartbeat, created_at"
            ),
            filters={"id": node_id},
        )
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Fetch tasks assigned to this node
    try:
        tasks = select(
            "tasks",
            columns=(
                "id, job_id, shard_index, status, checkpoint_path, "
                "error_message, assigned_at, started_at, completed_at"
            ),
            filters={"node_id": node_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Enrich tasks with job name
    job_ids = list({t["job_id"] for t in tasks if t.get("job_id")})
    job_names: dict[str, str | None] = {}
    for jid in job_ids:
        try:
            job = select_one("jobs", columns="id, job_name", filters={"id": jid})
            job_names[jid] = job.get("job_name")
        except (RecordNotFoundError, DatabaseError):
            job_names[jid] = None

    enriched_tasks = [
        {**t, "job_name": job_names.get(t["job_id"])} for t in tasks
    ]

    return {**node, "tasks": enriched_tasks}


# ---------------------------------------------------------------------------
# Job list endpoint
# ---------------------------------------------------------------------------


@router.get("/api/jobs", summary="List all training jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """Return all jobs with status, model type, dataset, shard count,
    round progress, and timestamps.

    Requirements: 10.1
    """
    try:
        jobs = select(
            "jobs",
            columns=(
                "id, job_name, dataset_name, model_type, hyperparameters, "
                "shard_count, status, current_round, total_rounds, "
                "aggregated_metrics, error_summary, "
                "created_at, started_at, completed_at"
            ),
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    return jobs


# ---------------------------------------------------------------------------
# Job detail endpoint
# ---------------------------------------------------------------------------


@router.get("/api/jobs/{job_id}", summary="Get job detail with tasks and metrics")
async def get_job(job_id: str) -> dict[str, Any]:
    """Return job detail including training round progress, per-worker
    contribution status, per-round global metrics, and task information.

    Requirements: 6.3, 10.1, 10.2, 10.3, 10.4
    """
    # Fetch the job record
    try:
        job = select_one("jobs", filters={"id": job_id})
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Fetch tasks for this job
    try:
        tasks = select(
            "tasks",
            columns=(
                "id, job_id, node_id, shard_index, status, task_config, "
                "last_submitted_round, error_message, assigned_at, started_at, "
                "completed_at, created_at"
            ),
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # For each task, attach the latest round metrics (most recent round row)
    latest_metrics_by_task: dict[str, dict[str, Any]] = {}
    try:
        all_worker_metrics = select(
            "metrics",
            columns="task_id, round_number, loss, accuracy",
            filters={"job_id": job_id, "metric_type": "worker_local"},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    for row in all_worker_metrics:
        tid = row.get("task_id")
        if tid is None:
            continue
        existing = latest_metrics_by_task.get(tid)
        if existing is None or row["round_number"] > existing["round_number"]:
            latest_metrics_by_task[tid] = row

    # Enrich task records with latest metrics
    enriched_tasks = []
    for task in tasks:
        latest = latest_metrics_by_task.get(task["id"])
        enriched_tasks.append(
            {
                **task,
                "latest_round": latest["round_number"] if latest else None,
                "latest_loss": latest["loss"] if latest else None,
                "latest_accuracy": latest["accuracy"] if latest else None,
            }
        )

    # -----------------------------------------------------------------------
    # Per-worker contribution status from gradient_submissions
    # -----------------------------------------------------------------------
    current_round = job.get("current_round") or 0
    worker_contributions: list[dict[str, Any]] = []

    try:
        gradient_subs = select(
            "gradient_submissions",
            columns="task_id, node_id, round_number, local_loss, local_accuracy, created_at",
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Build a lookup: task_id → latest gradient submission
    latest_sub_by_task: dict[str, dict[str, Any]] = {}
    for sub in gradient_subs:
        tid = sub.get("task_id")
        if tid is None:
            continue
        existing = latest_sub_by_task.get(tid)
        if existing is None or sub["round_number"] > existing["round_number"]:
            latest_sub_by_task[tid] = sub

    for task in tasks:
        task_id = task["id"]
        node_id = task.get("node_id")
        task_status = task.get("status", "")
        last_sub = latest_sub_by_task.get(task_id)
        last_submitted_round = last_sub["round_number"] if last_sub else None

        # Derive worker contribution status:
        # - "failed" if task is failed
        # - "submitted" if worker has submitted for the current round
        # - "computing" if task is running but hasn't submitted for current round
        # - "waiting" if task is assigned but not yet running
        # - "completed" if task is completed
        if task_status == "failed":
            contribution_status = "failed"
        elif task_status == "completed":
            contribution_status = "completed"
        elif last_submitted_round is not None and last_submitted_round >= current_round:
            contribution_status = "submitted"
        elif task_status == "running":
            contribution_status = "computing"
        else:
            contribution_status = "waiting"

        worker_contributions.append({
            "task_id": task_id,
            "node_id": node_id,
            "shard_index": task.get("shard_index"),
            "status": contribution_status,
            "last_submitted_round": last_submitted_round,
        })

    # -----------------------------------------------------------------------
    # Per-round global metrics from training_rounds for convergence chart
    # -----------------------------------------------------------------------
    try:
        training_rounds = select(
            "training_rounds",
            columns=(
                "round_number, status, active_worker_count, submitted_count, "
                "global_loss, global_accuracy, started_at, completed_at"
            ),
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Sort by round_number for consistent ordering
    training_rounds.sort(key=lambda r: r.get("round_number", 0))

    return {
        **job,
        "tasks": enriched_tasks,
        "worker_contributions": worker_contributions,
        "training_rounds": training_rounds,
    }


# ---------------------------------------------------------------------------
# Job results endpoint
# ---------------------------------------------------------------------------


@router.get("/api/jobs/{job_id}/results", summary="Get per-round metrics and final checkpoint")
async def get_job_results(job_id: str) -> dict[str, Any]:
    """Return per-round metrics history and the final Global_Model checkpoint
    path for a completed job.

    Requirements: 6.3, 7.3
    """
    try:
        job = select_one(
            "jobs",
            columns="id, status, current_round, total_rounds, global_model_path, aggregated_metrics, error_summary",
            filters={"id": job_id},
        )
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Fetch per-round global metrics from training_rounds
    try:
        training_rounds = select(
            "training_rounds",
            columns=(
                "round_number, status, active_worker_count, submitted_count, "
                "global_loss, global_accuracy, started_at, completed_at"
            ),
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    training_rounds.sort(key=lambda r: r.get("round_number", 0))

    # Fetch the final Global_Model checkpoint artifact (task_id is NULL for
    # global checkpoints)
    try:
        artifacts = select(
            "artifacts",
            columns="id, artifact_type, storage_path, round_number, size_bytes, created_at",
            filters={"job_id": job_id, "artifact_type": "checkpoint"},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Find the global checkpoint (task_id is NULL) — pick the one with the
    # highest round_number as the final checkpoint.
    global_checkpoints = [a for a in artifacts if a.get("task_id") is None]
    global_checkpoints.sort(
        key=lambda a: a.get("round_number") if a.get("round_number") is not None else -1,
        reverse=True,
    )
    final_checkpoint = global_checkpoints[0] if global_checkpoints else None

    return {
        "job_id": job["id"],
        "status": job["status"],
        "current_round": job.get("current_round"),
        "total_rounds": job.get("total_rounds"),
        "global_model_path": job.get("global_model_path"),
        "aggregated_metrics": job.get("aggregated_metrics"),
        "error_summary": job.get("error_summary"),
        "training_rounds": training_rounds,
        "final_checkpoint": final_checkpoint,
    }


# ---------------------------------------------------------------------------
# Job artifacts endpoint
# ---------------------------------------------------------------------------


@router.get("/api/jobs/{job_id}/artifacts", summary="List artifacts for a job")
async def list_job_artifacts(job_id: str) -> list[dict[str, Any]]:
    """Return all artifact records (checkpoints, logs, outputs) for a job.

    Requirements: 7.3
    """
    # Verify the job exists first so we return 404 rather than an empty list
    # for an unknown job ID.
    try:
        select_one("jobs", columns="id", filters={"id": job_id})
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    try:
        artifacts = select(
            "artifacts",
            columns=(
                "id, job_id, task_id, node_id, artifact_type, "
                "storage_path, round_number, size_bytes, created_at"
            ),
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    return artifacts


# ---------------------------------------------------------------------------
# Monitoring summary endpoint
# ---------------------------------------------------------------------------


@router.get("/api/monitoring/summary", summary="System-wide monitoring summary")
async def monitoring_summary() -> dict[str, Any]:
    """Return counts of nodes and jobs broken down by status, plus
    current_round for each running job.

    Requirements: 11.2
    """
    try:
        nodes = select("nodes", columns="status")
        jobs = select("jobs", columns="id, status, current_round, total_rounds")
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Count nodes by status
    node_counts: dict[str, int] = {"idle": 0, "busy": 0, "offline": 0}
    for node in nodes:
        s = node.get("status", "")
        if s in node_counts:
            node_counts[s] += 1

    # Derive "online" as idle + busy
    online_nodes = node_counts["idle"] + node_counts["busy"]

    # Count jobs by status and collect running job round progress
    job_counts: dict[str, int] = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
    running_jobs: list[dict[str, Any]] = []
    for job in jobs:
        s = job.get("status", "")
        if s in job_counts:
            job_counts[s] += 1
        if s == "running":
            running_jobs.append({
                "job_id": job["id"],
                "current_round": job.get("current_round"),
                "total_rounds": job.get("total_rounds"),
            })

    return {
        "nodes": {
            "online": online_nodes,
            "idle": node_counts["idle"],
            "busy": node_counts["busy"],
            "offline": node_counts["offline"],
            "total": len(nodes),
        },
        "jobs": {
            "queued": job_counts["queued"],
            "running": job_counts["running"],
            "completed": job_counts["completed"],
            "failed": job_counts["failed"],
            "total": len(jobs),
        },
        "running_jobs": running_jobs,
    }


# ---------------------------------------------------------------------------
# Delete job endpoint
# ---------------------------------------------------------------------------


@router.delete("/api/jobs/{job_id}", summary="Delete a job and its related data")
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job and cascade-delete its tasks, metrics, and artifacts.

    Only completed or failed jobs can be deleted. Returns 409 if the job
    is still queued or running.
    """
    try:
        job = select_one("jobs", columns="id, status", filters={"id": job_id})
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    if job["status"] in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail={"error": "Cannot delete a job that is queued or running"},
        )

    # Cascade delete: metrics → artifacts → tasks → job
    try:
        delete("metrics", filters={"job_id": job_id})
        delete("artifacts", filters={"job_id": job_id})
        delete("tasks", filters={"job_id": job_id})
        delete("jobs", filters={"id": job_id})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    return {"status": "deleted", "job_id": job_id}


# ---------------------------------------------------------------------------
# Delete offline node endpoint
# ---------------------------------------------------------------------------


@router.delete("/api/nodes/{node_id}", summary="Delete an offline worker node")
async def delete_node(node_id: str) -> dict[str, str]:
    """Delete an offline worker node.

    Only nodes with status 'offline' can be deleted. Returns 409 if the
    node is idle or busy.
    """
    try:
        node = select_one("nodes", columns="id, status", filters={"id": node_id})
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    if node["status"] != "offline":
        raise HTTPException(
            status_code=409,
            detail={"error": "Can only delete nodes that are offline"},
        )

    try:
        delete("nodes", filters={"id": node_id})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    return {"status": "deleted", "node_id": node_id}
