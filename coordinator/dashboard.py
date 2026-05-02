"""Dashboard-facing read endpoints (unauthenticated, local/demo only).

These endpoints are intended for local development and demo use only.
They expose node status, job progress, aggregated metrics, and artifact
metadata without requiring authentication.

Endpoints:
    GET /api/nodes                  — List all nodes
    GET /api/jobs                   — List all jobs
    GET /api/jobs/{id}              — Job detail with tasks and metrics
    GET /api/jobs/{id}/results      — Aggregated metrics + checkpoint paths
    GET /api/jobs/{id}/artifacts    — Artifacts for a job
    GET /api/monitoring/summary     — System-wide counts
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from coordinator.db import DatabaseError, RecordNotFoundError, select, select_one

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
# Job list endpoint
# ---------------------------------------------------------------------------


@router.get("/api/jobs", summary="List all training jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """Return all jobs with status, model type, dataset, shard count, and timestamps.

    Requirements: 10.1
    """
    try:
        jobs = select(
            "jobs",
            columns=(
                "id, job_name, dataset_name, model_type, hyperparameters, "
                "shard_count, status, aggregated_metrics, error_summary, "
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
    """Return job detail including per-task status and aggregated metrics.

    Requirements: 6.3, 10.2, 10.3, 10.4
    """
    # Fetch the job record
    try:
        job = select_one("jobs", filters={"id": job_id})
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Fetch tasks for this job, including the latest metrics per task
    try:
        tasks = select(
            "tasks",
            columns=(
                "id, job_id, node_id, shard_index, status, task_config, "
                "checkpoint_path, error_message, assigned_at, started_at, "
                "completed_at, created_at"
            ),
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # For each task, attach the latest epoch metrics (most recent epoch row)
    task_ids = [t["id"] for t in tasks]
    latest_metrics_by_task: dict[str, dict[str, Any]] = {}

    if task_ids:
        try:
            # Fetch all metrics for this job, then pick the latest per task
            all_metrics = select(
                "metrics",
                columns="task_id, epoch, loss, accuracy",
                filters={"job_id": job_id},
            )
        except DatabaseError as exc:
            raise _db_error_to_http(exc) from exc

        for row in all_metrics:
            tid = row["task_id"]
            existing = latest_metrics_by_task.get(tid)
            if existing is None or row["epoch"] > existing["epoch"]:
                latest_metrics_by_task[tid] = row

    # Enrich task records with latest metrics
    enriched_tasks = []
    for task in tasks:
        latest = latest_metrics_by_task.get(task["id"])
        enriched_tasks.append(
            {
                **task,
                "latest_epoch": latest["epoch"] if latest else None,
                "latest_loss": latest["loss"] if latest else None,
                "latest_accuracy": latest["accuracy"] if latest else None,
            }
        )

    return {**job, "tasks": enriched_tasks}


# ---------------------------------------------------------------------------
# Job results endpoint
# ---------------------------------------------------------------------------


@router.get("/api/jobs/{job_id}/results", summary="Get aggregated metrics and checkpoint paths")
async def get_job_results(job_id: str) -> dict[str, Any]:
    """Return aggregated metrics and per-task checkpoint paths for a completed job.

    Requirements: 6.3, 7.3
    """
    try:
        job = select_one(
            "jobs",
            columns="id, status, aggregated_metrics, error_summary",
            filters={"id": job_id},
        )
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "Not Found"})
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Fetch per-task checkpoint paths
    try:
        tasks = select(
            "tasks",
            columns="id, shard_index, node_id, status, checkpoint_path, error_message",
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    return {
        "job_id": job["id"],
        "status": job["status"],
        "aggregated_metrics": job.get("aggregated_metrics"),
        "error_summary": job.get("error_summary"),
        "tasks": tasks,
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
                "storage_path, epoch, size_bytes, created_at"
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
    """Return counts of nodes and jobs broken down by status.

    Requirements: 11.2
    """
    try:
        nodes = select("nodes", columns="status")
        jobs = select("jobs", columns="status")
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Count nodes by status
    node_counts: dict[str, int] = {"idle": 0, "busy": 0, "offline": 0}
    for node in nodes:
        status = node.get("status", "")
        if status in node_counts:
            node_counts[status] += 1

    # Derive "online" as idle + busy
    online_nodes = node_counts["idle"] + node_counts["busy"]

    # Count jobs by status
    job_counts: dict[str, int] = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
    for job in jobs:
        status = job.get("status", "")
        if status in job_counts:
            job_counts[status] += 1

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
    }
