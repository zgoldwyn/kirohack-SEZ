"""Dashboard-facing read endpoints (unauthenticated, local/demo only).

These endpoints are intended for local development and demo use only.
They expose node status, job progress, aggregated metrics, and artifact
metadata without requiring authentication.

Endpoints:
    GET /api/nodes                  — List all nodes
    GET /api/nodes/{id}             — Node detail with task history
    GET /api/jobs                   — List all jobs
    GET /api/jobs/{id}              — Job detail with tasks and metrics
    GET /api/jobs/{id}/results      — Aggregated metrics + checkpoint paths
    GET /api/jobs/{id}/artifacts    — Artifacts for a job
    GET /api/monitoring/summary     — System-wide counts
    DELETE /api/jobs/{id}           — Delete a completed/failed job
    DELETE /api/nodes/{id}          — Delete an offline node
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
    """Return all registered nodes with status, hardware info, and last heartbeat."""
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
    """Return all jobs with status, model type, dataset, shard count, and timestamps."""
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
    """Return job detail including per-task status and aggregated metrics."""
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
                "checkpoint_path, error_message, assigned_at, started_at, "
                "completed_at, created_at"
            ),
            filters={"job_id": job_id},
        )
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # For each task, attach the latest epoch metrics
    task_ids = [t["id"] for t in tasks]
    latest_metrics_by_task: dict[str, dict[str, Any]] = {}

    if task_ids:
        try:
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
    """Return aggregated metrics and per-task checkpoint paths for a completed job."""
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
    """Return all artifact records (checkpoints, logs, outputs) for a job."""
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
    """Return counts of nodes and jobs broken down by status."""
    try:
        nodes = select("nodes", columns="status")
        jobs = select("jobs", columns="status")
    except DatabaseError as exc:
        raise _db_error_to_http(exc) from exc

    # Count nodes by status
    node_counts: dict[str, int] = {"idle": 0, "busy": 0, "offline": 0}
    for node in nodes:
        s = node.get("status", "")
        if s in node_counts:
            node_counts[s] += 1

    online_nodes = node_counts["idle"] + node_counts["busy"]

    # Count jobs by status
    job_counts: dict[str, int] = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
    for job in jobs:
        s = job.get("status", "")
        if s in job_counts:
            job_counts[s] += 1

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


# ---------------------------------------------------------------------------
# Delete job endpoint
# ---------------------------------------------------------------------------


@router.delete("/api/jobs/{job_id}", summary="Delete a job and its related data")
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job and cascade-delete its tasks, metrics, and artifacts.

    Only completed or failed jobs can be deleted.
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

    Only nodes with status 'offline' can be deleted.
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
