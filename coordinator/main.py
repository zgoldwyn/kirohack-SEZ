"""Group ML Trainer — Coordinator (FastAPI entry point)."""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status

from coordinator import db
from coordinator.aggregator import aggregate_job_metrics, check_job_failure
from coordinator.auth import get_current_node
from coordinator.constants import ArtifactType, NodeStatus, TaskStatus
from coordinator.models import MetricsReportRequest, TaskCompleteRequest, TaskFailRequest
from coordinator.storage import generate_signed_upload_url

from coordinator.dashboard import router as dashboard_router

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable is not set")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY environment variable is not set")

app = FastAPI(
    title="Group ML Trainer — Coordinator",
    description="Distributed ML task orchestration platform coordinator service.",
    version="0.1.0",
)

# Dashboard read endpoints (unauthenticated, local/demo only)
app.include_router(dashboard_router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helper: fetch task and verify ownership
# ---------------------------------------------------------------------------


def _get_task_or_404(task_id: str) -> dict:
    """Fetch a task by ID or raise HTTP 404."""
    rows = db.select("tasks", filters={"id": task_id})
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )
    return rows[0]


def _verify_task_ownership(task: dict, node: dict) -> None:
    """Raise HTTP 403 if the task is not assigned to the given node."""
    if task.get("node_id") != node["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Task is not assigned to this node",
        )


# ---------------------------------------------------------------------------
# 12.1  POST /api/tasks/{id}/start
# ---------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/start")
async def start_task(
    task_id: str,
    node: dict = Depends(get_current_node),
):
    """Mark a task as running.

    - Verify task exists (404)
    - Verify task is assigned to requesting node (403)
    - Verify task status is "assigned" (409)
    - Update status → "running", set started_at
    """
    task = _get_task_or_404(task_id)
    _verify_task_ownership(task, node)

    if task.get("status") != TaskStatus.ASSIGNED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task status is '{task.get('status')}', expected 'assigned'",
        )

    now = datetime.now(timezone.utc).isoformat()
    db.update(
        "tasks",
        {"status": TaskStatus.RUNNING.value, "started_at": now},
        filters={"id": task_id},
    )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 12.2  POST /api/tasks/{id}/complete
# ---------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    body: TaskCompleteRequest,
    node: dict = Depends(get_current_node),
):
    """Mark a task as completed.

    - Verify task exists (404) and belongs to requesting node (403)
    - Update task: status → "completed", checkpoint_path, completed_at
    - Insert artifact record
    - Set node status → "idle"
    - Check if all tasks in job completed → aggregate
    - Check if job should be marked failed
    """
    task = _get_task_or_404(task_id)
    _verify_task_ownership(task, node)

    job_id = task["job_id"]
    node_id = node["id"]
    now = datetime.now(timezone.utc).isoformat()

    # Update task
    db.update(
        "tasks",
        {
            "status": TaskStatus.COMPLETED.value,
            "checkpoint_path": body.checkpoint_path,
            "completed_at": now,
        },
        filters={"id": task_id},
    )

    # Insert artifact record
    db.insert(
        "artifacts",
        {
            "job_id": job_id,
            "task_id": task_id,
            "node_id": node_id,
            "artifact_type": ArtifactType.CHECKPOINT.value,
            "storage_path": body.checkpoint_path,
        },
    )

    # Set node back to idle
    db.update(
        "nodes",
        {"status": NodeStatus.IDLE.value},
        filters={"id": node_id},
    )

    # Check if ALL tasks in the job are now completed → aggregate
    all_tasks = db.select("tasks", filters={"job_id": job_id})
    all_completed = all(
        t.get("status") == TaskStatus.COMPLETED.value for t in all_tasks
    )
    if all_completed:
        aggregate_job_metrics(job_id)
    else:
        # Some tasks may have failed; check if job should be marked failed
        check_job_failure(job_id)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 12.3  POST /api/tasks/{id}/fail
# ---------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/fail")
async def fail_task(
    task_id: str,
    body: TaskFailRequest,
    node: dict = Depends(get_current_node),
):
    """Mark a task as failed.

    - Verify task exists (404) and belongs to requesting node (403)
    - Update task: status → "failed", error_message, completed_at
    - Set node status → "idle"
    - Check if job should be marked failed
    """
    task = _get_task_or_404(task_id)
    _verify_task_ownership(task, node)

    job_id = task["job_id"]
    node_id = node["id"]
    now = datetime.now(timezone.utc).isoformat()

    # Update task
    db.update(
        "tasks",
        {
            "status": TaskStatus.FAILED.value,
            "error_message": body.error_message,
            "completed_at": now,
        },
        filters={"id": task_id},
    )

    # Set node back to idle
    db.update(
        "nodes",
        {"status": NodeStatus.IDLE.value},
        filters={"id": node_id},
    )

    # Check if job should be marked failed
    check_job_failure(job_id)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 12.4  POST /api/tasks/{id}/upload-url
# ---------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/upload-url")
async def request_upload_url(
    task_id: str,
    node: dict = Depends(get_current_node),
):
    """Generate a signed upload URL for the task's checkpoint.

    - Verify task exists (404) and belongs to requesting node (403)
    - Generate signed URL for path {job_id}/{task_id}/final.pt
    - Return {"signed_url": "<url>"}
    """
    task = _get_task_or_404(task_id)
    _verify_task_ownership(task, node)

    job_id = task["job_id"]

    try:
        signed_url = generate_signed_upload_url(job_id, task_id)
    except Exception as exc:
        logger.error("Failed to generate signed upload URL: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate signed upload URL",
        ) from exc

    return {"signed_url": signed_url}


# ---------------------------------------------------------------------------
# 13.1  POST /api/metrics
# ---------------------------------------------------------------------------


@app.post("/api/metrics")
async def report_metrics(
    body: MetricsReportRequest,
    node: dict = Depends(get_current_node),
):
    """Report per-epoch training metrics for a task.

    - Authenticate request
    - Verify the referenced task exists and belongs to the requesting node
    - Insert a metrics record with job_id, task_id, node_id, epoch, loss, accuracy
    """
    task = _get_task_or_404(body.task_id)
    _verify_task_ownership(task, node)

    db.insert(
        "metrics",
        {
            "job_id": task["job_id"],
            "task_id": body.task_id,
            "node_id": node["id"],
            "epoch": body.epoch,
            "loss": body.loss,
            "accuracy": body.accuracy,
        },
    )

    return {"status": "ok"}
