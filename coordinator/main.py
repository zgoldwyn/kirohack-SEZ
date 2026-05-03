"""Group ML Trainer — Coordinator (FastAPI entry point)."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from coordinator import db
from coordinator.aggregator import aggregate_round, check_job_failure, complete_job
from coordinator.auth import generate_token, get_current_node, hash_token
from coordinator.barrier import check_barrier, create_round, get_active_workers, record_submission
from coordinator.constants import JobStatus, NodeStatus, TaskStatus
from coordinator.config_parser import ConfigValidationError, parse_job_config
from coordinator.models import (
    GradientSubmissionRequest,
    JobSubmissionRequest,
    JobSubmissionResponse,
    NodeRegistrationRequest,
    NodeRegistrationResponse,
    ParameterDownloadResponse,
    TaskFailRequest,
)
from coordinator.heartbeat import heartbeat_monitor
from coordinator.models import TaskPollResponse
from coordinator.param_server import ParamServerError, get_parameters, initialize_model
from coordinator.scheduler import create_tasks_for_job, poll_task
from coordinator.storage import GRADIENTS_BUCKET, upload_blob

from coordinator.dashboard import router as dashboard_router

from coordinator.logging_config import setup_logging

load_dotenv()

# Configure structured logging before anything else logs.
setup_logging()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable is not set")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY environment variable is not set")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    monitor_task = heartbeat_monitor.start()
    yield
    heartbeat_monitor.stop()
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Group ML Trainer — Coordinator",
    description="Distributed ML task orchestration platform coordinator service.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Netlify frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://kirohacks.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dashboard read endpoints (unauthenticated, local/demo only)
app.include_router(dashboard_router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 7.1  POST /api/nodes/register
# ---------------------------------------------------------------------------


@app.post("/api/nodes/register", response_model=NodeRegistrationResponse)
async def register_node(body: NodeRegistrationRequest):
    """Register a new Worker node with the Coordinator.

    - Validate request body (handled by Pydantic model)
    - Check for duplicate node_id — return 409 if already registered
    - Generate auth token, hash it, store node record with status "idle"
    - Return node_db_id (database UUID) and the plaintext auth_token
    """
    # Check for duplicate node_id
    existing = db.select("nodes", columns="id", filters={"node_id": body.node_id})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"node_id '{body.node_id}' is already registered",
        )

    # Generate and hash auth token
    token = generate_token()
    token_hash = hash_token(token)

    # Build the node record
    node_data = {
        "node_id": body.node_id,
        "hostname": body.hostname,
        "cpu_cores": body.cpu_cores,
        "gpu_model": body.gpu_model,
        "vram_mb": body.vram_mb,
        "ram_mb": body.ram_mb,
        "disk_mb": body.disk_mb,
        "os": body.os,
        "python_version": body.python_version,
        "pytorch_version": body.pytorch_version,
        "status": NodeStatus.IDLE.value,
        "auth_token_hash": token_hash,
    }

    try:
        record = db.insert("nodes", node_data)
    except db.DatabaseError as exc:
        logger.error("Failed to register node '%s': %s", body.node_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    logger.info(
        "event=node_registered | node_id=%s | node_db_id=%s | hostname=%s | status=idle",
        body.node_id,
        record["id"],
        body.hostname,
    )

    return NodeRegistrationResponse(
        node_db_id=record["id"],
        auth_token=token,
    )


# ---------------------------------------------------------------------------
# 8.1  POST /api/nodes/heartbeat
# ---------------------------------------------------------------------------


@app.post("/api/nodes/heartbeat")
async def node_heartbeat(node: dict = Depends(get_current_node)):
    """Update heartbeat timestamp for the authenticated node.

    - Authenticate request using auth dependency
    - Update last_heartbeat timestamp
    - If node was "offline", set status back to "idle"
    Requirements: 2.1, 2.3
    """
    now = datetime.now(timezone.utc).isoformat()
    update_data: dict = {"last_heartbeat": now}

    if node.get("status") == NodeStatus.OFFLINE.value:
        update_data["status"] = NodeStatus.IDLE.value
        logger.info(
            "event=node_recovered | node_id=%s | node_db_id=%s | previous_status=offline | new_status=idle",
            node.get("node_id"),
            node["id"],
        )

    try:
        db.update("nodes", update_data, filters={"id": node["id"]})
    except db.DatabaseError as exc:
        logger.error("Failed to update heartbeat for node '%s': %s", node.get("node_id"), exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 9.1  POST /api/jobs
# ---------------------------------------------------------------------------


@app.post("/api/jobs", response_model=JobSubmissionResponse)
async def submit_job(body: JobSubmissionRequest):
    """Submit a new ML training job.

    - Validate request body (Pydantic handles required fields / types)
    - Validate dataset_name and model_type via config_parser
    - Create job record with status "queued"
    - Trigger task creation via scheduler
    - Tasks remain queued until workers poll and pick them up
    - Return job_id
    """
    # Validate dataset and model type, build structured config
    try:
        job_config = parse_job_config(body)
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors,
        ) from exc

    # Count idle nodes and reject if insufficient
    idle_nodes = db.select(
        "nodes",
        columns="id",
        filters={"status": NodeStatus.IDLE.value},
    )
    idle_count = len(idle_nodes)

    if body.shard_count > idle_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"shard_count ({body.shard_count}) exceeds the number of "
                f"idle nodes ({idle_count})"
            ),
        )

    # Derive total_rounds from hyperparameters (epochs)
    total_rounds = job_config.total_rounds

    # Create job record
    job_data = {
        "job_name": body.job_name,
        "dataset_name": body.dataset_name,
        "model_type": body.model_type,
        "hyperparameters": job_config.hyperparameters.model_dump(),
        "shard_count": body.shard_count,
        "status": JobStatus.QUEUED.value,
        "current_round": 0,
        "total_rounds": total_rounds,
    }

    try:
        job_record = db.insert("jobs", job_data)
    except db.DatabaseError as exc:
        logger.error("Failed to create job: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    job_id = job_record["id"]

    # Create tasks via scheduler
    try:
        create_tasks_for_job(job_id, job_config)
    except db.DatabaseError as exc:
        logger.error("Failed to create tasks for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    # Initialize Global_Model and store initial parameters
    try:
        global_model_path = initialize_model(job_id, job_config)
    except ParamServerError as exc:
        logger.error("Failed to initialize global model for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to initialize global model",
        ) from exc

    # Update job record with global_model_path
    try:
        db.update(
            "jobs",
            {"global_model_path": global_model_path},
            filters={"id": job_id},
        )
    except db.DatabaseError as exc:
        logger.error("Failed to update job %s with global_model_path: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    # Create initial training_rounds record for round 0
    try:
        create_round(
            job_id=job_id,
            round_number=0,
            active_worker_count=body.shard_count,
        )
    except db.DatabaseError as exc:
        logger.error("Failed to create initial training round for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    logger.info(
        "event=job_submitted | job_id=%s | dataset=%s | model_type=%s | shard_count=%d | total_rounds=%d | global_model_path=%s",
        job_id,
        body.dataset_name,
        body.model_type,
        body.shard_count,
        total_rounds,
        global_model_path,
    )

    return JobSubmissionResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# 11.1  GET /api/tasks/poll
# ---------------------------------------------------------------------------


@app.get("/api/tasks/poll", response_model=TaskPollResponse)
async def poll_for_task(node: dict = Depends(get_current_node)):
    """Worker polls for an eligible queued task.

    - Authenticate request using auth dependency
    - Check if polling node is idle (not already busy)
    - Select one eligible queued task based on resource requirements
    - Assign task to polling node, update statuses
    - If first task assigned for the job, set job to "running"
    - Return TaskPollResponse with task config, or empty response
    Requirements: 4.2, 4.3, 4.4, 4.5
    """
    return poll_task(node)


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
    """Mark a task as running and ensure round 0 tracking exists.

    - Verify task exists (404)
    - Verify task is assigned to requesting node (403)
    - Verify task status is "assigned" (409)
    - Update status → "running", set started_at
    - Idempotently ensure a training_rounds record exists for round 0
    Requirements: 5.1
    """
    task = _get_task_or_404(task_id)
    _verify_task_ownership(task, node)

    if task.get("status") != TaskStatus.ASSIGNED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task status is '{task.get('status')}', expected 'assigned'",
        )

    job_id = task["job_id"]
    now = datetime.now(timezone.utc).isoformat()

    db.update(
        "tasks",
        {"status": TaskStatus.RUNNING.value, "started_at": now},
        filters={"id": task_id},
    )

    # Idempotently ensure a training_rounds record exists for round 0.
    # The record is normally created during job submission, but we guard
    # against the case where it was not (e.g. older jobs, race conditions).
    existing_rounds = db.select(
        "training_rounds",
        filters={"job_id": job_id, "round_number": 0},
    )
    if not existing_rounds:
        active_count = len(get_active_workers(job_id))
        create_round(
            job_id=job_id,
            round_number=0,
            active_worker_count=active_count,
        )
        logger.info(
            "event=round_0_created_on_start | task_id=%s | job_id=%s | active_worker_count=%d",
            task_id,
            job_id,
            active_count,
        )

    logger.info(
        "event=task_started | task_id=%s | job_id=%s | node_id=%s | node_db_id=%s",
        task_id,
        job_id,
        node.get("node_id"),
        node["id"],
    )

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

    logger.info(
        "event=task_failed | task_id=%s | job_id=%s | node_id=%s | node_db_id=%s | error_message=%s",
        task_id,
        job_id,
        node.get("node_id"),
        node_id,
        body.error_message,
    )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 37.1  GET /api/jobs/{id}/parameters — Binary parameter download
# ---------------------------------------------------------------------------


def _get_active_task_for_node_in_job(job_id: str, node_id: str) -> dict:
    """Return the active task for *node_id* in *job_id*, or raise 403.

    A task is considered active if its status is ``assigned`` or
    ``running``.
    """
    tasks = db.select("tasks", filters={"job_id": job_id, "node_id": node_id})
    active_statuses = {TaskStatus.ASSIGNED.value, TaskStatus.RUNNING.value}
    for task in tasks:
        if task.get("status") in active_statuses:
            return task
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="No active task for this node in the requested job",
    )


@app.get("/api/jobs/{job_id}/parameters")
async def download_parameters(
    job_id: str,
    node: dict = Depends(get_current_node),
):
    """Download the current Global_Model parameters for a job.

    - Authenticate request using auth dependency
    - Verify the requesting node has an active task for this job
    - Fetch current Global_Model parameters via param_server
    - Return binary payload with Content-Type: application/octet-stream
    - Include ParameterDownloadResponse metadata in custom response headers
    - If job status is "completed" or "failed", return status so Worker
      can exit training loop
    Requirements: 4.4, 5.4, 13.1
    """
    # Verify node has an active task for this job
    _get_active_task_for_node_in_job(job_id, node["id"])

    # Fetch job record for metadata
    job_rows = db.select("jobs", filters={"id": job_id})
    if not job_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    job = job_rows[0]

    job_status = job.get("status", "")
    current_round = job.get("current_round", 0) or 0

    # Build metadata
    meta = ParameterDownloadResponse(
        job_id=job_id,
        current_round=current_round,
        job_status=job_status,
    )

    # If job is completed or failed, return metadata-only response so
    # the Worker knows to stop the training loop.
    if job_status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
        return Response(
            content=b"",
            media_type="application/octet-stream",
            headers={
                "X-Job-Id": meta.job_id,
                "X-Current-Round": str(meta.current_round),
                "X-Job-Status": meta.job_status,
            },
        )

    # Download parameters
    try:
        param_bytes = get_parameters(job_id)
    except ParamServerError as exc:
        logger.error(
            "Failed to download parameters for job %s: %s", job_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve model parameters",
        ) from exc

    logger.debug(
        "event=parameters_downloaded | job_id=%s | node_db_id=%s | round=%d | bytes=%d",
        job_id,
        node["id"],
        current_round,
        len(param_bytes),
    )

    return Response(
        content=param_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Job-Id": meta.job_id,
            "X-Current-Round": str(meta.current_round),
            "X-Job-Status": meta.job_status,
        },
    )


# ---------------------------------------------------------------------------
# 37.2  POST /api/jobs/{id}/gradients — Gradient submission
# ---------------------------------------------------------------------------


@app.post("/api/jobs/{job_id}/gradients")
async def submit_gradients(
    job_id: str,
    request: Request,
    node: dict = Depends(get_current_node),
):
    """Submit gradient update for the current training round.

    The request body is the raw binary gradient payload (application/octet-stream).
    Metadata is passed via query parameters: round_number, task_id,
    local_loss, local_accuracy.

    - Authenticate request using auth dependency
    - Verify the requesting node has an active task for this job
    - Validate round_number matches job's current_round (409 if mismatch)
    - Validate worker hasn't already submitted for this round (409 if dup)
    - Store gradient binary payload to Supabase Storage
    - Record submission via barrier.record_submission()
    - Store per-worker metrics in metrics table
    - Update task's last_submitted_round
    - Check barrier — if met, trigger aggregation
    Requirements: 5.1, 5.2, 5.3, 5.6, 13.2, 13.3
    """
    node_id = node["id"]

    # Parse metadata from query parameters
    params = request.query_params
    try:
        round_number = int(params["round_number"])
        task_id = str(params["task_id"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing or invalid required query parameters: round_number, task_id",
        ) from exc

    local_loss_raw = params.get("local_loss")
    local_accuracy_raw = params.get("local_accuracy")
    local_loss = float(local_loss_raw) if local_loss_raw is not None else None
    local_accuracy = float(local_accuracy_raw) if local_accuracy_raw is not None else None

    # Verify node has an active task for this job
    active_task = _get_active_task_for_node_in_job(job_id, node_id)

    # Verify the task_id matches the node's active task
    if active_task["id"] != task_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="task_id does not match the active task for this node",
        )

    # Fetch job record
    job_rows = db.select("jobs", filters={"id": job_id})
    if not job_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    job = job_rows[0]

    # Validate round_number matches job's current_round
    current_round = job.get("current_round", 0) or 0
    if round_number != current_round:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Round number mismatch: submitted round_number={round_number}, "
                f"job current_round={current_round}"
            ),
        )

    # Check for duplicate submission
    existing_submissions = db.select(
        "gradient_submissions",
        filters={
            "job_id": job_id,
            "task_id": task_id,
            "round_number": round_number,
        },
    )
    if existing_submissions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Worker already submitted gradients for round {round_number}",
        )

    # Read binary gradient payload from request body
    gradient_data = await request.body()
    if not gradient_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty gradient payload",
        )

    # Store gradient to Supabase Storage
    gradient_path = f"{job_id}/round_{round_number}/node_{node_id}.pt"
    try:
        upload_blob(GRADIENTS_BUCKET, gradient_path, gradient_data)
    except Exception as exc:
        logger.error(
            "Failed to upload gradient for job %s round %d node %s: %s",
            job_id,
            round_number,
            node_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to store gradient payload",
        ) from exc

    # Record submission via barrier (inserts gradient_submissions row and
    # increments submitted_count on training_rounds)
    try:
        submission = record_submission(
            job_id=job_id,
            round_number=round_number,
            task_id=task_id,
            node_id=node_id,
        )
        # Update the gradient_path on the submission record
        db.update(
            "gradient_submissions",
            {
                "gradient_path": gradient_path,
                "local_loss": local_loss,
                "local_accuracy": local_accuracy,
            },
            filters={"id": submission["id"]},
        )
    except db.DatabaseError as exc:
        logger.error(
            "Failed to record gradient submission for job %s round %d: %s",
            job_id,
            round_number,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to record gradient submission",
        ) from exc

    # Store per-worker metrics
    now = datetime.now(timezone.utc).isoformat()
    try:
        db.insert(
            "metrics",
            {
                "job_id": job_id,
                "task_id": task_id,
                "node_id": node_id,
                "round_number": round_number,
                "metric_type": "worker_local",
                "loss": local_loss,
                "accuracy": local_accuracy,
                "created_at": now,
            },
        )
    except db.DatabaseError:
        logger.exception(
            "event=worker_metric_insert_failed | job_id=%s | round=%d | task_id=%s",
            job_id,
            round_number,
            task_id,
        )

    # Update task's last_submitted_round
    try:
        db.update(
            "tasks",
            {"last_submitted_round": round_number},
            filters={"id": task_id},
        )
    except db.DatabaseError:
        logger.exception(
            "event=task_round_update_failed | job_id=%s | task_id=%s | round=%d",
            job_id,
            task_id,
            round_number,
        )

    # Check if barrier is met — if so, trigger aggregation
    barrier_met = check_barrier(job_id, round_number)

    if barrier_met:
        logger.info(
            "event=barrier_met | job_id=%s | round=%d — triggering aggregation",
            job_id,
            round_number,
        )
        try:
            aggregate_round(job_id, round_number)
        except Exception:
            logger.exception(
                "event=aggregation_failed | job_id=%s | round=%d",
                job_id,
                round_number,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gradient aggregation failed",
            )

    logger.info(
        "event=gradient_submitted | job_id=%s | round=%d | task_id=%s | node_db_id=%s | barrier_met=%s",
        job_id,
        round_number,
        task_id,
        node_id,
        barrier_met,
    )

    return {"status": "ok", "barrier_met": barrier_met}
