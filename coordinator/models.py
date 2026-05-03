"""Pydantic request and response models for the Coordinator API.

Includes request/response models for all endpoints as well as internal
configuration models (JobConfig, TaskConfig, HyperParameters) used by
the config parser and scheduler.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Internal Configuration Models
# ---------------------------------------------------------------------------


class HyperParameters(BaseModel):
    """Training hyper-parameters for a job/task."""

    learning_rate: float = Field(gt=0, default=0.001)
    epochs: int = Field(gt=0, default=10)
    batch_size: int = Field(gt=0, default=32)
    hidden_layers: list[int] = Field(default_factory=lambda: [128, 64])
    activation: str = Field(default="relu")


class JobConfig(BaseModel):
    """Internal structured representation of a job configuration.

    Used for validation, serialization round-trips, and task config generation.
    """

    dataset_name: str
    model_type: str
    hyperparameters: HyperParameters
    shard_count: int = Field(gt=0)


class TaskConfig(BaseModel):
    """Configuration payload sent to a Worker for a single task."""

    task_id: str
    job_id: str
    dataset_name: str
    model_type: str
    hyperparameters: HyperParameters
    shard_index: int
    shard_count: int
    total_rounds: int


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class NodeRegistrationRequest(BaseModel):
    """Payload sent by a Worker to register with the Coordinator."""

    node_id: str
    hostname: str
    cpu_cores: int = Field(gt=0)
    gpu_model: str | None = None
    vram_mb: int | None = None
    ram_mb: int = Field(gt=0)
    disk_mb: int = Field(gt=0)
    os: str
    python_version: str
    pytorch_version: str


class JobSubmissionRequest(BaseModel):
    """Payload sent by a user/dashboard to submit a training job."""

    job_name: str | None = None
    dataset_name: str
    model_type: str
    hyperparameters: dict = Field(default_factory=dict)
    shard_count: int = Field(gt=0)


class GradientSubmissionRequest(BaseModel):
    """Metadata sent alongside the binary gradient payload."""

    round_number: int = Field(ge=0)
    task_id: str
    local_loss: float | None = None
    local_accuracy: float | None = None


class TaskFailRequest(BaseModel):
    """Sent by a Worker when a task fails."""

    error_message: str


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class NodeRegistrationResponse(BaseModel):
    """Returned after successful node registration."""

    node_db_id: str
    auth_token: str


class JobSubmissionResponse(BaseModel):
    """Returned after successful job submission."""

    job_id: str


class TaskPollResponse(BaseModel):
    """Returned when a Worker polls for work."""

    task_id: str | None = None
    job_id: str | None = None
    dataset_name: str | None = None
    model_type: str | None = None
    hyperparameters: dict | None = None
    shard_index: int | None = None
    shard_count: int | None = None
    total_rounds: int | None = None


class ParameterDownloadResponse(BaseModel):
    """Metadata returned alongside binary parameter payload."""

    job_id: str
    current_round: int
    job_status: str


class RoundStatus(BaseModel):
    """Status of a training round for dashboard display."""

    round_number: int
    status: str
    active_worker_count: int
    submitted_count: int
    global_loss: float | None = None
    global_accuracy: float | None = None


class AggregatedMetrics(BaseModel):
    """Aggregated metrics for a completed job."""

    mean_loss: float | None = None
    mean_accuracy: float | None = None
    per_node: list[dict] = Field(default_factory=list)
