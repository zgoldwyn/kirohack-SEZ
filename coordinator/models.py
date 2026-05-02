"""Pydantic request and response models for the Coordinator API.

Only the models needed for the task lifecycle endpoints are defined here.
Additional models (registration, job submission, etc.) will be added as
their respective endpoints are implemented.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskCompleteRequest(BaseModel):
    """Body for POST /api/tasks/{id}/complete."""
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


class MetricsReportRequest(BaseModel):
    """Per-epoch metrics reported by a Worker during training."""

    task_id: str
    epoch: int = Field(ge=0)
    loss: float | None = None
    accuracy: float | None = None


class TaskCompleteRequest(BaseModel):
    """Sent by a Worker when a task finishes successfully."""

    checkpoint_path: str
    final_loss: float | None = None
    final_accuracy: float | None = None


class TaskFailRequest(BaseModel):
    """Body for POST /api/tasks/{id}/fail."""

    error_message: str


class MetricsReportRequest(BaseModel):
    """Body for POST /api/metrics."""

    task_id: str
    epoch: int = Field(ge=0)
    loss: float | None = None
    accuracy: float | None = None
