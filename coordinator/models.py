"""Pydantic request and response models for the Coordinator API.

Only the models needed for the task lifecycle endpoints are defined here.
Additional models (registration, job submission, etc.) will be added as
their respective endpoints are implemented.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskCompleteRequest(BaseModel):
    """Body for POST /api/tasks/{id}/complete."""

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
