"""Task configuration parsing for the Worker.

Parses the TaskConfig JSON payload received from the Coordinator's
``/api/tasks/poll`` endpoint into structured, validated Python objects
that the trainer, dataset loader, and model builder can consume directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HyperParameters(BaseModel):
    """Training hyper-parameters embedded in every task configuration."""

    learning_rate: float = Field(gt=0, default=0.001)
    epochs: int = Field(gt=0, default=10)
    batch_size: int = Field(gt=0, default=32)
    hidden_layers: list[int] = Field(default_factory=lambda: [128, 64])
    activation: str = Field(default="relu")


class TaskConfig(BaseModel):
    """Full configuration payload sent by the Coordinator for a single task.

    Workers receive this as the body of a successful ``GET /api/tasks/poll``
    response.  All fields required for training are present so the Worker
    never needs to call back to the Coordinator for additional config.
    """

    task_id: str
    job_id: str
    dataset_name: str
    model_type: str
    hyperparameters: HyperParameters
    shard_index: int
    shard_count: int = Field(gt=0)
    total_rounds: int = Field(gt=0)


def parse_task_config(raw: dict | str | bytes) -> TaskConfig:
    """Parse a raw JSON payload (dict, string, or bytes) into a ``TaskConfig``.

    Parameters
    ----------
    raw:
        The JSON payload from the Coordinator.  Accepts a pre-parsed *dict*,
        a JSON *str*, or raw *bytes*.

    Returns
    -------
    TaskConfig
        Validated task configuration ready for the training pipeline.

    Raises
    ------
    pydantic.ValidationError
        If required fields are missing or values are out of range.
    """
    if isinstance(raw, (str, bytes)):
        return TaskConfig.model_validate_json(raw)
    return TaskConfig.model_validate(raw)
