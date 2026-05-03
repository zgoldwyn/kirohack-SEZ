"""Job configuration parsing, validation, and task config generation."""

from __future__ import annotations

from dataclasses import dataclass

from coordinator.constants import SupportedDataset, SupportedModelType
from coordinator.models import (
    HyperParameters,
    JobConfig,
    JobSubmissionRequest,
    TaskConfig,
)


# ---------------------------------------------------------------------------
# Resource requirements per model type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceRequirements:
    """Minimum hardware a node must have to run a given model type."""

    min_ram_mb: int
    gpu_required: bool


# Lookup table — extend when new model types are added.
MODEL_RESOURCE_REQUIREMENTS: dict[str, ResourceRequirements] = {
    SupportedModelType.MLP.value: ResourceRequirements(min_ram_mb=512, gpu_required=False),
}


def get_resource_requirements(model_type: str) -> ResourceRequirements:
    """Return the resource requirements for *model_type*.

    Raises ``ValueError`` if the model type has no resource profile.
    """
    try:
        return MODEL_RESOURCE_REQUIREMENTS[model_type]
    except KeyError:
        raise ValueError(
            f"No resource requirements defined for model type '{model_type}'. "
            f"Supported: {list(MODEL_RESOURCE_REQUIREMENTS)}"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_SUPPORTED_DATASETS: set[str] = {d.value for d in SupportedDataset}
_SUPPORTED_MODEL_TYPES: set[str] = {m.value for m in SupportedModelType}


class ConfigValidationError(Exception):
    """Raised when a job configuration fails validation.

    Attributes:
        errors: list of per-field error descriptions.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def _validate_dataset(dataset_name: str) -> list[str]:
    if dataset_name not in _SUPPORTED_DATASETS:
        return [
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Supported: {sorted(_SUPPORTED_DATASETS)}"
        ]
    return []


def _validate_model_type(model_type: str) -> list[str]:
    if model_type not in _SUPPORTED_MODEL_TYPES:
        return [
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {sorted(_SUPPORTED_MODEL_TYPES)}"
        ]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_job_config(request: JobSubmissionRequest) -> JobConfig:
    """Parse and validate a ``JobSubmissionRequest`` into a ``JobConfig``.

    Raises:
        ConfigValidationError: if dataset or model type is unsupported.
        pydantic.ValidationError: if hyperparameters have invalid types/ranges.
    """
    errors: list[str] = []
    errors.extend(_validate_dataset(request.dataset_name))
    errors.extend(_validate_model_type(request.model_type))

    if errors:
        raise ConfigValidationError(errors)

    # Build structured hyperparameters — Pydantic will reject bad types/ranges.
    hyper = HyperParameters(**request.hyperparameters)

    return JobConfig(
        dataset_name=request.dataset_name,
        model_type=request.model_type,
        hyperparameters=hyper,
        shard_count=request.shard_count,
    )


def generate_task_configs(
    job_config: JobConfig,
    job_id: str,
    task_ids: list[str],
) -> list[TaskConfig]:
    """Create one ``TaskConfig`` per shard from a validated ``JobConfig``.

    Args:
        job_config: Validated job configuration.
        job_id: Database UUID of the parent job.
        task_ids: Pre-generated UUIDs for each task (length must equal
            ``job_config.shard_count``).

    Returns:
        A list of ``TaskConfig`` objects, one per shard index.

    Raises:
        ValueError: if ``len(task_ids) != job_config.shard_count``.
    """
    if len(task_ids) != job_config.shard_count:
        raise ValueError(
            f"Expected {job_config.shard_count} task IDs, got {len(task_ids)}"
        )

    return [
        TaskConfig(
            task_id=task_ids[i],
            job_id=job_id,
            dataset_name=job_config.dataset_name,
            model_type=job_config.model_type,
            hyperparameters=job_config.hyperparameters,
            shard_index=i,
            shard_count=job_config.shard_count,
            total_rounds=job_config.hyperparameters.epochs,
        )
        for i in range(job_config.shard_count)
    ]
