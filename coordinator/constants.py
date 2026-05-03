"""Shared constants and enums for the Group ML Trainer Coordinator."""

from enum import Enum


class NodeStatus(str, Enum):
    """Possible statuses for a registered worker node."""

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class JobStatus(str, Enum):
    """Possible statuses for a training job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Possible statuses for an individual task within a job."""

    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactType(str, Enum):
    """Types of artifacts produced by tasks."""

    CHECKPOINT = "checkpoint"
    LOG = "log"
    OUTPUT = "output"


# Supported datasets (core MVP)
SUPPORTED_DATASETS = {"MNIST", "Fashion-MNIST", "synthetic"}

# Supported model types
SUPPORTED_MODEL_TYPES = {"MLP"}


class TrainingRoundStatus(str, Enum):
    """Possible statuses for a training round."""

    IN_PROGRESS = "in_progress"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"


class SupportedDataset(str, Enum):
    """Datasets supported in the core MVP."""

    MNIST = "MNIST"
    FASHION_MNIST = "Fashion-MNIST"
    SYNTHETIC = "synthetic"


class SupportedModelType(str, Enum):
    """Model types supported in the core MVP."""

    MLP = "MLP"
