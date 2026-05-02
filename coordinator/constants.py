"""Shared constants and enums for the Group ML Trainer Coordinator."""

from enum import Enum


class NodeStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SupportedDataset(str, Enum):
    MNIST = "MNIST"
    FASHION_MNIST = "Fashion-MNIST"
    SYNTHETIC = "synthetic"


class SupportedModelType(str, Enum):
    MLP = "MLP"


class ArtifactType(str, Enum):
    CHECKPOINT = "checkpoint"
    LOG = "log"
    OUTPUT = "output"
