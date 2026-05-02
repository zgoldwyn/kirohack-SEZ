"""PyTorch training loop for the Worker.

Orchestrates the full lifecycle of a single training task:

1. Load the dataset shard using :mod:`worker.datasets`.
2. Instantiate the model using :mod:`worker.models`.
3. Train for the configured number of epochs, reporting per-epoch
   metrics (loss, accuracy) to the Coordinator via :mod:`worker.reporter`.
4. Save the final checkpoint to a local temporary file.
5. Upload the checkpoint via :mod:`worker.storage` and report completion.

Training exceptions (OOM, NaN loss, unexpected errors) are caught and
reported as task failures to the Coordinator so the job lifecycle can
proceed correctly.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from worker.config import TaskConfig
from worker.datasets import load_dataset
from worker.models import build_model
from worker.reporter import Reporter
from worker.storage import StorageClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class NaNLossError(Exception):
    """Raised when training produces a NaN loss value."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_task(
    task_config: TaskConfig,
    reporter: Reporter,
    storage_client: StorageClient,
    *,
    device: str | None = None,
) -> None:
    """Execute a complete training task.

    This is the main entry point called by the Worker's poll loop.  It
    handles the full lifecycle: start → train → upload → complete, with
    error handling that reports failures back to the Coordinator.

    Parameters
    ----------
    task_config:
        Validated task configuration from the Coordinator.
    reporter:
        HTTP client for communicating with the Coordinator.
    storage_client:
        Client for uploading checkpoints via signed URLs.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  If ``None``,
        automatically selects CUDA when available, otherwise CPU.
    """
    task_id = task_config.task_id
    hp = task_config.hyperparameters

    try:
        # --- Notify Coordinator that we're starting ----------------------
        await reporter.start_task(task_id)
        logger.info("Task %s started", task_id)

        # --- Resolve device ----------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)
        logger.info("Training on device: %s", torch_device)

        # --- Load dataset shard ------------------------------------------
        dataset = load_dataset(
            dataset_name=task_config.dataset_name,
            shard_index=task_config.shard_index,
            shard_count=task_config.shard_count,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=hp.batch_size,
            shuffle=True,
        )
        logger.info(
            "Loaded dataset shard %d/%d (%d samples, %d batches)",
            task_config.shard_index,
            task_config.shard_count,
            len(dataset),
            len(dataloader),
        )

        # --- Build model -------------------------------------------------
        model = build_model(
            dataset_name=task_config.dataset_name,
            model_type=task_config.model_type,
            hidden_layers=hp.hidden_layers,
            activation=hp.activation,
        )
        model = model.to(torch_device)
        logger.info("Model built: %s", task_config.model_type)

        # --- Set up optimizer and loss -----------------------------------
        optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # --- Training loop -----------------------------------------------
        final_loss: float | None = None
        final_accuracy: float | None = None

        for epoch in range(hp.epochs):
            epoch_loss, epoch_accuracy = _train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=torch_device,
                epoch=epoch,
            )

            # Report metrics to Coordinator
            await reporter.report_metrics(
                task_id=task_id,
                epoch=epoch,
                loss=epoch_loss,
                accuracy=epoch_accuracy,
            )
            logger.info(
                "Task %s epoch %d/%d — loss=%.4f accuracy=%.4f",
                task_id,
                epoch + 1,
                hp.epochs,
                epoch_loss,
                epoch_accuracy,
            )

            final_loss = epoch_loss
            final_accuracy = epoch_accuracy

        # --- Save checkpoint to temp file --------------------------------
        checkpoint_path = _save_checkpoint(model, task_id)
        logger.info("Checkpoint saved to %s", checkpoint_path)

        # --- Upload checkpoint -------------------------------------------
        await storage_client.upload_checkpoint(task_id, checkpoint_path)
        logger.info("Checkpoint uploaded for task %s", task_id)

        # --- Report completion -------------------------------------------
        # The storage path follows the convention {job_id}/{task_id}/final.pt
        storage_path = f"{task_config.job_id}/{task_id}/final.pt"
        await reporter.complete_task(
            task_id,
            checkpoint_path=storage_path,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
        )
        logger.info("Task %s completed successfully", task_id)

        # --- Clean up temp file ------------------------------------------
        _cleanup_checkpoint(checkpoint_path)

    except NaNLossError as exc:
        error_msg = f"Training produced NaN loss: {exc}"
        logger.error("Task %s failed: %s", task_id, error_msg)
        await _safe_fail_task(reporter, task_id, error_msg)

    except torch.cuda.OutOfMemoryError as exc:
        error_msg = f"CUDA out of memory: {exc}"
        logger.error("Task %s failed: %s", task_id, error_msg)
        await _safe_fail_task(reporter, task_id, error_msg)

    except Exception as exc:
        error_msg = f"Training failed with unexpected error: {type(exc).__name__}: {exc}"
        logger.error("Task %s failed: %s", task_id, error_msg, exc_info=True)
        await _safe_fail_task(reporter, task_id, error_msg)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train the model for one epoch and return (loss, accuracy).

    Parameters
    ----------
    model:
        The PyTorch model to train.
    dataloader:
        DataLoader yielding ``(inputs, labels)`` batches.
    optimizer:
        The optimizer instance.
    criterion:
        The loss function (e.g. ``CrossEntropyLoss``).
    device:
        Device to run training on.
    epoch:
        Current epoch number (for logging).

    Returns
    -------
    tuple[float, float]
        ``(mean_loss, accuracy)`` over the entire epoch.

    Raises
    ------
    NaNLossError
        If any batch produces a NaN loss value.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Check for NaN loss
        if math.isnan(loss.item()):
            raise NaNLossError(
                f"NaN loss at epoch {epoch}, batch {batch_idx}"
            )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_accuracy = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(model: nn.Module, task_id: str) -> Path:
    """Save model state dict to a temporary file and return its path.

    The file is created in the system temp directory with a name that
    includes the task ID for easy identification.  The caller is
    responsible for cleaning up the file after upload.
    """
    tmp_dir = Path(tempfile.gettempdir()) / "group-ml-trainer"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = tmp_dir / f"{task_id}_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def _cleanup_checkpoint(checkpoint_path: Path) -> None:
    """Remove a temporary checkpoint file.  Logs but does not raise on failure."""
    try:
        checkpoint_path.unlink(missing_ok=True)
        logger.debug("Cleaned up temp checkpoint %s", checkpoint_path)
    except OSError as exc:
        logger.warning("Failed to clean up checkpoint %s: %s", checkpoint_path, exc)


# ---------------------------------------------------------------------------
# Error reporting helper
# ---------------------------------------------------------------------------


async def _safe_fail_task(
    reporter: Reporter,
    task_id: str,
    error_message: str,
) -> None:
    """Best-effort report of task failure to the Coordinator.

    If reporting itself fails (e.g. Coordinator unreachable), the error
    is logged but not re-raised — the original exception context is
    preserved for the caller.
    """
    try:
        await reporter.fail_task(task_id, error_message=error_message)
        logger.info("Reported failure for task %s to Coordinator", task_id)
    except Exception as report_exc:
        logger.error(
            "Failed to report task %s failure to Coordinator: %s",
            task_id,
            report_exc,
        )
