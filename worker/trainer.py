"""Collaborative training loop for the Worker.

Orchestrates the Worker's participation in a synchronized distributed
training session coordinated by the Parameter Server (Coordinator):

1. Load the dataset shard using :mod:`worker.datasets`.
2. Instantiate the model using :mod:`worker.models`.
3. Notify the Coordinator that the task has started.
4. Enter a round-based training loop for ``total_rounds`` iterations:
   a. Download the current Global_Model parameters from the Coordinator.
   b. If the job is completed or failed, exit the loop.
   c. Load the received parameters into the local model.
   d. Forward pass on the local data shard to compute loss and accuracy.
   e. Backward pass to compute gradients (do NOT call ``optimizer.step()``).
   f. Collect gradients into a ``state_dict``-like dict.
   g. Serialize gradients via ``torch.save`` to bytes.
   h. Submit gradients and local metrics to the Coordinator.
5. Return to polling after the training loop completes.

The Worker never applies optimizer steps — it only computes gradients.
The Coordinator aggregates gradients from all Workers and applies the
SGD update to the Global_Model.

Training exceptions (OOM, NaN loss, unexpected errors) are caught and
reported as task failures to the Coordinator so the job lifecycle can
proceed correctly.
"""

from __future__ import annotations

import io
import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from worker.config import TaskConfig
from worker.datasets import load_dataset
from worker.models import build_model
from worker.reporter import Reporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class NaNLossError(Exception):
    """Raised when training produces a NaN loss value."""


# ---------------------------------------------------------------------------
# Job terminal statuses — Worker should exit the training loop
# ---------------------------------------------------------------------------

_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_task(
    task_config: TaskConfig,
    reporter: Reporter,
    *,
    device: str | None = None,
) -> None:
    """Execute a collaborative training task.

    This is the main entry point called by the Worker's poll loop.  It
    handles the full lifecycle: start → round-based gradient computation
    → submit, with error handling that reports failures back to the
    Coordinator.

    Parameters
    ----------
    task_config:
        Validated task configuration from the Coordinator.
    reporter:
        HTTP client for communicating with the Coordinator.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  If ``None``,
        automatically selects CUDA when available, otherwise CPU.
    """
    task_id = task_config.task_id
    job_id = task_config.job_id
    total_rounds = task_config.total_rounds
    hp = task_config.hyperparameters

    try:
        # --- Notify Coordinator that we're starting ----------------------
        await reporter.start_task(task_id)
        logger.info("Task %s started (job=%s, total_rounds=%d)", task_id, job_id, total_rounds)

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

        # --- Loss function (no optimizer — Worker only computes grads) ---
        criterion = nn.CrossEntropyLoss()

        # --- Round-based training loop -----------------------------------
        for round_number in range(total_rounds):
            # 1. Download current Global_Model parameters
            param_result = await reporter.download_parameters(job_id)

            # 2. Check if job has reached a terminal state
            if param_result.job_status in _TERMINAL_JOB_STATUSES:
                logger.info(
                    "Task %s: job %s is %s — exiting training loop",
                    task_id,
                    job_id,
                    param_result.job_status,
                )
                return

            # 3. Load received parameters into local model
            param_state_dict = _deserialize_parameters(param_result.param_bytes, torch_device)
            model.load_state_dict(param_state_dict)

            # 4 & 5. Forward + backward pass to compute gradients and metrics
            local_loss, local_accuracy = _compute_gradients(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                device=torch_device,
                round_number=round_number,
            )

            # 6 & 7. Collect gradients and serialize to bytes
            gradient_bytes = _collect_and_serialize_gradients(model)

            # 8. Submit gradients and local metrics to Coordinator
            await reporter.submit_gradients(
                job_id,
                round_number=round_number,
                task_id=task_id,
                gradient_bytes=gradient_bytes,
                local_loss=local_loss,
                local_accuracy=local_accuracy,
            )

            logger.info(
                "Task %s round %d/%d — loss=%.4f accuracy=%.4f (gradients submitted)",
                task_id,
                round_number + 1,
                total_rounds,
                local_loss,
                local_accuracy,
            )

        # All rounds completed — the Coordinator will finalize the job.
        logger.info("Task %s completed all %d rounds", task_id, total_rounds)

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
# Gradient computation
# ---------------------------------------------------------------------------


def _compute_gradients(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    round_number: int,
) -> tuple[float, float]:
    """Run a forward + backward pass over the full shard and return metrics.

    The model's ``.grad`` attributes are populated after this call.
    The caller is responsible for collecting them.  ``optimizer.step()``
    is intentionally NOT called — the Coordinator applies the SGD update
    after aggregating gradients from all Workers.

    Parameters
    ----------
    model:
        The PyTorch model with Global_Model parameters loaded.
    dataloader:
        DataLoader yielding ``(inputs, labels)`` batches for this shard.
    criterion:
        The loss function (e.g. ``CrossEntropyLoss``).
    device:
        Device to run computation on.
    round_number:
        Current training round (for error messages).

    Returns
    -------
    tuple[float, float]
        ``(mean_loss, accuracy)`` computed over the entire shard.

    Raises
    ------
    NaNLossError
        If any batch produces a NaN loss value.
    """
    model.train()

    # Zero out any existing gradients before accumulating
    model.zero_grad()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Check for NaN loss
        if math.isnan(loss.item()):
            raise NaNLossError(
                f"NaN loss at round {round_number}, batch {batch_idx}"
            )

        # Backward pass — accumulates gradients across all batches
        loss.backward()

        # Accumulate statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_accuracy = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------------------------
# Gradient collection and serialization
# ---------------------------------------------------------------------------


def _collect_and_serialize_gradients(model: nn.Module) -> bytes:
    """Collect gradients from model parameters and serialize to bytes.

    Builds a ``state_dict``-like dictionary mapping parameter names to
    their ``.grad`` tensors, then serializes it using ``torch.save``
    into an in-memory bytes buffer.

    Parameters
    ----------
    model:
        The model whose ``.grad`` attributes have been populated by a
        backward pass.

    Returns
    -------
    bytes
        Serialized gradient dict ready for submission to the Coordinator.
    """
    gradient_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Detach and clone to avoid holding references to the
            # computation graph, and move to CPU for serialization.
            gradient_dict[name] = param.grad.detach().cpu().clone()

    buffer = io.BytesIO()
    torch.save(gradient_dict, buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parameter deserialization
# ---------------------------------------------------------------------------


def _deserialize_parameters(param_bytes: bytes, device: torch.device) -> dict[str, torch.Tensor]:
    """Deserialize Global_Model parameters from bytes.

    Parameters
    ----------
    param_bytes:
        Raw bytes produced by ``torch.save`` on the Coordinator side.
    device:
        Target device to load tensors onto.

    Returns
    -------
    dict[str, torch.Tensor]
        A ``state_dict`` mapping parameter names to tensors.
    """
    buffer = io.BytesIO(param_bytes)
    state_dict = torch.load(buffer, map_location=device, weights_only=True)
    return state_dict


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
