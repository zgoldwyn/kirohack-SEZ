"""Unit tests for coordinator/aggregator.py — gradient aggregation, job
completion, and job failure detection.

Tests cover:
- aggregate_round: load gradients → compute mean → apply SGD step → verify
  updated parameters
- complete_job: stores final checkpoint and aggregated metrics
- check_job_failure: marks job failed with partial checkpoint storage
"""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock, call

import pytest
import torch

from coordinator.aggregator import aggregate_round, complete_job, check_job_failure
from coordinator.constants import JobStatus, TaskStatus, TrainingRoundStatus


# ---------------------------------------------------------------------------
# Helper: serialize a state_dict to bytes (same format as production code)
# ---------------------------------------------------------------------------


def _serialize(state_dict: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# aggregate_round
# ---------------------------------------------------------------------------


class TestAggregateRound:
    """Tests for aggregate_round — gradient aggregation with SGD step."""

    def test_load_gradients_compute_mean_apply_sgd(self):
        """Load gradients from storage, compute element-wise mean, apply
        SGD step (param -= lr * mean_grad), and verify updated parameters
        are uploaded via param_server.
        Requirements: 5.3, 6.1
        """
        # Setup: two workers submit gradients for round 0
        lr = 0.01
        initial_params = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        grad_worker_1 = {"weight": torch.tensor([0.1, 0.2, 0.3])}
        grad_worker_2 = {"weight": torch.tensor([0.3, 0.4, 0.5])}

        # Expected: mean_grad = [0.2, 0.3, 0.4]
        # new_params = [1.0 - 0.01*0.2, 2.0 - 0.01*0.3, 3.0 - 0.01*0.4]
        #            = [0.998, 1.997, 2.996]

        round_record = {
            "id": "round-0",
            "job_id": "job-1",
            "round_number": 0,
            "status": TrainingRoundStatus.IN_PROGRESS.value,
        }
        submissions = [
            {
                "id": "sub-1",
                "job_id": "job-1",
                "task_id": "task-1",
                "node_id": "node-1",
                "round_number": 0,
                "gradient_path": "job-1/round_0/node_node-1.pt",
                "local_loss": 0.5,
                "local_accuracy": 0.8,
            },
            {
                "id": "sub-2",
                "job_id": "job-1",
                "task_id": "task-2",
                "node_id": "node-2",
                "round_number": 0,
                "gradient_path": "job-1/round_0/node_node-2.pt",
                "local_loss": 0.6,
                "local_accuracy": 0.75,
            },
        ]
        job = {
            "id": "job-1",
            "hyperparameters": {"learning_rate": lr, "epochs": 3},
            "total_rounds": 3,
        }

        def mock_select(table, columns="*", filters=None):
            if table == "training_rounds":
                if filters and filters.get("job_id") == "job-1":
                    return [round_record]
            if table == "gradient_submissions":
                return submissions
            if table == "jobs":
                return [job]
            if table == "tasks":
                return [
                    {"id": "task-1", "status": TaskStatus.RUNNING.value},
                    {"id": "task-2", "status": TaskStatus.RUNNING.value},
                ]
            if table == "metrics":
                return []
            return []

        grad_data = {
            "job-1/round_0/node_node-1.pt": _serialize(grad_worker_1),
            "job-1/round_0/node_node-2.pt": _serialize(grad_worker_2),
        }

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps, \
             patch("coordinator.aggregator.download_blob") as mock_dl, \
             patch("coordinator.aggregator.delete_blob"), \
             patch("coordinator.aggregator.barrier_mod") as mock_barrier:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_db.insert.return_value = {}
            mock_dl.side_effect = lambda bucket, path: grad_data[path]
            mock_ps.get_parameters.return_value = _serialize(initial_params)
            mock_barrier.get_active_workers.return_value = {"task-1", "task-2"}
            mock_barrier.create_round.return_value = {}

            aggregate_round("job-1", 0)

            # Verify param_server.update_parameters was called with the
            # correct SGD-updated state_dict
            mock_ps.update_parameters.assert_called_once()
            call_args = mock_ps.update_parameters.call_args
            assert call_args[0][0] == "job-1"
            new_params = call_args[0][1]
            expected = initial_params["weight"] - lr * torch.tensor([0.2, 0.3, 0.4])
            assert torch.allclose(new_params["weight"], expected, atol=1e-6)

    def test_aggregate_round_advances_to_next_round(self):
        """After aggregation, current_round is incremented and a new round
        record is created for the next round.
        Requirements: 5.3, 6.1
        """
        initial_params = {"w": torch.tensor([1.0])}
        grad = {"w": torch.tensor([0.1])}
        round_record = {
            "id": "round-0",
            "job_id": "job-1",
            "round_number": 0,
            "status": TrainingRoundStatus.IN_PROGRESS.value,
        }
        submissions = [
            {
                "id": "sub-1",
                "job_id": "job-1",
                "task_id": "task-1",
                "node_id": "node-1",
                "round_number": 0,
                "gradient_path": "job-1/round_0/node_node-1.pt",
                "local_loss": 0.5,
                "local_accuracy": 0.8,
            },
        ]
        job = {
            "id": "job-1",
            "hyperparameters": {"learning_rate": 0.01, "epochs": 5},
            "total_rounds": 5,
        }

        def mock_select(table, columns="*", filters=None):
            if table == "training_rounds":
                return [round_record]
            if table == "gradient_submissions":
                return submissions
            if table == "jobs":
                return [job]
            if table == "tasks":
                return [{"id": "task-1", "status": TaskStatus.RUNNING.value}]
            if table == "metrics":
                return []
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps, \
             patch("coordinator.aggregator.download_blob") as mock_dl, \
             patch("coordinator.aggregator.delete_blob"), \
             patch("coordinator.aggregator.barrier_mod") as mock_barrier:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_db.insert.return_value = {}
            mock_dl.return_value = _serialize(grad)
            mock_ps.get_parameters.return_value = _serialize(initial_params)
            mock_barrier.get_active_workers.return_value = {"task-1"}
            mock_barrier.create_round.return_value = {}

            aggregate_round("job-1", 0)

            # Verify job's current_round was advanced to 1
            job_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "jobs" and "current_round" in c[0][1]
            ]
            assert len(job_update_calls) == 1
            assert job_update_calls[0][0][1]["current_round"] == 1

            # Verify next round was created
            mock_barrier.create_round.assert_called_once_with("job-1", 1, 1)

    def test_aggregate_round_calls_complete_job_on_final_round(self):
        """When next_round >= total_rounds, aggregate_round calls
        complete_job instead of creating a new round.
        Requirements: 6.1, 6.4
        """
        initial_params = {"w": torch.tensor([1.0])}
        grad = {"w": torch.tensor([0.1])}
        # This is the final round (round 2 of 3 total, so next_round=3 >= 3)
        round_record = {
            "id": "round-2",
            "job_id": "job-1",
            "round_number": 2,
            "status": TrainingRoundStatus.IN_PROGRESS.value,
        }
        submissions = [
            {
                "id": "sub-1",
                "job_id": "job-1",
                "task_id": "task-1",
                "node_id": "node-1",
                "round_number": 2,
                "gradient_path": "job-1/round_2/node_node-1.pt",
                "local_loss": 0.2,
                "local_accuracy": 0.95,
            },
        ]
        job = {
            "id": "job-1",
            "hyperparameters": {"learning_rate": 0.01, "epochs": 3},
            "total_rounds": 3,
        }
        completed_rounds = [
            {"id": "r0", "round_number": 0, "status": TrainingRoundStatus.COMPLETED.value, "global_loss": 0.5, "global_accuracy": 0.8},
            {"id": "r1", "round_number": 1, "status": TrainingRoundStatus.COMPLETED.value, "global_loss": 0.3, "global_accuracy": 0.9},
            {"id": "r2", "round_number": 2, "status": TrainingRoundStatus.COMPLETED.value, "global_loss": 0.2, "global_accuracy": 0.95},
        ]
        tasks = [
            {"id": "task-1", "status": TaskStatus.RUNNING.value},
        ]

        call_count = {"select": 0}

        def mock_select(table, columns="*", filters=None):
            if table == "training_rounds":
                if filters and filters.get("round_number") == 2:
                    return [round_record]
                # For _build_aggregated_metrics inside complete_job
                return completed_rounds
            if table == "gradient_submissions":
                return submissions
            if table == "jobs":
                return [job]
            if table == "tasks":
                return tasks
            if table == "metrics":
                return []
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps, \
             patch("coordinator.aggregator.download_blob") as mock_dl, \
             patch("coordinator.aggregator.delete_blob"), \
             patch("coordinator.aggregator.barrier_mod") as mock_barrier:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_db.insert.return_value = {}
            mock_dl.return_value = _serialize(grad)
            mock_ps.get_parameters.return_value = _serialize(initial_params)
            mock_ps.store_checkpoint.return_value = "checkpoints/job-1/round_2.pt"

            aggregate_round("job-1", 2)

            # complete_job should have been called, which stores checkpoint
            mock_ps.store_checkpoint.assert_called_once_with("job-1", 2)

            # Job should be marked completed
            job_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "jobs" and c[0][1].get("status") == JobStatus.COMPLETED.value
            ]
            assert len(job_update_calls) == 1

            # create_round should NOT have been called (no next round)
            mock_barrier.create_round.assert_not_called()


# ---------------------------------------------------------------------------
# complete_job
# ---------------------------------------------------------------------------


class TestCompleteJob:
    """Tests for complete_job — final checkpoint and aggregated metrics."""

    def test_stores_final_checkpoint_and_aggregated_metrics(self):
        """complete_job stores the final checkpoint via param_server and
        builds aggregated_metrics from training_rounds.
        Requirements: 6.1, 6.4
        """
        job = {
            "id": "job-1",
            "hyperparameters": {"learning_rate": 0.01, "epochs": 3},
            "total_rounds": 3,
        }
        completed_rounds = [
            {
                "id": "r0",
                "round_number": 0,
                "status": TrainingRoundStatus.COMPLETED.value,
                "global_loss": 0.5,
                "global_accuracy": 0.8,
            },
            {
                "id": "r1",
                "round_number": 1,
                "status": TrainingRoundStatus.COMPLETED.value,
                "global_loss": 0.3,
                "global_accuracy": 0.9,
            },
            {
                "id": "r2",
                "round_number": 2,
                "status": TrainingRoundStatus.COMPLETED.value,
                "global_loss": 0.2,
                "global_accuracy": 0.95,
            },
        ]
        tasks = [
            {"id": "task-1", "status": TaskStatus.RUNNING.value},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "jobs":
                return [job]
            if table == "training_rounds":
                return completed_rounds
            if table == "tasks":
                return tasks
            if table == "metrics":
                return []
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_ps.store_checkpoint.return_value = "checkpoints/job-1/round_2.pt"

            complete_job("job-1")

            # Verify checkpoint stored for final round (round 2)
            mock_ps.store_checkpoint.assert_called_once_with("job-1", 2)

            # Verify job marked completed with aggregated_metrics
            job_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "jobs"
            ]
            assert len(job_update_calls) >= 1
            job_data = job_update_calls[0][0][1]
            assert job_data["status"] == JobStatus.COMPLETED.value
            assert "aggregated_metrics" in job_data
            assert job_data["global_model_path"] == "checkpoints/job-1/round_2.pt"
            assert "completed_at" in job_data

    def test_complete_job_marks_active_tasks_completed(self):
        """complete_job marks all remaining active (assigned/running) tasks
        as completed.
        Requirements: 6.1
        """
        job = {
            "id": "job-1",
            "hyperparameters": {"learning_rate": 0.01, "epochs": 2},
            "total_rounds": 2,
        }
        completed_rounds = [
            {"id": "r0", "round_number": 0, "status": TrainingRoundStatus.COMPLETED.value, "global_loss": 0.5, "global_accuracy": 0.8},
            {"id": "r1", "round_number": 1, "status": TrainingRoundStatus.COMPLETED.value, "global_loss": 0.3, "global_accuracy": 0.9},
        ]
        tasks = [
            {"id": "task-1", "status": TaskStatus.RUNNING.value},
            {"id": "task-2", "status": TaskStatus.ASSIGNED.value},
            {"id": "task-3", "status": TaskStatus.FAILED.value},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "jobs":
                return [job]
            if table == "training_rounds":
                return completed_rounds
            if table == "tasks":
                return tasks
            if table == "metrics":
                return []
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_ps.store_checkpoint.return_value = "checkpoints/job-1/round_1.pt"

            complete_job("job-1")

            # Find task update calls that set status to completed
            task_completed_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks" and c[0][1].get("status") == TaskStatus.COMPLETED.value
            ]
            # task-1 (running) and task-2 (assigned) should be completed
            # task-3 (failed) should NOT be touched
            assert len(task_completed_calls) == 2
            completed_task_ids = {
                c[1]["filters"]["id"] for c in task_completed_calls
            }
            assert completed_task_ids == {"task-1", "task-2"}

            # Each completed task should have completed_at set
            for c in task_completed_calls:
                assert "completed_at" in c[0][1]


# ---------------------------------------------------------------------------
# check_job_failure
# ---------------------------------------------------------------------------


class TestCheckJobFailure:
    """Tests for check_job_failure."""

    def test_marks_job_failed_when_all_tasks_done_with_failures(self):
        """Job is marked failed when some tasks failed and none are active."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.COMPLETED.value, "shard_index": 0, "node_id": "n1", "error_message": None},
            {"id": "t2", "status": TaskStatus.FAILED.value, "shard_index": 1, "node_id": "n2", "error_message": "OOM"},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "tasks":
                return all_tasks
            if table == "training_rounds":
                return []  # No completed rounds
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            check_job_failure("job-1")

            mock_db.update.assert_called_once()
            data = mock_db.update.call_args[0][1]
            assert data["status"] == JobStatus.FAILED.value
            assert "error_summary" in data
            assert len(data["error_summary"]["failed_tasks"]) == 1
            assert data["error_summary"]["failed_tasks"][0]["error_message"] == "OOM"

    def test_does_not_mark_failed_when_tasks_still_active(self):
        """Job is NOT marked failed when some tasks are still running."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.RUNNING.value, "shard_index": 0, "node_id": "n1"},
            {"id": "t2", "status": TaskStatus.FAILED.value, "shard_index": 1, "node_id": "n2", "error_message": "OOM"},
        ]

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.return_value = all_tasks

            check_job_failure("job-1")

            mock_db.update.assert_not_called()

    def test_does_not_mark_failed_when_no_failures(self):
        """Job is NOT marked failed when all tasks completed successfully."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.COMPLETED.value, "shard_index": 0, "node_id": "n1"},
            {"id": "t2", "status": TaskStatus.COMPLETED.value, "shard_index": 1, "node_id": "n2"},
        ]

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.return_value = all_tasks

            check_job_failure("job-1")

            mock_db.update.assert_not_called()

    def test_stores_partial_checkpoint_when_rounds_completed(self):
        """When job fails with completed rounds, a partial checkpoint is stored."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.FAILED.value, "shard_index": 0, "node_id": "n1", "error_message": "OOM"},
        ]
        completed_rounds = [
            {"id": "r1", "round_number": 0, "status": TrainingRoundStatus.COMPLETED.value},
            {"id": "r2", "round_number": 1, "status": TrainingRoundStatus.COMPLETED.value},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "tasks":
                return all_tasks
            if table == "training_rounds":
                return completed_rounds
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_ps.store_checkpoint.return_value = "checkpoints/job-1/round_1.pt"

            check_job_failure("job-1")

            # Should store checkpoint for the last completed round (round 1)
            mock_ps.store_checkpoint.assert_called_once_with("job-1", 1)

            # Job update should include the checkpoint path
            data = mock_db.update.call_args[0][1]
            assert data["status"] == JobStatus.FAILED.value
            assert data["global_model_path"] == "checkpoints/job-1/round_1.pt"

    def test_no_partial_checkpoint_when_no_rounds_completed(self):
        """When job fails with no completed rounds, no checkpoint is stored."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.FAILED.value, "shard_index": 0, "node_id": "n1", "error_message": "crash"},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "tasks":
                return all_tasks
            if table == "training_rounds":
                return []  # No completed rounds
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            check_job_failure("job-1")

            mock_ps.store_checkpoint.assert_not_called()
            data = mock_db.update.call_args[0][1]
            assert data["status"] == JobStatus.FAILED.value
            assert "global_model_path" not in data

    def test_does_nothing_when_no_tasks(self):
        """Does nothing when there are no tasks for the job."""
        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.return_value = []

            check_job_failure("job-1")

            mock_db.update.assert_not_called()

    def test_handles_checkpoint_store_failure_gracefully(self):
        """Job is still marked failed even if checkpoint storage fails."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.FAILED.value, "shard_index": 0, "node_id": "n1", "error_message": "OOM"},
        ]
        completed_rounds = [
            {"id": "r1", "round_number": 0, "status": TrainingRoundStatus.COMPLETED.value},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "tasks":
                return all_tasks
            if table == "training_rounds":
                return completed_rounds
            return []

        with patch("coordinator.aggregator.db") as mock_db, \
             patch("coordinator.aggregator.param_server") as mock_ps:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []
            mock_ps.store_checkpoint.side_effect = Exception("Storage unavailable")

            check_job_failure("job-1")

            # Job should still be marked failed
            data = mock_db.update.call_args[0][1]
            assert data["status"] == JobStatus.FAILED.value
            assert "global_model_path" not in data
