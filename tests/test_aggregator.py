"""Unit tests for coordinator/aggregator.py — job failure detection.

Tests the check_job_failure function which now:
- Queries tasks for the job
- If has_failed and not has_active: marks job failed
- Queries training_rounds for completed rounds to store partial checkpoint
  via param_server.store_checkpoint()
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from coordinator.aggregator import check_job_failure
from coordinator.constants import JobStatus, TaskStatus, TrainingRoundStatus


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
