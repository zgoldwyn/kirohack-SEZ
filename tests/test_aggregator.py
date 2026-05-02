"""Unit tests for coordinator/aggregator.py — metrics aggregation and job failure detection."""

from __future__ import annotations

from unittest.mock import patch, call

import pytest

from coordinator.aggregator import aggregate_job_metrics, check_job_failure
from coordinator.constants import JobStatus, TaskStatus


# ---------------------------------------------------------------------------
# aggregate_job_metrics
# ---------------------------------------------------------------------------


class TestAggregateJobMetrics:
    """Tests for aggregate_job_metrics."""

    def test_aggregates_mean_loss_and_accuracy(self):
        """Computes correct mean loss and accuracy from final epoch metrics."""
        tasks = [
            {"id": "t1", "node_id": "n1", "status": TaskStatus.COMPLETED.value},
            {"id": "t2", "node_id": "n2", "status": TaskStatus.COMPLETED.value},
        ]
        # Task t1: epochs 0,1,2 — final epoch is 2
        t1_metrics = [
            {"epoch": 0, "loss": 1.0, "accuracy": 0.5},
            {"epoch": 1, "loss": 0.5, "accuracy": 0.7},
            {"epoch": 2, "loss": 0.2, "accuracy": 0.9},
        ]
        # Task t2: epochs 0,1 — final epoch is 1
        t2_metrics = [
            {"epoch": 0, "loss": 0.8, "accuracy": 0.6},
            {"epoch": 1, "loss": 0.4, "accuracy": 0.8},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "tasks":
                return tasks
            if table == "metrics":
                task_id = filters.get("task_id")
                if task_id == "t1":
                    return t1_metrics
                if task_id == "t2":
                    return t2_metrics
            return []

        with patch("coordinator.aggregator.db") as mock_db:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            aggregate_job_metrics("job-1")

            # Verify the update call
            mock_db.update.assert_called_once()
            args = mock_db.update.call_args[0]
            assert args[0] == "jobs"
            data = args[1]
            assert data["status"] == JobStatus.COMPLETED.value
            assert "completed_at" in data

            agg = data["aggregated_metrics"]
            # mean_loss = (0.2 + 0.4) / 2 = 0.3
            assert abs(agg["mean_loss"] - 0.3) < 1e-9
            # mean_accuracy = (0.9 + 0.8) / 2 = 0.85
            assert abs(agg["mean_accuracy"] - 0.85) < 1e-9
            assert len(agg["per_node"]) == 2

    def test_handles_no_metrics(self):
        """Tasks with no metrics get None values in breakdown."""
        tasks = [
            {"id": "t1", "node_id": "n1", "status": TaskStatus.COMPLETED.value},
        ]

        def mock_select(table, columns="*", filters=None):
            if table == "tasks":
                return tasks
            if table == "metrics":
                return []
            return []

        with patch("coordinator.aggregator.db") as mock_db:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            aggregate_job_metrics("job-1")

            data = mock_db.update.call_args[0][1]
            agg = data["aggregated_metrics"]
            assert agg["mean_loss"] is None
            assert agg["mean_accuracy"] is None
            assert agg["per_node"][0]["loss"] is None


# ---------------------------------------------------------------------------
# check_job_failure
# ---------------------------------------------------------------------------


class TestCheckJobFailure:
    """Tests for check_job_failure."""

    def test_marks_job_failed_when_all_tasks_done_with_failures(self):
        """Job is marked failed when some tasks failed and none are active."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.COMPLETED.value, "shard_index": 0, "error_message": None},
            {"id": "t2", "status": TaskStatus.FAILED.value, "shard_index": 1, "error_message": "OOM"},
        ]

        with patch("coordinator.aggregator.db") as mock_db:
            mock_db.select.return_value = all_tasks
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
            {"id": "t1", "status": TaskStatus.RUNNING.value, "shard_index": 0},
            {"id": "t2", "status": TaskStatus.FAILED.value, "shard_index": 1, "error_message": "OOM"},
        ]

        with patch("coordinator.aggregator.db") as mock_db:
            mock_db.select.return_value = all_tasks

            check_job_failure("job-1")

            mock_db.update.assert_not_called()

    def test_does_not_mark_failed_when_no_failures(self):
        """Job is NOT marked failed when all tasks completed successfully."""
        all_tasks = [
            {"id": "t1", "status": TaskStatus.COMPLETED.value, "shard_index": 0},
            {"id": "t2", "status": TaskStatus.COMPLETED.value, "shard_index": 1},
        ]

        with patch("coordinator.aggregator.db") as mock_db:
            mock_db.select.return_value = all_tasks

            check_job_failure("job-1")

            mock_db.update.assert_not_called()
