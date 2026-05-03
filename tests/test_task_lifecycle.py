"""Unit tests for task lifecycle endpoints (12.1–12.4).

Tests use FastAPI's TestClient with monkeypatched db and storage
functions so we don't need a live Supabase instance.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from coordinator.constants import JobStatus, NodeStatus, TaskStatus
from coordinator.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AUTH_TOKEN = "test-token-abc123"
AUTH_TOKEN_HASH = hashlib.sha256(AUTH_TOKEN.encode()).hexdigest()

SAMPLE_NODE = {
    "id": "node-uuid-1",
    "node_id": "worker-1",
    "hostname": "host1",
    "status": NodeStatus.BUSY.value,
    "auth_token_hash": AUTH_TOKEN_HASH,
    "cpu_cores": 4,
    "ram_mb": 8192,
    "gpu_model": None,
}

SAMPLE_TASK_ASSIGNED = {
    "id": "task-uuid-1",
    "job_id": "job-uuid-1",
    "node_id": "node-uuid-1",
    "shard_index": 0,
    "status": TaskStatus.ASSIGNED.value,
    "task_config": {},
    "checkpoint_path": None,
    "error_message": None,
}

SAMPLE_TASK_RUNNING = {
    **SAMPLE_TASK_ASSIGNED,
    "status": TaskStatus.RUNNING.value,
}


@pytest.fixture
def client():
    return TestClient(app)


def auth_headers(token: str = AUTH_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Helpers for monkeypatching db
# ---------------------------------------------------------------------------


def make_select_side_effect(
    node: dict | None = None,
    task: dict | None = None,
    all_tasks: list[dict] | None = None,
):
    """Build a side_effect function for db.select that returns the right
    data depending on the table and filters."""

    def _select(table: str, columns: str = "*", filters: dict | None = None):
        if table == "nodes":
            return [node] if node else []
        if table == "tasks":
            if filters and "id" in filters:
                return [task] if task else []
            if filters and "job_id" in filters:
                return all_tasks if all_tasks is not None else []
        if table == "metrics":
            return []
        return []

    return _select


# ---------------------------------------------------------------------------
# 12.1 POST /api/tasks/{id}/start
# ---------------------------------------------------------------------------


class TestStartTask:
    """Tests for POST /api/tasks/{task_id}/start."""

    def test_start_task_success(self, client):
        """Assigned task transitions to running with started_at set and round 0 ensured."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            def _select(table, columns="*", filters=None):
                if table == "tasks":
                    return [SAMPLE_TASK_ASSIGNED]
                if table == "training_rounds":
                    # Round 0 already exists
                    return [{"id": "round-0", "round_number": 0}]
                return []

            mock_db.select.side_effect = _select
            mock_db.update.return_value = [SAMPLE_TASK_ASSIGNED]

            resp = client.post(
                f"/api/tasks/{SAMPLE_TASK_ASSIGNED['id']}/start",
                headers=auth_headers(),
            )

            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            # Verify update was called with correct args
            mock_db.update.assert_called_once()
            call_args = mock_db.update.call_args
            assert call_args[0][0] == "tasks"
            assert call_args[0][1]["status"] == TaskStatus.RUNNING.value
            assert "started_at" in call_args[0][1]

            # Round 0 already existed, so insert should NOT have been called
            mock_db.insert.assert_not_called()

    def test_start_task_creates_round_0_if_missing(self, client):
        """When no training_rounds record exists for round 0, one is created."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db, \
             patch("coordinator.barrier.db") as mock_barrier_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            active_task = {
                **SAMPLE_TASK_ASSIGNED,
                "status": TaskStatus.RUNNING.value,
            }

            def _select(table, columns="*", filters=None):
                if table == "tasks":
                    if filters and "id" in filters:
                        return [SAMPLE_TASK_ASSIGNED]
                    # get_active_workers queries tasks by job_id
                    if filters and "job_id" in filters:
                        return [active_task]
                    return []
                if table == "training_rounds":
                    # No round 0 exists yet
                    return []
                return []

            mock_db.select.side_effect = _select
            # barrier.py uses its own db import for get_active_workers
            mock_barrier_db.select.side_effect = _select
            mock_barrier_db.insert.return_value = {"id": "new-round-0", "round_number": 0}
            mock_db.update.return_value = [SAMPLE_TASK_ASSIGNED]
            mock_db.insert.return_value = {"id": "new-round-0", "round_number": 0}

            resp = client.post(
                f"/api/tasks/{SAMPLE_TASK_ASSIGNED['id']}/start",
                headers=auth_headers(),
            )

            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            # Verify round 0 was created via barrier.db.insert (create_round uses barrier's db)
            mock_barrier_db.insert.assert_called_once()
            insert_args = mock_barrier_db.insert.call_args
            assert insert_args[0][0] == "training_rounds"
            insert_data = insert_args[0][1]
            assert insert_data["job_id"] == "job-uuid-1"
            assert insert_data["round_number"] == 0
            assert insert_data["status"] == "in_progress"
            assert insert_data["submitted_count"] == 0

    def test_start_task_not_found(self, client):
        """Returns 404 when task does not exist."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.return_value = []

            resp = client.post(
                "/api/tasks/nonexistent-id/start",
                headers=auth_headers(),
            )

            assert resp.status_code == 404

    def test_start_task_wrong_node(self, client):
        """Returns 403 when task is assigned to a different node."""
        other_task = {**SAMPLE_TASK_ASSIGNED, "node_id": "other-node-uuid"}
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=other_task,
            )

            resp = client.post(
                f"/api/tasks/{SAMPLE_TASK_ASSIGNED['id']}/start",
                headers=auth_headers(),
            )

            assert resp.status_code == 403

    def test_start_task_wrong_status(self, client):
        """Returns 409 when task is not in 'assigned' status."""
        running_task = {**SAMPLE_TASK_ASSIGNED, "status": TaskStatus.RUNNING.value}
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=running_task,
            )

            resp = client.post(
                f"/api/tasks/{SAMPLE_TASK_ASSIGNED['id']}/start",
                headers=auth_headers(),
            )

            assert resp.status_code == 409

    def test_start_task_no_auth(self, client):
        """Returns 401 when no auth header is provided."""
        resp = client.post(f"/api/tasks/{SAMPLE_TASK_ASSIGNED['id']}/start")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 12.3 POST /api/tasks/{id}/fail
# ---------------------------------------------------------------------------


class TestFailTask:
    """Tests for POST /api/tasks/{task_id}/fail."""

    def test_fail_task_success(self, client):
        """Task is marked failed with error message and node set to idle."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db, \
             patch("coordinator.main.check_job_failure") as mock_fail:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=SAMPLE_TASK_RUNNING,
            )
            mock_db.update.return_value = []

            resp = client.post(
                f"/api/tasks/{SAMPLE_TASK_RUNNING['id']}/fail",
                headers=auth_headers(),
                json={"error_message": "OOM during training"},
            )

            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            # Verify task update
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 1
            task_data = task_update_calls[0][0][1]
            assert task_data["status"] == TaskStatus.FAILED.value
            assert task_data["error_message"] == "OOM during training"
            assert "completed_at" in task_data

            # Verify node set to idle
            node_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "nodes"
            ]
            assert len(node_update_calls) == 1
            assert node_update_calls[0][0][1]["status"] == NodeStatus.IDLE.value

            # Verify job failure check
            mock_fail.assert_called_once_with("job-uuid-1")

    def test_fail_task_not_found(self, client):
        """Returns 404 when task does not exist."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.return_value = []

            resp = client.post(
                "/api/tasks/nonexistent/fail",
                headers=auth_headers(),
                json={"error_message": "error"},
            )

            assert resp.status_code == 404

    def test_fail_task_wrong_node(self, client):
        """Returns 403 when task belongs to a different node."""
        other_task = {**SAMPLE_TASK_RUNNING, "node_id": "other-node"}
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(task=other_task)

            resp = client.post(
                f"/api/tasks/{SAMPLE_TASK_RUNNING['id']}/fail",
                headers=auth_headers(),
                json={"error_message": "error"},
            )

            assert resp.status_code == 403



# ---------------------------------------------------------------------------
# Gradient submission: POST /api/jobs/{job_id}/gradients
# ---------------------------------------------------------------------------


SAMPLE_JOB_RUNNING = {
    "id": "job-uuid-1",
    "status": JobStatus.RUNNING.value,
    "current_round": 0,
    "total_rounds": 5,
    "hyperparameters": {"learning_rate": 0.01, "epochs": 5},
}


class TestGradientSubmission:
    """Tests for POST /api/jobs/{job_id}/gradients — collaborative training flow.
    Requirements: 5.1, 5.2
    """

    def test_gradient_submission_records_and_checks_barrier(self, client):
        """Submitting gradients stores the payload, records the submission
        via barrier, and checks if the barrier is met.
        """
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db, \
             patch("coordinator.main.upload_blob") as mock_upload, \
             patch("coordinator.main.record_submission") as mock_record, \
             patch("coordinator.main.check_barrier") as mock_barrier, \
             patch("coordinator.main.aggregate_round") as mock_agg:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            def _select(table, columns="*", filters=None):
                if table == "tasks":
                    if filters and filters.get("job_id") and filters.get("node_id"):
                        return [SAMPLE_TASK_RUNNING]
                    if filters and "id" in filters:
                        return [SAMPLE_TASK_RUNNING]
                    return []
                if table == "jobs":
                    return [SAMPLE_JOB_RUNNING]
                if table == "gradient_submissions":
                    return []  # No duplicate
                return []

            mock_db.select.side_effect = _select
            mock_db.update.return_value = []
            mock_db.insert.return_value = {}
            mock_record.return_value = {"id": "sub-1"}
            mock_barrier.return_value = False  # Barrier not yet met

            resp = client.post(
                f"/api/jobs/job-uuid-1/gradients?round_number=0&task_id=task-uuid-1&local_loss=0.5&local_accuracy=0.8",
                headers=auth_headers(),
                content=b"fake-gradient-data",
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["barrier_met"] is False

            # Verify gradient was uploaded to storage
            mock_upload.assert_called_once()

            # Verify submission was recorded via barrier
            mock_record.assert_called_once_with(
                job_id="job-uuid-1",
                round_number=0,
                task_id="task-uuid-1",
                node_id="node-uuid-1",
            )

            # Barrier not met, so aggregation should NOT be triggered
            mock_agg.assert_not_called()

    def test_gradient_submission_triggers_aggregation_when_barrier_met(self, client):
        """When the barrier is met after gradient submission, aggregation
        is triggered for the current round.
        """
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db, \
             patch("coordinator.main.upload_blob"), \
             patch("coordinator.main.record_submission") as mock_record, \
             patch("coordinator.main.check_barrier") as mock_barrier, \
             patch("coordinator.main.aggregate_round") as mock_agg:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            def _select(table, columns="*", filters=None):
                if table == "tasks":
                    if filters and filters.get("job_id") and filters.get("node_id"):
                        return [SAMPLE_TASK_RUNNING]
                    return []
                if table == "jobs":
                    return [SAMPLE_JOB_RUNNING]
                if table == "gradient_submissions":
                    return []
                return []

            mock_db.select.side_effect = _select
            mock_db.update.return_value = []
            mock_db.insert.return_value = {}
            mock_record.return_value = {"id": "sub-1"}
            mock_barrier.return_value = True  # Barrier met!

            resp = client.post(
                f"/api/jobs/job-uuid-1/gradients?round_number=0&task_id=task-uuid-1",
                headers=auth_headers(),
                content=b"fake-gradient-data",
            )

            assert resp.status_code == 200
            assert resp.json()["barrier_met"] is True

            # Aggregation should be triggered
            mock_agg.assert_called_once_with("job-uuid-1", 0)

    def test_gradient_submission_wrong_round_returns_409(self, client):
        """Submitting gradients for a round that doesn't match the job's
        current_round returns 409.
        Requirements: 13.3
        """
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            def _select(table, columns="*", filters=None):
                if table == "tasks":
                    if filters and filters.get("job_id") and filters.get("node_id"):
                        return [SAMPLE_TASK_RUNNING]
                    return []
                if table == "jobs":
                    return [SAMPLE_JOB_RUNNING]  # current_round=0
                return []

            mock_db.select.side_effect = _select

            resp = client.post(
                f"/api/jobs/job-uuid-1/gradients?round_number=5&task_id=task-uuid-1",
                headers=auth_headers(),
                content=b"fake-gradient-data",
            )

            assert resp.status_code == 409

    def test_gradient_submission_duplicate_returns_409(self, client):
        """Submitting gradients twice for the same round returns 409.
        Requirements: 13.2
        """
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            def _select(table, columns="*", filters=None):
                if table == "tasks":
                    if filters and filters.get("job_id") and filters.get("node_id"):
                        return [SAMPLE_TASK_RUNNING]
                    return []
                if table == "jobs":
                    return [SAMPLE_JOB_RUNNING]  # current_round=0
                if table == "gradient_submissions":
                    # Already submitted for this round
                    return [{"id": "existing-sub", "round_number": 0}]
                return []

            mock_db.select.side_effect = _select

            resp = client.post(
                f"/api/jobs/job-uuid-1/gradients?round_number=0&task_id=task-uuid-1",
                headers=auth_headers(),
                content=b"fake-gradient-data",
            )

            assert resp.status_code == 409
            assert "already submitted" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Removed endpoints return 404/405
# ---------------------------------------------------------------------------


class TestRemovedEndpoints:
    """Verify that removed endpoints (from the independent training model)
    return 404 or 405.
    Requirements: 5.1, 5.2
    """

    def test_complete_endpoint_not_found(self, client):
        """POST /api/tasks/{id}/complete should return 404 or 405."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            resp = client.post(
                "/api/tasks/task-uuid-1/complete",
                headers=auth_headers(),
            )

            # The endpoint doesn't exist, so FastAPI returns 404 or 405
            assert resp.status_code in (404, 405)

    def test_upload_url_endpoint_not_found(self, client):
        """POST /api/tasks/{id}/upload-url should return 404 or 405."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            resp = client.post(
                "/api/tasks/task-uuid-1/upload-url",
                headers=auth_headers(),
            )

            assert resp.status_code in (404, 405)
