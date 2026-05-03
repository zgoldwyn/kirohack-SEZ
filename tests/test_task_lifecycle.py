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

from coordinator.constants import NodeStatus, TaskStatus
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
        """Assigned task transitions to running with started_at set."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=SAMPLE_TASK_ASSIGNED,
            )
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

