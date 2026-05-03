"""Unit tests for GET /api/tasks/poll (task 11.1).

Tests use FastAPI's TestClient with monkeypatched db and
config_parser functions so we don't need a live Supabase instance.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from coordinator.constants import JobStatus, NodeStatus, TaskStatus
from coordinator.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AUTH_TOKEN = "test-token-poll-123"
AUTH_TOKEN_HASH = hashlib.sha256(AUTH_TOKEN.encode()).hexdigest()

IDLE_NODE = {
    "id": "node-uuid-1",
    "node_id": "worker-1",
    "hostname": "host1",
    "status": NodeStatus.IDLE.value,
    "auth_token_hash": AUTH_TOKEN_HASH,
    "cpu_cores": 4,
    "ram_mb": 8192,
    "gpu_model": None,
    "disk_mb": 50000,
}

BUSY_NODE = {
    **IDLE_NODE,
    "status": NodeStatus.BUSY.value,
}

IDLE_NODE_WITH_GPU = {
    **IDLE_NODE,
    "gpu_model": "NVIDIA RTX 3080",
    "vram_mb": 10240,
}

LOW_RAM_NODE = {
    **IDLE_NODE,
    "ram_mb": 256,  # Below MLP minimum of 512
}

SAMPLE_TASK_CONFIG = {
    "task_id": "task-uuid-1",
    "job_id": "job-uuid-1",
    "dataset_name": "MNIST",
    "model_type": "MLP",
    "hyperparameters": {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "hidden_layers": [128, 64],
        "activation": "relu",
    },
    "shard_index": 0,
    "shard_count": 2,
    "total_rounds": 10,
}

QUEUED_TASK = {
    "id": "task-uuid-1",
    "job_id": "job-uuid-1",
    "node_id": None,
    "shard_index": 0,
    "status": TaskStatus.QUEUED.value,
    "task_config": SAMPLE_TASK_CONFIG,
    "checkpoint_path": None,
    "error_message": None,
}

QUEUED_JOB = {
    "id": "job-uuid-1",
    "status": JobStatus.QUEUED.value,
    "model_type": "MLP",
    "dataset_name": "MNIST",
}

RUNNING_JOB = {
    **QUEUED_JOB,
    "status": JobStatus.RUNNING.value,
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
    queued_tasks: list[dict] | None = None,
    job: dict | None = None,
):
    """Build a side_effect for db.select that returns the right data
    depending on the table and filters."""

    def _select(table: str, columns: str = "*", filters: dict | None = None):
        if table == "nodes":
            if filters and "auth_token_hash" in filters:
                return [node] if node else []
            return [node] if node else []
        if table == "tasks":
            if filters and filters.get("status") == TaskStatus.QUEUED.value:
                return queued_tasks if queued_tasks is not None else []
            return []
        if table == "jobs":
            return [job] if job else []
        return []

    return _select


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPollForTask:
    """Tests for GET /api/tasks/poll."""

    def test_poll_assigns_eligible_task(self, client):
        """An idle node receives an eligible queued task."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK],
                job=QUEUED_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())

            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] == "task-uuid-1"
            assert data["job_id"] == "job-uuid-1"
            assert data["dataset_name"] == "MNIST"
            assert data["model_type"] == "MLP"
            assert data["shard_index"] == 0
            assert data["shard_count"] == 2
            assert data["hyperparameters"] is not None

    def test_poll_updates_task_to_assigned(self, client):
        """Task status is updated to 'assigned' with node_id and assigned_at."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK],
                job=QUEUED_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())
            assert resp.status_code == 200

            # Find the task update call
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 1
            task_data = task_update_calls[0][0][1]
            assert task_data["status"] == TaskStatus.ASSIGNED.value
            assert task_data["node_id"] == "node-uuid-1"
            assert "assigned_at" in task_data

    def test_poll_sets_node_busy(self, client):
        """Node status is updated to 'busy' after task assignment."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK],
                job=QUEUED_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())
            assert resp.status_code == 200

            # Find the node update call
            node_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "nodes"
            ]
            assert len(node_update_calls) == 1
            assert node_update_calls[0][0][1]["status"] == NodeStatus.BUSY.value

    def test_poll_sets_job_running_on_first_assignment(self, client):
        """Job status transitions from 'queued' to 'running' on first task assignment."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK],
                job=QUEUED_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())
            assert resp.status_code == 200

            # Find the job update call
            job_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "jobs"
            ]
            assert len(job_update_calls) == 1
            job_data = job_update_calls[0][0][1]
            assert job_data["status"] == JobStatus.RUNNING.value
            assert "started_at" in job_data

    def test_poll_does_not_update_already_running_job(self, client):
        """Job already in 'running' status is not updated again."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK],
                job=RUNNING_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())
            assert resp.status_code == 200

            # Should NOT have a job update call
            job_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "jobs"
            ]
            assert len(job_update_calls) == 0

    def test_poll_busy_node_gets_empty_response(self, client):
        """A busy node receives an empty response (no task assigned)."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [BUSY_NODE]

            resp = client.get("/api/tasks/poll", headers=auth_headers())

            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] is None
            assert data["job_id"] is None

    def test_poll_no_queued_tasks_returns_empty(self, client):
        """When no tasks are queued, returns empty response."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[],
            )

            resp = client.get("/api/tasks/poll", headers=auth_headers())

            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] is None

    def test_poll_insufficient_ram_skips_task(self, client):
        """A node with insufficient RAM does not get assigned the task."""
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [LOW_RAM_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=LOW_RAM_NODE,
                queued_tasks=[QUEUED_TASK],
            )

            resp = client.get("/api/tasks/poll", headers=auth_headers())

            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] is None

    def test_poll_no_auth_returns_401(self, client):
        """Returns 401 when no auth header is provided."""
        resp = client.get("/api/tasks/poll")
        assert resp.status_code == 401

    def test_poll_invalid_auth_returns_401(self, client):
        """Returns 401 when auth token is invalid."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = []

            resp = client.get(
                "/api/tasks/poll",
                headers={"Authorization": "Bearer bad-token"},
            )

            assert resp.status_code == 401

    def test_poll_selects_first_eligible_task(self, client):
        """When multiple queued tasks exist, the first eligible one is assigned."""
        task2 = {
            **QUEUED_TASK,
            "id": "task-uuid-2",
            "shard_index": 1,
            "task_config": {**SAMPLE_TASK_CONFIG, "task_id": "task-uuid-2", "shard_index": 1},
        }

        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK, task2],
                job=QUEUED_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())

            assert resp.status_code == 200
            data = resp.json()
            # Should get the first task
            assert data["task_id"] == "task-uuid-1"
            assert data["shard_index"] == 0


    def test_poll_response_includes_total_rounds(self, client):
        """Poll response includes total_rounds field from task_config.
        Requirements: 4.4
        """
        with patch("coordinator.scheduler.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [IDLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                node=IDLE_NODE,
                queued_tasks=[QUEUED_TASK],
                job=QUEUED_JOB,
            )
            mock_db.update.return_value = []

            resp = client.get("/api/tasks/poll", headers=auth_headers())

            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] == "task-uuid-1"
            assert data["total_rounds"] == 10
