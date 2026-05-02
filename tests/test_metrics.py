"""Unit tests for the POST /api/metrics endpoint (task 13.1).

Tests use FastAPI's TestClient with monkeypatched db and auth
functions so we don't need a live Supabase instance.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from coordinator.constants import TaskStatus
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
    "status": "busy",
    "auth_token_hash": AUTH_TOKEN_HASH,
    "cpu_cores": 4,
    "ram_mb": 8192,
    "gpu_model": None,
}

SAMPLE_TASK_RUNNING = {
    "id": "task-uuid-1",
    "job_id": "job-uuid-1",
    "node_id": "node-uuid-1",
    "shard_index": 0,
    "status": TaskStatus.RUNNING.value,
    "task_config": {},
    "checkpoint_path": None,
    "error_message": None,
}


@pytest.fixture
def client():
    return TestClient(app)


def auth_headers(token: str = AUTH_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def make_select_side_effect(
    node: dict | None = None,
    task: dict | None = None,
):
    """Build a side_effect for db.select that returns the right data
    depending on the table and filters."""

    def _select(table: str, columns: str = "*", filters: dict | None = None):
        if table == "nodes":
            return [node] if node else []
        if table == "tasks":
            if filters and "id" in filters:
                return [task] if task else []
        return []

    return _select


# ---------------------------------------------------------------------------
# 13.1 POST /api/metrics
# ---------------------------------------------------------------------------


class TestReportMetrics:
    """Tests for POST /api/metrics."""

    def test_report_metrics_success(self, client):
        """Valid metrics report inserts a record and returns ok."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=SAMPLE_TASK_RUNNING,
            )
            mock_db.insert.return_value = {}

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "task_id": "task-uuid-1",
                    "epoch": 3,
                    "loss": 0.25,
                    "accuracy": 0.92,
                },
            )

            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            # Verify insert was called with correct data
            mock_db.insert.assert_called_once()
            call_args = mock_db.insert.call_args
            assert call_args[0][0] == "metrics"
            data = call_args[0][1]
            assert data["job_id"] == "job-uuid-1"
            assert data["task_id"] == "task-uuid-1"
            assert data["node_id"] == "node-uuid-1"
            assert data["epoch"] == 3
            assert data["loss"] == 0.25
            assert data["accuracy"] == 0.92

    def test_report_metrics_with_nulls(self, client):
        """Metrics with null loss and accuracy are accepted."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=SAMPLE_TASK_RUNNING,
            )
            mock_db.insert.return_value = {}

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "task_id": "task-uuid-1",
                    "epoch": 0,
                },
            )

            assert resp.status_code == 200
            data = mock_db.insert.call_args[0][1]
            assert data["loss"] is None
            assert data["accuracy"] is None

    def test_report_metrics_task_not_found(self, client):
        """Returns 404 when the referenced task does not exist."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(task=None)

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "task_id": "nonexistent-task",
                    "epoch": 0,
                    "loss": 0.5,
                },
            )

            assert resp.status_code == 404

    def test_report_metrics_wrong_node(self, client):
        """Returns 403 when the task belongs to a different node."""
        other_task = {**SAMPLE_TASK_RUNNING, "node_id": "other-node-uuid"}
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]
            mock_db.select.side_effect = make_select_side_effect(
                task=other_task,
            )

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "task_id": "task-uuid-1",
                    "epoch": 1,
                    "loss": 0.3,
                },
            )

            assert resp.status_code == 403

    def test_report_metrics_no_auth(self, client):
        """Returns 401 when no auth header is provided."""
        resp = client.post(
            "/api/metrics",
            json={
                "task_id": "task-uuid-1",
                "epoch": 0,
                "loss": 0.5,
            },
        )

        assert resp.status_code == 401

    def test_report_metrics_invalid_auth(self, client):
        """Returns 401 when an invalid token is provided."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = []

            resp = client.post(
                "/api/metrics",
                headers=auth_headers("bad-token"),
                json={
                    "task_id": "task-uuid-1",
                    "epoch": 0,
                    "loss": 0.5,
                },
            )

            assert resp.status_code == 401

    def test_report_metrics_negative_epoch_rejected(self, client):
        """Epoch must be >= 0; negative values are rejected by validation."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "task_id": "task-uuid-1",
                    "epoch": -1,
                    "loss": 0.5,
                },
            )

            assert resp.status_code == 422

    def test_report_metrics_missing_task_id_rejected(self, client):
        """Request without task_id is rejected by validation."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "epoch": 0,
                    "loss": 0.5,
                },
            )

            assert resp.status_code == 422

    def test_report_metrics_missing_epoch_rejected(self, client):
        """Request without epoch is rejected by validation."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE]

            resp = client.post(
                "/api/metrics",
                headers=auth_headers(),
                json={
                    "task_id": "task-uuid-1",
                    "loss": 0.5,
                },
            )

            assert resp.status_code == 422
