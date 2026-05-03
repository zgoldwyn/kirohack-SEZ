"""Unit tests for heartbeat endpoint (8.1) and staleness monitor (8.2).

Tests use FastAPI's TestClient with monkeypatched db functions
so we don't need a live Supabase instance.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from coordinator.constants import JobStatus, NodeStatus, TaskStatus
from coordinator.heartbeat import HeartbeatMonitor
from coordinator.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AUTH_TOKEN = "test-token-abc123"
AUTH_TOKEN_HASH = hashlib.sha256(AUTH_TOKEN.encode()).hexdigest()

SAMPLE_NODE_IDLE = {
    "id": "node-uuid-1",
    "node_id": "worker-1",
    "hostname": "host1",
    "status": NodeStatus.IDLE.value,
    "auth_token_hash": AUTH_TOKEN_HASH,
    "cpu_cores": 4,
    "ram_mb": 8192,
    "gpu_model": None,
    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
}

SAMPLE_NODE_OFFLINE = {
    **SAMPLE_NODE_IDLE,
    "status": NodeStatus.OFFLINE.value,
}

SAMPLE_NODE_BUSY = {
    **SAMPLE_NODE_IDLE,
    "status": NodeStatus.BUSY.value,
}


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def auth_headers(token: str = AUTH_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 8.1 POST /api/nodes/heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeatEndpoint:
    """Tests for POST /api/nodes/heartbeat."""

    def test_heartbeat_updates_timestamp(self, client):
        """Heartbeat updates last_heartbeat for the authenticated node."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE_IDLE]
            mock_db.update.return_value = [SAMPLE_NODE_IDLE]

            resp = client.post(
                "/api/nodes/heartbeat",
                headers=auth_headers(),
            )

            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            # Verify update was called with last_heartbeat
            mock_db.update.assert_called_once()
            call_args = mock_db.update.call_args
            assert call_args[0][0] == "nodes"
            update_data = call_args[0][1]
            assert "last_heartbeat" in update_data
            assert call_args[1]["filters"] == {"id": "node-uuid-1"}

    def test_heartbeat_offline_node_recovers_to_idle(self, client):
        """An offline node that sends a heartbeat is set back to idle."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE_OFFLINE]
            mock_db.update.return_value = [SAMPLE_NODE_OFFLINE]

            resp = client.post(
                "/api/nodes/heartbeat",
                headers=auth_headers(),
            )

            assert resp.status_code == 200

            # Verify status is set to idle along with heartbeat
            call_args = mock_db.update.call_args
            update_data = call_args[0][1]
            assert update_data["status"] == NodeStatus.IDLE.value
            assert "last_heartbeat" in update_data

    def test_heartbeat_idle_node_does_not_change_status(self, client):
        """An idle node's heartbeat does not include a status change."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE_IDLE]
            mock_db.update.return_value = [SAMPLE_NODE_IDLE]

            resp = client.post(
                "/api/nodes/heartbeat",
                headers=auth_headers(),
            )

            assert resp.status_code == 200

            update_data = mock_db.update.call_args[0][1]
            assert "status" not in update_data
            assert "last_heartbeat" in update_data

    def test_heartbeat_busy_node_does_not_change_status(self, client):
        """A busy node's heartbeat does not include a status change."""
        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE_BUSY]
            mock_db.update.return_value = [SAMPLE_NODE_BUSY]

            resp = client.post(
                "/api/nodes/heartbeat",
                headers=auth_headers(),
            )

            assert resp.status_code == 200

            update_data = mock_db.update.call_args[0][1]
            assert "status" not in update_data

    def test_heartbeat_no_auth(self, client):
        """Returns 401 when no auth header is provided."""
        resp = client.post("/api/nodes/heartbeat")
        assert resp.status_code == 401

    def test_heartbeat_invalid_auth(self, client):
        """Returns 401 when an invalid token is provided."""
        with patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = []

            resp = client.post(
                "/api/nodes/heartbeat",
                headers=auth_headers("bad-token"),
            )

            assert resp.status_code == 401

    def test_heartbeat_db_error(self, client):
        """Returns 503 when the database is unavailable."""
        from coordinator.db import DatabaseError

        with patch("coordinator.main.db") as mock_db, \
             patch("coordinator.auth.db") as mock_auth_db:
            mock_auth_db.select.return_value = [SAMPLE_NODE_IDLE]
            mock_db.update.side_effect = DatabaseError("connection lost")
            mock_db.DatabaseError = DatabaseError

            resp = client.post(
                "/api/nodes/heartbeat",
                headers=auth_headers(),
            )

            assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 8.2 Heartbeat staleness monitor
# ---------------------------------------------------------------------------


class TestHeartbeatMonitor:
    """Tests for the HeartbeatMonitor._check_stale_nodes logic."""

    def _make_monitor(self, staleness_threshold: float = 30) -> HeartbeatMonitor:
        return HeartbeatMonitor(
            scan_interval=10,
            staleness_threshold=staleness_threshold,
        )

    def test_marks_stale_node_offline(self):
        """A node with a heartbeat older than the threshold is marked offline."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.IDLE.value,
            "last_heartbeat": stale_time,
        }

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.side_effect = lambda table, **kw: (
                [stale_node] if table == "nodes" else []
            )
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # Verify node was marked offline
            node_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "nodes"
            ]
            assert len(node_update_calls) == 1
            assert node_update_calls[0][0][1]["status"] == NodeStatus.OFFLINE.value
            assert node_update_calls[0][1]["filters"] == {"id": "node-1"}

    def test_fresh_node_not_marked_offline(self):
        """A node with a recent heartbeat is not marked offline."""
        fresh_time = datetime.now(timezone.utc).isoformat()
        fresh_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.IDLE.value,
            "last_heartbeat": fresh_time,
        }

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.return_value = [fresh_node]

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # No update calls should be made
            mock_db.update.assert_not_called()

    def test_already_offline_node_skipped(self):
        """A node already marked offline is not processed again."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        offline_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.OFFLINE.value,
            "last_heartbeat": stale_time,
        }

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.return_value = [offline_node]

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            mock_db.update.assert_not_called()

    def test_stale_node_with_running_task_fails_task(self):
        """When a node goes offline, its running tasks are marked failed."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.BUSY.value,
            "last_heartbeat": stale_time,
        }
        running_task = {
            "id": "task-1",
            "job_id": "job-1",
            "node_id": "node-1",
            "status": TaskStatus.RUNNING.value,
        }

        def mock_select(table, **kwargs):
            filters = kwargs.get("filters")
            if table == "nodes":
                return [stale_node]
            if table == "tasks":
                if filters and filters.get("node_id") == "node-1":
                    return [running_task]
                if filters and filters.get("job_id") == "job-1":
                    return [running_task]
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure") as mock_check:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # Verify task was marked failed
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 1
            task_data = task_update_calls[0][0][1]
            assert task_data["status"] == TaskStatus.FAILED.value
            assert task_data["error_message"] == "node went offline"
            assert "completed_at" in task_data

            # Verify job failure check was triggered
            mock_check.assert_called_once_with("job-1")

    def test_stale_node_with_assigned_task_fails_task(self):
        """When a node goes offline, its assigned tasks are also marked failed."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.IDLE.value,
            "last_heartbeat": stale_time,
        }
        assigned_task = {
            "id": "task-1",
            "job_id": "job-1",
            "node_id": "node-1",
            "status": TaskStatus.ASSIGNED.value,
        }

        def mock_select(table, **kwargs):
            filters = kwargs.get("filters")
            if table == "nodes":
                return [stale_node]
            if table == "tasks":
                if filters and filters.get("node_id") == "node-1":
                    return [assigned_task]
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure") as mock_check:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 1
            assert task_update_calls[0][0][1]["status"] == TaskStatus.FAILED.value

    def test_completed_task_not_failed_on_node_offline(self):
        """Completed tasks are not affected when a node goes offline."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.IDLE.value,
            "last_heartbeat": stale_time,
        }
        completed_task = {
            "id": "task-1",
            "job_id": "job-1",
            "node_id": "node-1",
            "status": TaskStatus.COMPLETED.value,
        }

        def mock_select(table, **kwargs):
            filters = kwargs.get("filters")
            if table == "nodes":
                return [stale_node]
            if table == "tasks":
                if filters and filters.get("node_id") == "node-1":
                    return [completed_task]
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # Only the node update (marking offline), no task updates
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 0

    def test_multiple_stale_nodes_processed(self):
        """Multiple stale nodes are all marked offline."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        nodes = [
            {
                "id": "node-1",
                "node_id": "worker-1",
                "status": NodeStatus.IDLE.value,
                "last_heartbeat": stale_time,
            },
            {
                "id": "node-2",
                "node_id": "worker-2",
                "status": NodeStatus.BUSY.value,
                "last_heartbeat": stale_time,
            },
        ]

        def mock_select(table, **kwargs):
            if table == "nodes":
                return nodes
            if table == "tasks":
                return []
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            node_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "nodes"
            ]
            assert len(node_update_calls) == 2
            offline_node_ids = {
                c[1]["filters"]["id"] for c in node_update_calls
            }
            assert offline_node_ids == {"node-1", "node-2"}

    def test_stale_node_with_tasks_from_multiple_jobs(self):
        """Failing tasks from multiple jobs triggers check_job_failure for each."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.BUSY.value,
            "last_heartbeat": stale_time,
        }
        tasks = [
            {
                "id": "task-1",
                "job_id": "job-1",
                "node_id": "node-1",
                "status": TaskStatus.RUNNING.value,
            },
            {
                "id": "task-2",
                "job_id": "job-2",
                "node_id": "node-1",
                "status": TaskStatus.ASSIGNED.value,
            },
        ]

        def mock_select(table, **kwargs):
            filters = kwargs.get("filters")
            if table == "nodes":
                return [stale_node]
            if table == "tasks":
                if filters and filters.get("node_id") == "node-1":
                    return tasks
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure") as mock_check:
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # Both tasks should be failed
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 2

            # check_job_failure called for both jobs
            checked_jobs = {c[0][0] for c in mock_check.call_args_list}
            assert checked_jobs == {"job-1", "job-2"}

    def test_node_with_no_heartbeat_treated_as_stale(self):
        """A node that never sent a heartbeat (None) is treated as stale."""
        node_no_hb = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.IDLE.value,
            "last_heartbeat": None,
        }

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.side_effect = lambda table, **kw: (
                [node_no_hb] if table == "nodes" else []
            )
            mock_db.update.return_value = []

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            node_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "nodes"
            ]
            assert len(node_update_calls) == 1
            assert node_update_calls[0][0][1]["status"] == NodeStatus.OFFLINE.value

    def test_offline_node_with_active_task_adjusts_barrier_and_triggers_aggregation(self):
        """When a node goes offline with a running task in a collaborative
        training job, the heartbeat monitor:
        1. Fails the task
        2. Calls barrier.remove_worker() to adjust the synchronization barrier
        3. Checks if the barrier is now met (other workers already submitted)
        4. If met, triggers aggregate_round()
        Requirements: 14.1, 14.2
        """
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.BUSY.value,
            "last_heartbeat": stale_time,
        }
        running_task = {
            "id": "task-1",
            "job_id": "job-1",
            "node_id": "node-1",
            "status": TaskStatus.RUNNING.value,
        }
        running_job = {
            "id": "job-1",
            "status": JobStatus.RUNNING.value,
            "current_round": 2,
        }

        def mock_select(table, **kwargs):
            filters = kwargs.get("filters")
            if table == "nodes":
                return [stale_node]
            if table == "tasks":
                if filters and filters.get("node_id") == "node-1":
                    return [running_task]
            if table == "jobs":
                return [running_job]
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.barrier_mod") as mock_barrier, \
             patch("coordinator.heartbeat.aggregate_round") as mock_agg, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            # Barrier is met after removing the worker (other workers
            # already submitted their gradients for round 2)
            mock_barrier.check_barrier.return_value = True

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # Verify task was failed
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 1
            assert task_update_calls[0][0][1]["status"] == TaskStatus.FAILED.value

            # Verify barrier.remove_worker was called
            mock_barrier.remove_worker.assert_called_once_with("job-1", "task-1")

            # Verify barrier.check_barrier was called for current_round
            mock_barrier.check_barrier.assert_called_once_with("job-1", 2)

            # Verify aggregation was triggered since barrier is met
            mock_agg.assert_called_once_with("job-1", 2)

    def test_offline_node_barrier_not_met_no_aggregation(self):
        """When a node goes offline and the barrier is NOT met after
        removing the worker, aggregation should NOT be triggered.
        Requirements: 14.1, 14.2
        """
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        stale_node = {
            "id": "node-1",
            "node_id": "worker-1",
            "status": NodeStatus.BUSY.value,
            "last_heartbeat": stale_time,
        }
        running_task = {
            "id": "task-1",
            "job_id": "job-1",
            "node_id": "node-1",
            "status": TaskStatus.RUNNING.value,
        }
        running_job = {
            "id": "job-1",
            "status": JobStatus.RUNNING.value,
            "current_round": 1,
        }

        def mock_select(table, **kwargs):
            filters = kwargs.get("filters")
            if table == "nodes":
                return [stale_node]
            if table == "tasks":
                if filters and filters.get("node_id") == "node-1":
                    return [running_task]
            if table == "jobs":
                return [running_job]
            return []

        with patch("coordinator.heartbeat.db") as mock_db, \
             patch("coordinator.heartbeat.barrier_mod") as mock_barrier, \
             patch("coordinator.heartbeat.aggregate_round") as mock_agg, \
             patch("coordinator.heartbeat.check_job_failure"):
            mock_db.select.side_effect = mock_select
            mock_db.update.return_value = []

            # Barrier is NOT met after removing the worker
            mock_barrier.check_barrier.return_value = False

            monitor = self._make_monitor()
            monitor._check_stale_nodes()

            # Verify task was failed
            task_update_calls = [
                c for c in mock_db.update.call_args_list
                if c[0][0] == "tasks"
            ]
            assert len(task_update_calls) == 1
            assert task_update_calls[0][0][1]["status"] == TaskStatus.FAILED.value

            # Verify barrier.remove_worker was called
            mock_barrier.remove_worker.assert_called_once_with("job-1", "task-1")

            # Verify barrier.check_barrier was called
            mock_barrier.check_barrier.assert_called_once_with("job-1", 1)

            # Aggregation should NOT be triggered since barrier is not met
            mock_agg.assert_not_called()
