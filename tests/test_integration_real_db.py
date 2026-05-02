"""Integration tests against a live Supabase database.

These tests exercise the full Coordinator API through FastAPI's TestClient
with the REAL database layer — no mocking of coordinator.db.  The only
things mocked are:

- Supabase Storage signed URL generation (requires bucket config)
- The heartbeat background monitor (to prevent interference)

Each test cleans up all records it creates, so the database is left in
the same state it was found in.

Requirements:
    - SUPABASE_URL and SUPABASE_SERVICE_KEY set in .env (real credentials)
    - The schema from scripts/bootstrap.sql applied to the database
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip the entire module if Supabase credentials are not set
_url = os.getenv("SUPABASE_URL", "")
_key = os.getenv("SUPABASE_KEY", "")
pytestmark = pytest.mark.skipif(
    not _url or not _key,
    reason="Real Supabase credentials required (SUPABASE_URL / SUPABASE_KEY)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Unique prefix so concurrent test runs don't collide
_RUN_ID = uuid.uuid4().hex[:8]


def _unique(name: str) -> str:
    """Return a test-unique identifier to avoid collisions."""
    return f"test-{_RUN_ID}-{name}"


class DBCleanup:
    """Track records created during a test and delete them in reverse order."""

    def __init__(self):
        # List of (table, record_id) tuples
        self._records: list[tuple[str, str]] = []

    def track(self, table: str, record_id: str) -> None:
        self._records.append((table, record_id))

    def cleanup(self) -> None:
        """Delete all tracked records in reverse insertion order."""
        from coordinator.db import delete

        # Reverse so child records (metrics, artifacts, tasks) are deleted
        # before parent records (jobs, nodes).
        for table, record_id in reversed(self._records):
            try:
                delete(table, filters={"id": record_id})
            except Exception:
                pass  # Best-effort cleanup


@pytest.fixture
def cleanup():
    """Provide a DBCleanup instance and run cleanup after the test."""
    c = DBCleanup()
    yield c
    c.cleanup()


@pytest.fixture
def client():
    """Create a FastAPI TestClient with the heartbeat monitor disabled."""
    import coordinator.heartbeat as heartbeat_mod

    with patch.object(heartbeat_mod.heartbeat_monitor, "start", return_value=None), \
         patch.object(heartbeat_mod.heartbeat_monitor, "stop"):
        from coordinator.main import app
        from fastapi.testclient import TestClient
        yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Test: Full job lifecycle against real DB
# ---------------------------------------------------------------------------


class TestRealDBFullLifecycle:
    """End-to-end lifecycle test using the real Supabase database."""

    def test_register_submit_poll_complete(self, client, cleanup):
        """Walk through the full single-worker lifecycle with real DB calls.

        1. Register a node → verify it's stored in the DB
        2. Heartbeat → verify timestamp updates
        3. Submit a job → verify job + tasks created in DB
        4. Poll → verify task assigned, node busy
        5. Start task → verify status transition
        6. Report metrics → verify metrics stored
        7. Complete task → verify aggregation, job completed
        8. Verify dashboard endpoints return correct data
        9. Clean up all created records
        """
        node_id = _unique("worker")

        # ---- Step 1: Register ----
        reg_resp = client.post("/api/nodes/register", json={
            "node_id": node_id,
            "hostname": "integration-test-host",
            "cpu_cores": 4,
            "ram_mb": 8192,
            "disk_mb": 50000,
            "os": "Linux",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
        reg_data = reg_resp.json()
        node_db_id = reg_data["node_db_id"]
        auth_token = reg_data["auth_token"]
        cleanup.track("nodes", node_db_id)

        headers = {"Authorization": f"Bearer {auth_token}"}

        # Verify node exists in DB via the dashboard endpoint
        nodes_resp = client.get("/api/nodes")
        assert nodes_resp.status_code == 200
        nodes = nodes_resp.json()
        our_node = next((n for n in nodes if n["id"] == node_db_id), None)
        assert our_node is not None, "Node not found in /api/nodes"
        assert our_node["status"] == "idle"
        assert our_node["node_id"] == node_id

        # ---- Step 2: Heartbeat ----
        hb_resp = client.post("/api/nodes/heartbeat", headers=headers)
        assert hb_resp.status_code == 200

        # Verify heartbeat updated (re-fetch node)
        nodes_after_hb = client.get("/api/nodes").json()
        node_after_hb = next((n for n in nodes_after_hb if n["id"] == node_db_id), None)
        assert node_after_hb is not None
        assert node_after_hb["last_heartbeat"] is not None

        # ---- Step 3: Submit a job ----
        # Mock only the storage signed URL (not the DB)
        with patch("coordinator.main.generate_signed_upload_url") as mock_storage:
            mock_storage.return_value = {
                "signed_url": "https://fake-storage.example.com/upload",
                "token": "fake-token",
                "path": "test/final.pt",
            }

            job_resp = client.post("/api/jobs", json={
                "job_name": _unique("job"),
                "dataset_name": "synthetic",
                "model_type": "MLP",
                "shard_count": 1,
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "epochs": 2,
                    "batch_size": 32,
                    "hidden_layers": [64, 32],
                    "activation": "relu",
                },
            })
            assert job_resp.status_code == 200, f"Job submission failed: {job_resp.text}"
            job_id = job_resp.json()["job_id"]
            cleanup.track("jobs", job_id)

            # Verify job in DB
            job_detail = client.get(f"/api/jobs/{job_id}").json()
            assert job_detail["status"] == "queued"
            assert len(job_detail["tasks"]) == 1
            task_id = job_detail["tasks"][0]["id"]
            cleanup.track("tasks", task_id)
            assert job_detail["tasks"][0]["shard_index"] == 0
            assert job_detail["tasks"][0]["status"] == "queued"

            # ---- Step 4: Poll for task ----
            poll_resp = client.get("/api/tasks/poll", headers=headers)
            assert poll_resp.status_code == 200
            poll_data = poll_resp.json()
            assert poll_data["task_id"] == task_id
            assert poll_data["job_id"] == job_id
            assert poll_data["dataset_name"] == "synthetic"
            assert poll_data["model_type"] == "MLP"
            assert poll_data["shard_index"] == 0
            assert poll_data["shard_count"] == 1

            # Verify node is now busy
            node_now = next(
                (n for n in client.get("/api/nodes").json() if n["id"] == node_db_id),
                None,
            )
            assert node_now["status"] == "busy"

            # Verify job is now running
            job_now = client.get(f"/api/jobs/{job_id}").json()
            assert job_now["status"] == "running"
            assert job_now["started_at"] is not None

            # ---- Step 5: Start task ----
            start_resp = client.post(f"/api/tasks/{task_id}/start", headers=headers)
            assert start_resp.status_code == 200

            # Verify task is running in DB
            job_after_start = client.get(f"/api/jobs/{job_id}").json()
            task_after_start = job_after_start["tasks"][0]
            assert task_after_start["status"] == "running"
            assert task_after_start["started_at"] is not None

            # ---- Step 6: Report metrics ----
            for epoch, loss, acc in [(0, 2.3, 0.15), (1, 1.5, 0.42)]:
                m_resp = client.post("/api/metrics", headers=headers, json={
                    "task_id": task_id,
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": acc,
                })
                assert m_resp.status_code == 200, f"Metrics epoch {epoch} failed: {m_resp.text}"

            # Verify metrics visible in job detail
            job_with_metrics = client.get(f"/api/jobs/{job_id}").json()
            task_with_metrics = job_with_metrics["tasks"][0]
            assert task_with_metrics["latest_epoch"] == 1
            assert task_with_metrics["latest_loss"] is not None
            assert task_with_metrics["latest_accuracy"] is not None

            # Track metrics for cleanup (fetch them from DB)
            from coordinator.db import select
            metrics_rows = select("metrics", filters={"task_id": task_id})
            for m in metrics_rows:
                cleanup.track("metrics", m["id"])

            # ---- Step 7: Request upload URL (mocked storage) ----
            upload_resp = client.post(f"/api/tasks/{task_id}/upload-url", headers=headers)
            assert upload_resp.status_code == 200
            assert "signed_url" in upload_resp.json()

            # ---- Step 8: Complete task ----
            checkpoint_path = f"{job_id}/{task_id}/final.pt"
            complete_resp = client.post(
                f"/api/tasks/{task_id}/complete",
                headers=headers,
                json={
                    "checkpoint_path": checkpoint_path,
                    "final_loss": 1.5,
                    "final_accuracy": 0.42,
                },
            )
            assert complete_resp.status_code == 200, f"Complete failed: {complete_resp.text}"

            # Track artifact for cleanup
            artifacts = select("artifacts", filters={"task_id": task_id})
            for a in artifacts:
                cleanup.track("artifacts", a["id"])

            # ---- Step 9: Verify job completed with aggregated metrics ----
            job_final = client.get(f"/api/jobs/{job_id}").json()
            assert job_final["status"] == "completed", (
                f"Expected completed, got {job_final['status']}"
            )
            assert job_final["completed_at"] is not None
            agg = job_final["aggregated_metrics"]
            assert agg is not None
            # Aggregator uses final epoch (epoch 1): loss=1.5, accuracy=0.42
            assert abs(float(agg["mean_loss"]) - 1.5) < 0.01
            assert abs(float(agg["mean_accuracy"]) - 0.42) < 0.01
            assert len(agg["per_node"]) == 1
            assert agg["per_node"][0]["node_id"] == node_db_id

            # ---- Step 10: Verify node back to idle ----
            node_final = next(
                (n for n in client.get("/api/nodes").json() if n["id"] == node_db_id),
                None,
            )
            assert node_final["status"] == "idle"

            # ---- Step 11: Verify artifacts endpoint ----
            artifacts_resp = client.get(f"/api/jobs/{job_id}/artifacts")
            assert artifacts_resp.status_code == 200
            art_list = artifacts_resp.json()
            assert len(art_list) >= 1
            assert any(a["storage_path"] == checkpoint_path for a in art_list)

            # ---- Step 12: Verify results endpoint ----
            results_resp = client.get(f"/api/jobs/{job_id}/results")
            assert results_resp.status_code == 200
            results = results_resp.json()
            assert results["status"] == "completed"
            assert results["aggregated_metrics"] is not None
            assert len(results["tasks"]) == 1
            assert results["tasks"][0]["checkpoint_path"] == checkpoint_path

            # ---- Step 13: Verify monitoring summary ----
            summary = client.get("/api/monitoring/summary").json()
            assert summary["nodes"]["total"] >= 1
            assert summary["jobs"]["total"] >= 1


class TestRealDBDuplicateRegistration:
    """Test that duplicate node registration is rejected by the real DB."""

    def test_duplicate_node_id_returns_409(self, client, cleanup):
        node_id = _unique("dup-node")

        # First registration succeeds
        resp1 = client.post("/api/nodes/register", json={
            "node_id": node_id,
            "hostname": "dup-test",
            "cpu_cores": 2,
            "ram_mb": 4096,
            "disk_mb": 20000,
            "os": "Linux",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert resp1.status_code == 200
        cleanup.track("nodes", resp1.json()["node_db_id"])

        # Second registration with same node_id fails
        resp2 = client.post("/api/nodes/register", json={
            "node_id": node_id,
            "hostname": "dup-test-2",
            "cpu_cores": 2,
            "ram_mb": 4096,
            "disk_mb": 20000,
            "os": "Linux",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert resp2.status_code == 409


class TestRealDBJobValidation:
    """Test job submission validation against the real DB."""

    def test_unsupported_dataset_rejected(self, client, cleanup):
        """Submitting a job with an unsupported dataset returns 422."""
        # Need at least one idle node for shard_count validation to pass
        node_id = _unique("val-node")
        reg = client.post("/api/nodes/register", json={
            "node_id": node_id,
            "hostname": "val-test",
            "cpu_cores": 2,
            "ram_mb": 4096,
            "disk_mb": 20000,
            "os": "Linux",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert reg.status_code == 200
        cleanup.track("nodes", reg.json()["node_db_id"])

        resp = client.post("/api/jobs", json={
            "dataset_name": "IMAGENET",
            "model_type": "MLP",
            "shard_count": 1,
        })
        assert resp.status_code == 422

    def test_shard_count_exceeds_idle_nodes(self, client, cleanup):
        """Submitting a job with shard_count > idle nodes returns 400."""
        node_id = _unique("shard-node")
        reg = client.post("/api/nodes/register", json={
            "node_id": node_id,
            "hostname": "shard-test",
            "cpu_cores": 2,
            "ram_mb": 4096,
            "disk_mb": 20000,
            "os": "Linux",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert reg.status_code == 200
        cleanup.track("nodes", reg.json()["node_db_id"])

        # Request more shards than idle nodes (we only registered 1,
        # but there may be other idle nodes in the DB — use a very large number)
        resp = client.post("/api/jobs", json={
            "dataset_name": "synthetic",
            "model_type": "MLP",
            "shard_count": 9999,
        })
        assert resp.status_code == 400


class TestRealDBTaskFailure:
    """Test task failure and job failure detection against the real DB."""

    def test_task_failure_marks_job_failed(self, client, cleanup):
        """When the only task in a job fails, the job is marked failed."""
        node_id = _unique("fail-worker")

        # Register
        reg = client.post("/api/nodes/register", json={
            "node_id": node_id,
            "hostname": "fail-test",
            "cpu_cores": 4,
            "ram_mb": 8192,
            "disk_mb": 50000,
            "os": "Linux",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert reg.status_code == 200
        node_db_id = reg.json()["node_db_id"]
        auth_token = reg.json()["auth_token"]
        cleanup.track("nodes", node_db_id)
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Submit job
        with patch("coordinator.main.generate_signed_upload_url"):
            job_resp = client.post("/api/jobs", json={
                "job_name": _unique("fail-job"),
                "dataset_name": "MNIST",
                "model_type": "MLP",
                "shard_count": 1,
            })
            assert job_resp.status_code == 200
            job_id = job_resp.json()["job_id"]
            cleanup.track("jobs", job_id)

            # Get task ID
            job_detail = client.get(f"/api/jobs/{job_id}").json()
            task_id = job_detail["tasks"][0]["id"]
            cleanup.track("tasks", task_id)

            # Poll and start
            poll = client.get("/api/tasks/poll", headers=headers)
            assert poll.status_code == 200
            assert poll.json()["task_id"] == task_id

            start = client.post(f"/api/tasks/{task_id}/start", headers=headers)
            assert start.status_code == 200

            # Fail the task
            fail_resp = client.post(
                f"/api/tasks/{task_id}/fail",
                headers=headers,
                json={"error_message": "OOM during training"},
            )
            assert fail_resp.status_code == 200

            # Verify job is failed
            job_final = client.get(f"/api/jobs/{job_id}").json()
            assert job_final["status"] == "failed"
            assert job_final["error_summary"] is not None
            failed_tasks = job_final["error_summary"]["failed_tasks"]
            assert len(failed_tasks) == 1
            assert failed_tasks[0]["error_message"] == "OOM during training"

            # Verify node is back to idle
            node_final = next(
                (n for n in client.get("/api/nodes").json() if n["id"] == node_db_id),
                None,
            )
            assert node_final["status"] == "idle"


class TestRealDBAuthRejection:
    """Test that invalid auth tokens are rejected by the real DB lookup."""

    def test_invalid_token_returns_401(self, client):
        """A request with a bogus token gets 401 from the real auth flow."""
        resp = client.post(
            "/api/nodes/heartbeat",
            headers={"Authorization": "Bearer totally-bogus-token-12345"},
        )
        assert resp.status_code == 401

    def test_missing_auth_returns_401(self, client):
        """A request with no auth header gets 401."""
        resp = client.get("/api/tasks/poll")
        assert resp.status_code == 401
