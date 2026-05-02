"""End-to-end single-worker validation test (Task 20.1).

Validates the complete single-worker flow through the Coordinator API:
1. Register a worker node → receives auth token
2. Heartbeat → proves heartbeat works
3. Submit a job (synthetic, MLP, 2 epochs, shard_count=1)
4. Worker polls → receives task assignment
5. Worker calls /start → task status becomes "running"
6. Worker reports per-epoch metrics
7. Worker requests signed upload URL
8. Worker calls /complete with checkpoint path
9. Verify: job status "completed", aggregated_metrics populated
10. Verify: artifact record exists
11. Verify: GET /api/jobs/{id}/results returns correct metrics + checkpoint path
12. Verify: monitoring summary reflects completed job

Since the .env contains placeholder Supabase credentials, this test uses
an in-memory database mock for the coordinator.db module. This validates
the full API contract and business logic end-to-end. A live Supabase
instance would additionally validate network I/O and storage uploads.

What this test DOES verify:
  - Full API request/response contract for all Worker-facing endpoints
  - Node registration, heartbeat, status transitions (idle → busy → idle)
  - Job submission, task creation, shard assignment
  - Task lifecycle: queued → assigned → running → completed
  - Job lifecycle: queued → running → completed
  - Per-epoch metrics reporting and aggregation
  - Artifact record creation
  - Aggregated metrics computation (mean loss, mean accuracy, per-node)
  - Dashboard read endpoints (/api/jobs/{id}, /api/jobs/{id}/results, etc.)
  - Monitoring summary counts

What requires a live Supabase instance:
  - Actual signed upload URL generation from Supabase Storage
  - Checkpoint file upload via signed URL
  - Persistent data across process restarts
"""

from __future__ import annotations

import copy
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# In-memory database mock
# ---------------------------------------------------------------------------

class InMemoryDB:
    """Simple in-memory store that mimics coordinator.db operations."""

    def __init__(self):
        self.tables: dict[str, list[dict[str, Any]]] = {
            "nodes": [],
            "jobs": [],
            "tasks": [],
            "metrics": [],
            "artifacts": [],
        }

    def insert(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        self.tables[table].append(record)
        return copy.deepcopy(record)

    def select(
        self,
        table: str,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.tables.get(table, [])
        if filters:
            for col, val in filters.items():
                rows = [r for r in rows if r.get(col) == val]
        return [copy.deepcopy(r) for r in rows]

    def select_one(
        self,
        table: str,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from coordinator.db import RecordNotFoundError
        rows = self.select(table, columns, filters)
        if not rows:
            raise RecordNotFoundError(
                f"No record found in '{table}' matching {filters}"
            )
        return rows[0]

    def update(
        self,
        table: str,
        data: dict[str, Any],
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        updated = []
        for record in self.tables.get(table, []):
            match = all(record.get(k) == v for k, v in filters.items())
            if match:
                record.update(data)
                updated.append(copy.deepcopy(record))
        return updated


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestSingleWorkerE2E:
    """End-to-end test for the single-worker flow."""

    def test_full_lifecycle(self):
        """Validate the complete single-worker lifecycle end-to-end.

        This is the core milestone validation for Task 20.1.
        """
        # Set required env vars before importing coordinator modules
        os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
        os.environ.setdefault("SUPABASE_KEY", "fake-key")

        mem_db = InMemoryDB()

        # Import the actual modules so they're in sys.modules before patching.
        # This is necessary because coordinator/__init__.py doesn't import
        # submodules, so patch("coordinator.db.insert") would fail with
        # "module 'coordinator' has no attribute 'db'".
        import coordinator.db as db_mod
        import coordinator.dashboard as dashboard_mod
        import coordinator.storage as storage_mod
        import coordinator.heartbeat as heartbeat_mod
        import coordinator.main  # noqa: F401 — ensure app is importable

        patches = [
            # Core db module — all coordinator modules do
            # `from coordinator import db` then call db.select(...) etc.
            # patch.object targets the function on the module object directly.
            patch.object(db_mod, "insert", side_effect=mem_db.insert),
            patch.object(db_mod, "select", side_effect=mem_db.select),
            patch.object(db_mod, "select_one", side_effect=mem_db.select_one),
            patch.object(db_mod, "update", side_effect=mem_db.update),
            # Dashboard imports `from coordinator.db import select, select_one`
            # so we also need to patch the names bound in the dashboard module.
            patch.object(dashboard_mod, "select", side_effect=mem_db.select),
            patch.object(dashboard_mod, "select_one", side_effect=mem_db.select_one),
            # Storage — patch at the import location in main.py
            patch.object(
                coordinator.main,
                "generate_signed_upload_url",
                return_value={
                    "signed_url": "https://fake.supabase.co/storage/v1/upload/sign/checkpoints/test",
                    "token": "fake-token",
                    "path": "test-path/final.pt",
                },
            ),
            # Heartbeat monitor — prevent background task from starting
            patch.object(heartbeat_mod.heartbeat_monitor, "start", return_value=None),
            patch.object(heartbeat_mod.heartbeat_monitor, "stop"),
        ]

        for p in patches:
            p.start()

        try:
            self._run_lifecycle(mem_db)
        finally:
            for p in patches:
                p.stop()

    def _run_lifecycle(self, mem_db: InMemoryDB):
        """Core lifecycle test logic."""
        from fastapi.testclient import TestClient
        from coordinator.main import app

        client = TestClient(app, raise_server_exceptions=False)

        # ==============================================================
        # Step 1: Register a worker node
        # ==============================================================
        reg_resp = client.post("/api/nodes/register", json={
            "node_id": "e2e-worker-001",
            "hostname": "e2e-test-host",
            "cpu_cores": 4,
            "ram_mb": 8192,
            "disk_mb": 50000,
            "os": "Linux 5.15",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        })
        assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
        reg_data = reg_resp.json()
        node_db_id = reg_data["node_db_id"]
        auth_token = reg_data["auth_token"]
        assert node_db_id, "node_db_id should be non-empty"
        assert auth_token, "auth_token should be non-empty"
        auth_headers = {"Authorization": f"Bearer {auth_token}"}

        # Verify node is idle
        nodes_resp = client.get("/api/nodes")
        assert nodes_resp.status_code == 200
        nodes = nodes_resp.json()
        our_node = next((n for n in nodes if n["id"] == node_db_id), None)
        assert our_node is not None, "Registered node not found in /api/nodes"
        assert our_node["status"] == "idle"

        # ==============================================================
        # Step 2: Send a heartbeat
        # ==============================================================
        hb_resp = client.post("/api/nodes/heartbeat", headers=auth_headers)
        assert hb_resp.status_code == 200, f"Heartbeat failed: {hb_resp.text}"

        # ==============================================================
        # Step 3: Submit a job (synthetic, MLP, 2 epochs, shard_count=1)
        # ==============================================================
        job_resp = client.post("/api/jobs", json={
            "job_name": "E2E Test Job",
            "dataset_name": "synthetic",
            "model_type": "MLP",
            "shard_count": 1,
            "hyperparameters": {
                "learning_rate": 0.01,
                "epochs": 2,
                "batch_size": 64,
                "hidden_layers": [64, 32],
                "activation": "relu",
            },
        })
        assert job_resp.status_code == 200, f"Job submission failed: {job_resp.text}"
        job_id = job_resp.json()["job_id"]
        assert job_id, "job_id should be non-empty"

        # Verify job is queued with 1 task
        job_detail_resp = client.get(f"/api/jobs/{job_id}")
        assert job_detail_resp.status_code == 200
        job_detail = job_detail_resp.json()
        assert job_detail["status"] == "queued"
        assert len(job_detail["tasks"]) == 1
        assert job_detail["tasks"][0]["shard_index"] == 0

        # ==============================================================
        # Step 4: Poll for task
        # ==============================================================
        poll_resp = client.get("/api/tasks/poll", headers=auth_headers)
        assert poll_resp.status_code == 200, f"Poll failed: {poll_resp.text}"
        poll_data = poll_resp.json()
        assert poll_data["task_id"] is not None, "Expected a task to be assigned"
        task_id = poll_data["task_id"]
        assert poll_data["job_id"] == job_id
        assert poll_data["dataset_name"] == "synthetic"
        assert poll_data["model_type"] == "MLP"
        assert poll_data["shard_index"] == 0
        assert poll_data["shard_count"] == 1
        assert poll_data["hyperparameters"] is not None
        hp = poll_data["hyperparameters"]
        assert hp["learning_rate"] == 0.01
        assert hp["epochs"] == 2
        assert hp["batch_size"] == 64

        # Verify job is now "running"
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "running"

        # Verify node is now "busy"
        nodes = client.get("/api/nodes").json()
        our_node = next((n for n in nodes if n["id"] == node_db_id), None)
        assert our_node["status"] == "busy"

        # ==============================================================
        # Step 5: Start task
        # ==============================================================
        start_resp = client.post(
            f"/api/tasks/{task_id}/start",
            headers=auth_headers,
        )
        assert start_resp.status_code == 200, f"Start task failed: {start_resp.text}"

        # Verify task is now "running" in job detail
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        task_info = job_detail["tasks"][0]
        assert task_info["status"] == "running"

        # ==============================================================
        # Step 6: Report metrics for 2 epochs
        # ==============================================================
        epoch_metrics = [
            {"epoch": 0, "loss": 2.3, "accuracy": 0.15},
            {"epoch": 1, "loss": 1.8, "accuracy": 0.35},
        ]

        for metrics in epoch_metrics:
            metrics_resp = client.post(
                "/api/metrics",
                json={"task_id": task_id, **metrics},
                headers=auth_headers,
            )
            assert metrics_resp.status_code == 200, (
                f"Metrics report failed for epoch {metrics['epoch']}: {metrics_resp.text}"
            )

        # Verify latest metrics visible in job detail
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        task_info = job_detail["tasks"][0]
        assert task_info["latest_epoch"] == 1
        assert task_info["latest_loss"] is not None
        assert task_info["latest_accuracy"] is not None

        # ==============================================================
        # Step 7: Request signed upload URL
        # ==============================================================
        upload_url_resp = client.post(
            f"/api/tasks/{task_id}/upload-url",
            headers=auth_headers,
        )
        assert upload_url_resp.status_code == 200, (
            f"Upload URL request failed: {upload_url_resp.text}"
        )
        upload_data = upload_url_resp.json()
        assert "signed_url" in upload_data, "Response should contain signed_url"

        # ==============================================================
        # Step 8: Complete task with checkpoint path
        # ==============================================================
        checkpoint_path = f"{job_id}/{task_id}/final.pt"
        complete_resp = client.post(
            f"/api/tasks/{task_id}/complete",
            json={
                "checkpoint_path": checkpoint_path,
                "final_loss": 1.8,
                "final_accuracy": 0.35,
            },
            headers=auth_headers,
        )
        assert complete_resp.status_code == 200, (
            f"Complete task failed: {complete_resp.text}"
        )

        # ==============================================================
        # Step 9: Verify job is "completed" with aggregated metrics
        # ==============================================================
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "completed", (
            f"Expected completed, got {job_detail['status']}"
        )
        assert job_detail["aggregated_metrics"] is not None, (
            "aggregated_metrics should be populated"
        )
        agg = job_detail["aggregated_metrics"]
        assert agg["mean_loss"] is not None, "mean_loss should be set"
        assert agg["mean_accuracy"] is not None, "mean_accuracy should be set"
        # Aggregator uses metrics table: final epoch (epoch 1) has loss=1.8, accuracy=0.35
        assert abs(agg["mean_loss"] - 1.8) < 0.01, (
            f"Expected mean_loss ~1.8, got {agg['mean_loss']}"
        )
        assert abs(agg["mean_accuracy"] - 0.35) < 0.01, (
            f"Expected mean_accuracy ~0.35, got {agg['mean_accuracy']}"
        )
        assert len(agg.get("per_node", [])) == 1, "Should have 1 per-node entry"
        per_node_entry = agg["per_node"][0]
        assert per_node_entry["node_id"] == node_db_id
        assert per_node_entry["task_id"] == task_id

        # Verify completed_at is set
        assert job_detail.get("completed_at") is not None, "completed_at should be set"

        # ==============================================================
        # Step 10: Verify node is back to "idle"
        # ==============================================================
        nodes = client.get("/api/nodes").json()
        our_node = next((n for n in nodes if n["id"] == node_db_id), None)
        assert our_node["status"] == "idle", (
            f"Expected idle after completion, got {our_node['status']}"
        )

        # ==============================================================
        # Step 11: Verify artifact record exists
        # ==============================================================
        artifacts_resp = client.get(f"/api/jobs/{job_id}/artifacts")
        assert artifacts_resp.status_code == 200
        artifacts = artifacts_resp.json()
        assert len(artifacts) >= 1, f"Expected at least 1 artifact, got {len(artifacts)}"
        checkpoint_artifact = next(
            (a for a in artifacts if a["artifact_type"] == "checkpoint"),
            None,
        )
        assert checkpoint_artifact is not None, "Checkpoint artifact should exist"
        assert checkpoint_artifact["storage_path"] == checkpoint_path
        assert checkpoint_artifact["task_id"] == task_id
        assert checkpoint_artifact["job_id"] == job_id
        assert checkpoint_artifact["node_id"] == node_db_id

        # ==============================================================
        # Step 12: Verify GET /api/jobs/{id}/results
        # ==============================================================
        results_resp = client.get(f"/api/jobs/{job_id}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        assert results["job_id"] == job_id
        assert results["status"] == "completed"
        assert results["aggregated_metrics"] is not None
        assert abs(results["aggregated_metrics"]["mean_loss"] - 1.8) < 0.01
        assert abs(results["aggregated_metrics"]["mean_accuracy"] - 0.35) < 0.01

        # Verify per-task checkpoint paths in results
        assert len(results["tasks"]) == 1
        result_task = results["tasks"][0]
        assert result_task["checkpoint_path"] == checkpoint_path
        assert result_task["status"] == "completed"

        # ==============================================================
        # Step 13: Verify monitoring summary
        # ==============================================================
        summary_resp = client.get("/api/monitoring/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert summary["nodes"]["total"] >= 1
        assert summary["nodes"]["idle"] >= 1
        assert summary["jobs"]["completed"] >= 1
        assert summary["jobs"]["total"] >= 1

        # ==============================================================
        # Summary
        # ==============================================================
        print("\n" + "=" * 60)
        print("E2E SINGLE-WORKER VALIDATION: ALL CHECKS PASSED")
        print("=" * 60)
        print(f"  Node ID:       e2e-worker-001")
        print(f"  Node DB ID:    {node_db_id}")
        print(f"  Job ID:        {job_id}")
        print(f"  Task ID:       {task_id}")
        print(f"  Job Status:    completed")
        print(f"  Mean Loss:     {agg['mean_loss']:.4f}")
        print(f"  Mean Accuracy: {agg['mean_accuracy']:.4f}")
        print(f"  Artifact:      {checkpoint_path}")
        print(f"  Per-node:      {per_node_entry}")
        print("=" * 60)
        print("\nVerified endpoints:")
        print("  ✓ POST /api/nodes/register")
        print("  ✓ POST /api/nodes/heartbeat")
        print("  ✓ POST /api/jobs")
        print("  ✓ GET  /api/tasks/poll")
        print("  ✓ POST /api/tasks/{id}/start")
        print("  ✓ POST /api/metrics")
        print("  ✓ POST /api/tasks/{id}/upload-url")
        print("  ✓ POST /api/tasks/{id}/complete")
        print("  ✓ GET  /api/nodes")
        print("  ✓ GET  /api/jobs/{id}")
        print("  ✓ GET  /api/jobs/{id}/results")
        print("  ✓ GET  /api/jobs/{id}/artifacts")
        print("  ✓ GET  /api/monitoring/summary")
        print("\nNote: Signed URL upload to Supabase Storage")
        print("requires a live Supabase instance with valid credentials.")
