"""End-to-end single-worker validation test (Task 20.1).

Validates the available Worker-facing endpoints through the Coordinator API:
1. Register a worker node → receives auth token
2. Heartbeat → proves heartbeat works
3. Submit a job (synthetic, MLP, 2 epochs, shard_count=1)
4. Worker polls → receives task assignment
5. Worker calls /start → task status becomes "running"
6. Worker calls /fail → task and job marked failed

In the collaborative distributed training model, job completion is driven
by the Coordinator's aggregation loop (barrier → aggregate_round →
complete_job) rather than individual worker /complete calls.  This E2E
test validates the endpoints that are still available without requiring
torch or complex param_server/barrier mocking.

Since the .env contains placeholder Supabase credentials, this test uses
an in-memory database mock for the coordinator.db module.
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
            "training_rounds": [],
            "gradient_submissions": [],
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
        """Validate the worker lifecycle: register, heartbeat, job submit,
        poll, start, and fail — all endpoints that exist in the
        collaborative training model.
        """
        # Set required env vars before importing coordinator modules
        os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
        os.environ.setdefault("SUPABASE_KEY", "fake-key")

        mem_db = InMemoryDB()

        import coordinator.db as db_mod
        import coordinator.dashboard as dashboard_mod
        import coordinator.heartbeat as heartbeat_mod
        import coordinator.main  # noqa: F401

        patches = [
            patch.object(db_mod, "insert", side_effect=mem_db.insert),
            patch.object(db_mod, "select", side_effect=mem_db.select),
            patch.object(db_mod, "select_one", side_effect=mem_db.select_one),
            patch.object(db_mod, "update", side_effect=mem_db.update),
            patch.object(dashboard_mod, "select", side_effect=mem_db.select),
            patch.object(dashboard_mod, "select_one", side_effect=mem_db.select_one),
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
        # Step 6: Fail the task (simulating a training error)
        # ==============================================================
        fail_resp = client.post(
            f"/api/tasks/{task_id}/fail",
            headers=auth_headers,
            json={"error_message": "OOM during training"},
        )
        assert fail_resp.status_code == 200, f"Fail task failed: {fail_resp.text}"

        # Verify task is "failed"
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        task_info = job_detail["tasks"][0]
        assert task_info["status"] == "failed"
        assert task_info["error_message"] == "OOM during training"

        # Verify job is "failed" (only task failed, no active tasks remain)
        assert job_detail["status"] == "failed"
        assert job_detail.get("error_summary") is not None
        failed_tasks = job_detail["error_summary"]["failed_tasks"]
        assert len(failed_tasks) == 1
        assert failed_tasks[0]["error_message"] == "OOM during training"

        # Verify node is back to "idle"
        nodes = client.get("/api/nodes").json()
        our_node = next((n for n in nodes if n["id"] == node_db_id), None)
        assert our_node["status"] == "idle"

        # ==============================================================
        # Step 7: Verify monitoring summary
        # ==============================================================
        summary_resp = client.get("/api/monitoring/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert summary["nodes"]["total"] >= 1
        assert summary["nodes"]["idle"] >= 1
        assert summary["jobs"]["failed"] >= 1
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
        print(f"  Job Status:    failed (expected — single task failed)")
        print("=" * 60)
        print("\nVerified endpoints:")
        print("  ✓ POST /api/nodes/register")
        print("  ✓ POST /api/nodes/heartbeat")
        print("  ✓ POST /api/jobs")
        print("  ✓ GET  /api/tasks/poll")
        print("  ✓ POST /api/tasks/{id}/start")
        print("  ✓ POST /api/tasks/{id}/fail")
        print("  ✓ GET  /api/nodes")
        print("  ✓ GET  /api/jobs/{id}")
        print("  ✓ GET  /api/monitoring/summary")
