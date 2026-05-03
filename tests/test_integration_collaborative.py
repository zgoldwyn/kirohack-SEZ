"""Integration tests for collaborative distributed training (Task 50).

Validates the full collaborative training lifecycle through the Coordinator
API using an in-memory database mock (same pattern as test_e2e_single_worker.py).

Tests:
  50.1 Full collaborative training lifecycle (2 workers, 2 rounds)
  50.2 Worker failure mid-training (3 workers, 1 fails at round 1)
  50.3 All workers fail (2 workers both fail)
  50.4 Round validation (wrong round → 409, correct round → 200)

Requirements: 1.1, 3.1, 4.1, 4.2, 5.1, 5.2, 5.3, 6.1, 6.2, 6.4, 6.5,
              7.1, 13.3, 14.1, 14.2, 14.3
"""

from __future__ import annotations

import copy
import io
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# In-memory database mock (extended from test_e2e_single_worker.py)
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

    def delete(
        self,
        table: str,
        filters: dict[str, Any],
    ) -> None:
        rows = self.tables.get(table, [])
        self.tables[table] = [
            r
            for r in rows
            if not all(r.get(k) == v for k, v in filters.items())
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_model_bytes() -> bytes:
    """Create a small serialized state_dict for testing."""
    state_dict = {
        "layer.weight": torch.randn(4, 4),
        "layer.bias": torch.randn(4),
    }
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _make_fake_gradient_bytes() -> bytes:
    """Create a small serialized gradient dict for testing."""
    grads = {
        "layer.weight": torch.randn(4, 4),
        "layer.bias": torch.randn(4),
    }
    buf = io.BytesIO()
    torch.save(grads, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_patches(mem_db: InMemoryDB):
    """Create the standard set of patches for the in-memory DB mock."""
    # Set required env vars before importing coordinator modules
    os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "fake-key")

    import coordinator.db as db_mod
    import coordinator.dashboard as dashboard_mod
    import coordinator.heartbeat as heartbeat_mod
    import coordinator.main as main_mod
    import coordinator.storage as storage_mod
    import coordinator.param_server as ps_mod
    import coordinator.barrier as barrier_mod
    import coordinator.aggregator as agg_mod
    import coordinator.scheduler as sched_mod
    import coordinator.auth as auth_mod

    patches = [
        # Core DB operations — coordinator.db
        patch.object(db_mod, "insert", side_effect=mem_db.insert),
        patch.object(db_mod, "select", side_effect=mem_db.select),
        patch.object(db_mod, "select_one", side_effect=mem_db.select_one),
        patch.object(db_mod, "update", side_effect=mem_db.update),
        patch.object(db_mod, "delete", side_effect=mem_db.delete),
        # Dashboard uses its own db import
        patch.object(dashboard_mod, "select", side_effect=mem_db.select),
        patch.object(dashboard_mod, "select_one", side_effect=mem_db.select_one),
        # Heartbeat monitor — disable background tasks
        patch.object(heartbeat_mod.heartbeat_monitor, "start", return_value=None),
        patch.object(heartbeat_mod.heartbeat_monitor, "stop"),
        # Mock initialize_model at the main module level (where it's imported)
        patch.object(
            main_mod,
            "initialize_model",
            return_value="parameters/fake-job-id/current.pt",
        ),
        # Mock upload_blob at the main module level (where it's imported via
        # `from coordinator.storage import upload_blob`)
        patch.object(main_mod, "upload_blob", return_value=None),
        # Also mock at the storage module level for any internal callers
        patch.object(storage_mod, "upload_blob", return_value=None),
        patch.object(storage_mod, "download_blob", return_value=_make_fake_model_bytes()),
        patch.object(storage_mod, "delete_blob", return_value=None),
        patch.object(storage_mod, "list_blobs", return_value=[]),
        # Mock get_parameters at the main module level (imported from param_server)
        patch.object(main_mod, "get_parameters", return_value=_make_fake_model_bytes()),
        # Also mock at param_server level for aggregator usage
        patch.object(ps_mod, "get_parameters", return_value=_make_fake_model_bytes()),
        patch.object(ps_mod, "update_parameters", return_value=None),
        patch.object(ps_mod, "store_checkpoint", return_value="checkpoints/fake/final.pt"),
    ]

    return patches


def _register_node(client, node_id: str) -> tuple[str, str, dict]:
    """Register a worker node and return (node_db_id, auth_token, headers)."""
    resp = client.post(
        "/api/nodes/register",
        json={
            "node_id": node_id,
            "hostname": f"{node_id}-host",
            "cpu_cores": 4,
            "ram_mb": 8192,
            "disk_mb": 50000,
            "os": "Linux 5.15",
            "python_version": "3.11.0",
            "pytorch_version": "2.6.0",
        },
    )
    assert resp.status_code == 200, f"Registration failed for {node_id}: {resp.text}"
    data = resp.json()
    node_db_id = data["node_db_id"]
    auth_token = data["auth_token"]
    headers = {"Authorization": f"Bearer {auth_token}"}
    return node_db_id, auth_token, headers


def _submit_job(client, shard_count: int, epochs: int) -> str:
    """Submit a training job and return the job_id."""
    resp = client.post(
        "/api/jobs",
        json={
            "job_name": "Integration Test Job",
            "dataset_name": "synthetic",
            "model_type": "MLP",
            "shard_count": shard_count,
            "hyperparameters": {
                "learning_rate": 0.01,
                "epochs": epochs,
                "batch_size": 64,
                "hidden_layers": [64, 32],
                "activation": "relu",
            },
        },
    )
    assert resp.status_code == 200, f"Job submission failed: {resp.text}"
    return resp.json()["job_id"]


def _poll_and_start(client, headers: dict) -> tuple[str, str]:
    """Poll for a task and start it. Returns (task_id, job_id)."""
    poll_resp = client.get("/api/tasks/poll", headers=headers)
    assert poll_resp.status_code == 200, f"Poll failed: {poll_resp.text}"
    poll_data = poll_resp.json()
    assert poll_data["task_id"] is not None, "Expected a task to be assigned"
    task_id = poll_data["task_id"]
    job_id = poll_data["job_id"]

    start_resp = client.post(f"/api/tasks/{task_id}/start", headers=headers)
    assert start_resp.status_code == 200, f"Start task failed: {start_resp.text}"

    return task_id, job_id


def _submit_gradient(
    client,
    job_id: str,
    task_id: str,
    round_number: int,
    headers: dict,
    local_loss: float = 0.5,
    local_accuracy: float = 0.8,
) -> dict:
    """Submit a gradient for a given round. Returns the response JSON."""
    gradient_data = _make_fake_gradient_bytes()
    resp = client.post(
        f"/api/jobs/{job_id}/gradients"
        f"?round_number={round_number}"
        f"&task_id={task_id}"
        f"&local_loss={local_loss}"
        f"&local_accuracy={local_accuracy}",
        headers=headers,
        content=gradient_data,
    )
    return resp



# ---------------------------------------------------------------------------
# 50.1 Full collaborative training lifecycle
# ---------------------------------------------------------------------------


class TestFullCollaborativeLifecycle:
    """Integration test: Full collaborative training lifecycle.

    Register 2 nodes → submit job with worker_count=2 → tasks assigned →
    Workers download params → compute gradients → submit gradients →
    barrier met → aggregation → repeat for all rounds → job completed
    with final checkpoint and per-round metrics.

    Requirements: 1.1, 3.1, 4.1, 4.2, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1
    """

    def test_full_lifecycle(self):
        os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
        os.environ.setdefault("SUPABASE_KEY", "fake-key")

        mem_db = InMemoryDB()
        patches = _setup_patches(mem_db)

        # We also need to mock aggregate_round at the coordinator.main level
        # so we can simulate its effects on the in-memory DB.
        import coordinator.main as main_mod

        def mock_aggregate_round(job_id: str, round_number: int):
            """Simulate aggregation: advance round, create next round or complete."""
            # Get job
            jobs = mem_db.select("jobs", filters={"id": job_id})
            if not jobs:
                return
            job = jobs[0]
            total_rounds = job.get("total_rounds", 2)
            next_round = round_number + 1

            # Mark current round as completed
            rounds = mem_db.select(
                "training_rounds",
                filters={"job_id": job_id, "round_number": round_number},
            )
            if rounds:
                now = datetime.now(timezone.utc).isoformat()
                mem_db.update(
                    "training_rounds",
                    {
                        "status": "completed",
                        "completed_at": now,
                        "global_loss": 0.3 - 0.1 * round_number,
                        "global_accuracy": 0.7 + 0.1 * round_number,
                    },
                    filters={"id": rounds[0]["id"]},
                )

            # Advance job's current_round
            mem_db.update("jobs", {"current_round": next_round}, filters={"id": job_id})

            if next_round >= total_rounds:
                # Complete the job
                now = datetime.now(timezone.utc).isoformat()
                # Build aggregated_metrics
                all_rounds = mem_db.select("training_rounds", filters={"job_id": job_id})
                completed = [r for r in all_rounds if r.get("status") == "completed"]
                per_round = []
                for r in sorted(completed, key=lambda x: x.get("round_number", 0)):
                    per_round.append({
                        "round_number": r.get("round_number"),
                        "global_loss": r.get("global_loss"),
                        "global_accuracy": r.get("global_accuracy"),
                    })

                aggregated_metrics = {
                    "per_round": per_round,
                    "mean_loss": sum(
                        r["global_loss"] for r in per_round if r["global_loss"] is not None
                    ) / max(len(per_round), 1),
                    "mean_accuracy": sum(
                        r["global_accuracy"] for r in per_round if r["global_accuracy"] is not None
                    ) / max(len(per_round), 1),
                    "per_worker": [],
                }

                mem_db.update(
                    "jobs",
                    {
                        "status": "completed",
                        "aggregated_metrics": aggregated_metrics,
                        "completed_at": now,
                        "global_model_path": f"checkpoints/{job_id}/round_{round_number}.pt",
                    },
                    filters={"id": job_id},
                )

                # Mark all active tasks as completed
                tasks = mem_db.select("tasks", filters={"job_id": job_id})
                for task in tasks:
                    if task.get("status") in ("assigned", "running"):
                        mem_db.update(
                            "tasks",
                            {"status": "completed", "completed_at": now},
                            filters={"id": task["id"]},
                        )
            else:
                # Create next round
                active_tasks = [
                    t
                    for t in mem_db.select("tasks", filters={"job_id": job_id})
                    if t.get("status") in ("assigned", "running")
                ]
                now = datetime.now(timezone.utc).isoformat()
                mem_db.insert(
                    "training_rounds",
                    {
                        "job_id": job_id,
                        "round_number": next_round,
                        "status": "in_progress",
                        "active_worker_count": len(active_tasks),
                        "submitted_count": 0,
                        "started_at": now,
                    },
                )

        agg_patch = patch.object(main_mod, "aggregate_round", side_effect=mock_aggregate_round)
        patches.append(agg_patch)

        for p in patches:
            p.start()

        try:
            self._run_test(mem_db)
        finally:
            for p in patches:
                p.stop()

    def _run_test(self, mem_db: InMemoryDB):
        from fastapi.testclient import TestClient
        from coordinator.main import app

        client = TestClient(app, raise_server_exceptions=False)

        # Step 1: Register 2 nodes
        node1_id, _, headers1 = _register_node(client, "collab-worker-001")
        node2_id, _, headers2 = _register_node(client, "collab-worker-002")

        # Step 2: Submit job (shard_count=2, epochs=2)
        job_id = _submit_job(client, shard_count=2, epochs=2)

        # Verify job is queued with 2 tasks
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "queued"
        assert len(job_detail["tasks"]) == 2

        # Step 3: Both workers poll → get tasks assigned
        task1_id, _ = _poll_and_start(client, headers1)
        task2_id, _ = _poll_and_start(client, headers2)

        # Verify job is now running
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "running"

        # Step 4: Round 0 — both workers submit gradients
        resp1 = _submit_gradient(client, job_id, task1_id, 0, headers1, 0.5, 0.7)
        assert resp1.status_code == 200
        assert resp1.json()["barrier_met"] is False  # First worker, barrier not met

        resp2 = _submit_gradient(client, job_id, task2_id, 0, headers2, 0.4, 0.75)
        assert resp2.status_code == 200
        assert resp2.json()["barrier_met"] is True  # Second worker, barrier met

        # Verify round 0 completed and round 1 created
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["current_round"] == 1
        assert job_detail["status"] == "running"

        # Step 5: Round 1 — both workers submit gradients
        resp1 = _submit_gradient(client, job_id, task1_id, 1, headers1, 0.3, 0.85)
        assert resp1.status_code == 200
        assert resp1.json()["barrier_met"] is False

        resp2 = _submit_gradient(client, job_id, task2_id, 1, headers2, 0.25, 0.88)
        assert resp2.status_code == 200
        assert resp2.json()["barrier_met"] is True

        # Step 6: Verify job is completed
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "completed"
        assert job_detail["aggregated_metrics"] is not None
        assert "per_round" in job_detail["aggregated_metrics"]
        assert len(job_detail["aggregated_metrics"]["per_round"]) == 2
        assert job_detail["completed_at"] is not None

        # Verify training rounds
        training_rounds = job_detail.get("training_rounds", [])
        assert len(training_rounds) == 2
        for tr in training_rounds:
            assert tr["status"] == "completed"

        # Verify tasks are completed
        for task in job_detail["tasks"]:
            assert task["status"] == "completed"

        # Verify results endpoint
        results = client.get(f"/api/jobs/{job_id}/results").json()
        assert results["status"] == "completed"
        assert results["aggregated_metrics"] is not None

        print("\n" + "=" * 60)
        print("50.1 FULL COLLABORATIVE LIFECYCLE: ALL CHECKS PASSED")
        print("=" * 60)


# ---------------------------------------------------------------------------
# 50.2 Worker failure mid-training
# ---------------------------------------------------------------------------


class TestWorkerFailureMidTraining:
    """Integration test: Worker failure mid-training.

    Register 3 nodes → submit job → start training → one worker fails
    at round 1 → barrier adjusts → remaining 2 workers continue →
    job completes.

    Requirements: 6.5, 14.1, 14.2
    """

    def test_worker_failure_mid_training(self):
        os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
        os.environ.setdefault("SUPABASE_KEY", "fake-key")

        mem_db = InMemoryDB()
        patches = _setup_patches(mem_db)

        import coordinator.main as main_mod

        def mock_aggregate_round(job_id: str, round_number: int):
            """Simulate aggregation with worker failure handling."""
            jobs = mem_db.select("jobs", filters={"id": job_id})
            if not jobs:
                return
            job = jobs[0]
            total_rounds = job.get("total_rounds", 3)
            next_round = round_number + 1

            # Mark current round as completed
            rounds = mem_db.select(
                "training_rounds",
                filters={"job_id": job_id, "round_number": round_number},
            )
            if rounds:
                now = datetime.now(timezone.utc).isoformat()
                mem_db.update(
                    "training_rounds",
                    {
                        "status": "completed",
                        "completed_at": now,
                        "global_loss": 0.5 - 0.1 * round_number,
                        "global_accuracy": 0.6 + 0.1 * round_number,
                    },
                    filters={"id": rounds[0]["id"]},
                )

            # Advance job's current_round
            mem_db.update("jobs", {"current_round": next_round}, filters={"id": job_id})

            if next_round >= total_rounds:
                # Complete the job
                now = datetime.now(timezone.utc).isoformat()
                all_rounds = mem_db.select("training_rounds", filters={"job_id": job_id})
                completed = [r for r in all_rounds if r.get("status") == "completed"]
                per_round = []
                for r in sorted(completed, key=lambda x: x.get("round_number", 0)):
                    per_round.append({
                        "round_number": r.get("round_number"),
                        "global_loss": r.get("global_loss"),
                        "global_accuracy": r.get("global_accuracy"),
                    })

                aggregated_metrics = {
                    "per_round": per_round,
                    "mean_loss": sum(
                        r["global_loss"] for r in per_round if r["global_loss"] is not None
                    ) / max(len(per_round), 1),
                    "mean_accuracy": sum(
                        r["global_accuracy"] for r in per_round if r["global_accuracy"] is not None
                    ) / max(len(per_round), 1),
                    "per_worker": [],
                }

                mem_db.update(
                    "jobs",
                    {
                        "status": "completed",
                        "aggregated_metrics": aggregated_metrics,
                        "completed_at": now,
                        "global_model_path": f"checkpoints/{job_id}/round_{round_number}.pt",
                    },
                    filters={"id": job_id},
                )

                tasks = mem_db.select("tasks", filters={"job_id": job_id})
                for task in tasks:
                    if task.get("status") in ("assigned", "running"):
                        mem_db.update(
                            "tasks",
                            {"status": "completed", "completed_at": now},
                            filters={"id": task["id"]},
                        )
            else:
                # Create next round with current active workers
                active_tasks = [
                    t
                    for t in mem_db.select("tasks", filters={"job_id": job_id})
                    if t.get("status") in ("assigned", "running")
                ]
                now = datetime.now(timezone.utc).isoformat()
                mem_db.insert(
                    "training_rounds",
                    {
                        "job_id": job_id,
                        "round_number": next_round,
                        "status": "in_progress",
                        "active_worker_count": len(active_tasks),
                        "submitted_count": 0,
                        "started_at": now,
                    },
                )

        agg_patch = patch.object(main_mod, "aggregate_round", side_effect=mock_aggregate_round)
        patches.append(agg_patch)

        for p in patches:
            p.start()

        try:
            self._run_test(mem_db)
        finally:
            for p in patches:
                p.stop()

    def _run_test(self, mem_db: InMemoryDB):
        from fastapi.testclient import TestClient
        from coordinator.main import app

        client = TestClient(app, raise_server_exceptions=False)

        # Step 1: Register 3 nodes
        node1_id, _, headers1 = _register_node(client, "fail-worker-001")
        node2_id, _, headers2 = _register_node(client, "fail-worker-002")
        node3_id, _, headers3 = _register_node(client, "fail-worker-003")

        # Step 2: Submit job (shard_count=3, epochs=3)
        job_id = _submit_job(client, shard_count=3, epochs=3)

        # Step 3: All 3 workers poll and start
        task1_id, _ = _poll_and_start(client, headers1)
        task2_id, _ = _poll_and_start(client, headers2)
        task3_id, _ = _poll_and_start(client, headers3)

        # Verify job is running
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "running"

        # Step 4: Round 0 — all 3 workers submit gradients
        resp1 = _submit_gradient(client, job_id, task1_id, 0, headers1)
        assert resp1.status_code == 200
        resp2 = _submit_gradient(client, job_id, task2_id, 0, headers2)
        assert resp2.status_code == 200
        resp3 = _submit_gradient(client, job_id, task3_id, 0, headers3)
        assert resp3.status_code == 200
        assert resp3.json()["barrier_met"] is True  # 3/3 submitted

        # Verify round 0 completed
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["current_round"] == 1

        # Step 5: Round 1 — worker 3 fails
        fail_resp = client.post(
            f"/api/tasks/{task3_id}/fail",
            headers=headers3,
            json={"error_message": "GPU OOM at round 1"},
        )
        assert fail_resp.status_code == 200

        # The fail_task endpoint marks the task as failed and calls
        # check_job_failure. In the real system, the heartbeat monitor
        # would also call barrier.remove_worker() to adjust the barrier.
        # Since the heartbeat monitor is mocked out, we simulate the
        # barrier adjustment that would happen when the monitor detects
        # the offline node (Req 14.2: barrier adjusts for remaining workers).
        from coordinator.barrier import remove_worker
        remove_worker(job_id, task3_id)

        # Verify the barrier was adjusted — the round 1 record should have
        # active_worker_count decremented.
        # The job should still be running since 2 workers remain.
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "running"

        # Step 6: Round 1 — remaining 2 workers submit
        resp1 = _submit_gradient(client, job_id, task1_id, 1, headers1)
        assert resp1.status_code == 200
        resp2 = _submit_gradient(client, job_id, task2_id, 1, headers2)
        assert resp2.status_code == 200
        assert resp2.json()["barrier_met"] is True  # 2/2 submitted (adjusted barrier)

        # Verify round 1 completed
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["current_round"] == 2

        # Step 7: Round 2 — remaining 2 workers submit
        resp1 = _submit_gradient(client, job_id, task1_id, 2, headers1)
        assert resp1.status_code == 200
        resp2 = _submit_gradient(client, job_id, task2_id, 2, headers2)
        assert resp2.status_code == 200
        assert resp2.json()["barrier_met"] is True

        # Step 8: Verify job completed
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "completed"
        assert job_detail["aggregated_metrics"] is not None

        # Verify worker 3's task is failed
        task3_info = next(
            (t for t in job_detail["tasks"] if t["id"] == task3_id), None
        )
        assert task3_info is not None
        assert task3_info["status"] == "failed"
        assert task3_info["error_message"] == "GPU OOM at round 1"

        print("\n" + "=" * 60)
        print("50.2 WORKER FAILURE MID-TRAINING: ALL CHECKS PASSED")
        print("=" * 60)


# ---------------------------------------------------------------------------
# 50.3 All workers fail
# ---------------------------------------------------------------------------


class TestAllWorkersFail:
    """Integration test: All workers fail.

    Register 2 nodes → submit job → both workers fail → job marked
    failed with partial checkpoint.

    Requirements: 6.4, 14.3
    """

    def test_all_workers_fail(self):
        os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
        os.environ.setdefault("SUPABASE_KEY", "fake-key")

        mem_db = InMemoryDB()
        patches = _setup_patches(mem_db)

        import coordinator.main as main_mod
        import coordinator.param_server as ps_mod

        def mock_aggregate_round(job_id: str, round_number: int):
            """Simulate aggregation for round 0."""
            jobs = mem_db.select("jobs", filters={"id": job_id})
            if not jobs:
                return
            job = jobs[0]
            total_rounds = job.get("total_rounds", 3)
            next_round = round_number + 1

            # Mark current round as completed
            rounds = mem_db.select(
                "training_rounds",
                filters={"job_id": job_id, "round_number": round_number},
            )
            if rounds:
                now = datetime.now(timezone.utc).isoformat()
                mem_db.update(
                    "training_rounds",
                    {
                        "status": "completed",
                        "completed_at": now,
                        "global_loss": 0.5,
                        "global_accuracy": 0.6,
                    },
                    filters={"id": rounds[0]["id"]},
                )

            # Advance job's current_round
            mem_db.update("jobs", {"current_round": next_round}, filters={"id": job_id})

            if next_round < total_rounds:
                active_tasks = [
                    t
                    for t in mem_db.select("tasks", filters={"job_id": job_id})
                    if t.get("status") in ("assigned", "running")
                ]
                now = datetime.now(timezone.utc).isoformat()
                mem_db.insert(
                    "training_rounds",
                    {
                        "job_id": job_id,
                        "round_number": next_round,
                        "status": "in_progress",
                        "active_worker_count": len(active_tasks),
                        "submitted_count": 0,
                        "started_at": now,
                    },
                )

        agg_patch = patch.object(main_mod, "aggregate_round", side_effect=mock_aggregate_round)

        # Mock store_checkpoint for partial checkpoint on failure
        def mock_store_checkpoint(job_id: str, round_number: int) -> str:
            path = f"checkpoints/{job_id}/round_{round_number}.pt"
            mem_db.insert(
                "artifacts",
                {
                    "job_id": job_id,
                    "task_id": None,
                    "node_id": None,
                    "artifact_type": "checkpoint",
                    "storage_path": path,
                    "round_number": round_number,
                    "size_bytes": 1024,
                },
            )
            return path

        checkpoint_patch = patch.object(ps_mod, "store_checkpoint", side_effect=mock_store_checkpoint)

        patches.append(agg_patch)
        patches.append(checkpoint_patch)

        for p in patches:
            p.start()

        try:
            self._run_test(mem_db)
        finally:
            for p in patches:
                p.stop()

    def _run_test(self, mem_db: InMemoryDB):
        from fastapi.testclient import TestClient
        from coordinator.main import app

        client = TestClient(app, raise_server_exceptions=False)

        # Step 1: Register 2 nodes
        node1_id, _, headers1 = _register_node(client, "allfail-worker-001")
        node2_id, _, headers2 = _register_node(client, "allfail-worker-002")

        # Step 2: Submit job (shard_count=2, epochs=3)
        job_id = _submit_job(client, shard_count=2, epochs=3)

        # Step 3: Both workers poll and start
        task1_id, _ = _poll_and_start(client, headers1)
        task2_id, _ = _poll_and_start(client, headers2)

        # Verify job is running
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "running"

        # Step 4: Round 0 — both submit → aggregation
        resp1 = _submit_gradient(client, job_id, task1_id, 0, headers1)
        assert resp1.status_code == 200
        resp2 = _submit_gradient(client, job_id, task2_id, 0, headers2)
        assert resp2.status_code == 200
        assert resp2.json()["barrier_met"] is True

        # Verify round 0 completed
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["current_round"] == 1
        assert job_detail["status"] == "running"

        # Step 5: Round 1 — both workers fail
        fail_resp1 = client.post(
            f"/api/tasks/{task1_id}/fail",
            headers=headers1,
            json={"error_message": "Worker 1 crashed"},
        )
        assert fail_resp1.status_code == 200

        # After first worker fails, job should still be running (1 active worker)
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        # Job might still be running or might be failed depending on check_job_failure
        # With 1 active worker remaining, it should still be running
        # But let's check — if the second worker hasn't failed yet, there's still an active task

        fail_resp2 = client.post(
            f"/api/tasks/{task2_id}/fail",
            headers=headers2,
            json={"error_message": "Worker 2 crashed"},
        )
        assert fail_resp2.status_code == 200

        # Step 6: Verify job is failed
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "failed"
        assert job_detail["error_summary"] is not None
        failed_tasks = job_detail["error_summary"]["failed_tasks"]
        assert len(failed_tasks) == 2

        error_messages = {ft["error_message"] for ft in failed_tasks}
        assert "Worker 1 crashed" in error_messages
        assert "Worker 2 crashed" in error_messages

        # Verify partial checkpoint was stored (round 0 was completed)
        # The check_job_failure in aggregator.py should have stored a partial checkpoint
        artifacts = mem_db.select("artifacts", filters={"job_id": job_id})
        # There should be at least one checkpoint artifact from the partial save
        checkpoint_artifacts = [a for a in artifacts if a.get("artifact_type") == "checkpoint"]
        assert len(checkpoint_artifacts) >= 1, "Expected a partial checkpoint artifact"

        print("\n" + "=" * 60)
        print("50.3 ALL WORKERS FAIL: ALL CHECKS PASSED")
        print("=" * 60)


# ---------------------------------------------------------------------------
# 50.4 Round validation
# ---------------------------------------------------------------------------


class TestRoundValidation:
    """Integration test: Round validation.

    Worker submits gradient for wrong round → 409 rejected → Worker
    re-syncs and submits for correct round → 200.

    Requirements: 13.3
    """

    def test_round_validation(self):
        os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
        os.environ.setdefault("SUPABASE_KEY", "fake-key")

        mem_db = InMemoryDB()
        patches = _setup_patches(mem_db)

        for p in patches:
            p.start()

        try:
            self._run_test(mem_db)
        finally:
            for p in patches:
                p.stop()

    def _run_test(self, mem_db: InMemoryDB):
        from fastapi.testclient import TestClient
        from coordinator.main import app

        client = TestClient(app, raise_server_exceptions=False)

        # Step 1: Register 1 node
        node1_id, _, headers1 = _register_node(client, "round-val-worker-001")

        # Step 2: Submit job (shard_count=1, epochs=2)
        job_id = _submit_job(client, shard_count=1, epochs=2)

        # Step 3: Worker polls and starts
        task1_id, _ = _poll_and_start(client, headers1)

        # Verify job is running and current_round is 0
        job_detail = client.get(f"/api/jobs/{job_id}").json()
        assert job_detail["status"] == "running"
        assert job_detail["current_round"] == 0

        # Step 4: Worker submits gradient for wrong round (5) → 409
        wrong_resp = _submit_gradient(client, job_id, task1_id, 5, headers1)
        assert wrong_resp.status_code == 409
        assert "round" in wrong_resp.json()["detail"].lower()

        # Step 5: Worker re-syncs and submits for correct round (0) → 200
        correct_resp = _submit_gradient(client, job_id, task1_id, 0, headers1)
        assert correct_resp.status_code == 200
        assert correct_resp.json()["status"] == "ok"

        print("\n" + "=" * 60)
        print("50.4 ROUND VALIDATION: ALL CHECKS PASSED")
        print("=" * 60)
