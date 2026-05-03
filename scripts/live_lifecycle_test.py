#!/usr/bin/env python3
"""Live database lifecycle test.

Runs the full task lifecycle against the real Supabase database:
1. Register a worker node
2. Submit a small MNIST job (shard_count=1, 1 epoch)
3. Poll for the task
4. Start the task
5. Report metrics
6. Complete the task
7. Verify the job is marked "completed" with aggregated metrics
8. Clean up all test data

Usage:
    python scripts/live_lifecycle_test.py

Requires SUPABASE_URL and SUPABASE_KEY in .env
"""

from __future__ import annotations

import os
import sys
import uuid

# Add project root to path so we can import coordinator modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Validate env before importing coordinator modules
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
if not url or not key:
    print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
    sys.exit(1)

from coordinator import db
from coordinator.auth import generate_token, hash_token

# Track IDs for cleanup
created_node_id: str | None = None
created_job_id: str | None = None


def cleanup():
    """Remove all test data we created, in dependency order."""
    print("\n--- Cleanup ---")
    try:
        if created_job_id:
            # Delete artifacts, metrics, tasks (all reference job_id)
            for table in ("artifacts", "metrics", "tasks"):
                try:
                    db.delete(table, filters={"job_id": created_job_id})
                    print(f"  Deleted {table} for job {created_job_id[:8]}…")
                except Exception as e:
                    print(f"  Warning: could not delete {table}: {e}")
            # Delete the job
            try:
                db.delete("jobs", filters={"id": created_job_id})
                print(f"  Deleted job {created_job_id[:8]}…")
            except Exception as e:
                print(f"  Warning: could not delete job: {e}")

        if created_node_id:
            try:
                db.delete("nodes", filters={"id": created_node_id})
                print(f"  Deleted node {created_node_id[:8]}…")
            except Exception as e:
                print(f"  Warning: could not delete node: {e}")
    except Exception as e:
        print(f"  Cleanup error: {e}")
    print("  Done.")


def run():
    global created_node_id, created_job_id

    unique = uuid.uuid4().hex[:8]
    print("=" * 60)
    print("Live Database Lifecycle Test")
    print("=" * 60)
    print(f"Supabase URL: {url}")
    print(f"Test ID: {unique}")
    print()

    # ==================================================================
    # Step 1: Register a worker node
    # ==================================================================
    print("[1/7] Registering worker node...")
    token = generate_token()
    token_hash = hash_token(token)

    node_record = db.insert("nodes", {
        "node_id": f"live-test-{unique}",
        "hostname": "live-test-host",
        "cpu_cores": 4,
        "gpu_model": None,
        "vram_mb": None,
        "ram_mb": 8192,
        "disk_mb": 50000,
        "os": "test",
        "python_version": "3.11.0",
        "pytorch_version": "2.6.0",
        "status": "idle",
        "auth_token_hash": token_hash,
    })
    created_node_id = node_record["id"]
    print(f"  ✓ Node registered: {created_node_id[:8]}… (status=idle)")

    # Verify we can read it back
    nodes = db.select("nodes", filters={"id": created_node_id})
    assert len(nodes) == 1, f"Expected 1 node, got {len(nodes)}"
    assert nodes[0]["status"] == "idle"
    print(f"  ✓ Node read back: status={nodes[0]['status']}")

    # ==================================================================
    # Step 2: Submit a small MNIST job
    # ==================================================================
    print("\n[2/7] Submitting MNIST job (1 shard, 1 epoch)...")
    job_record = db.insert("jobs", {
        "job_name": f"live-test-{unique}",
        "dataset_name": "MNIST",
        "model_type": "MLP",
        "hyperparameters": {
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 64,
            "hidden_layers": [64, 32],
            "activation": "relu",
        },
        "shard_count": 1,
        "status": "queued",
    })
    created_job_id = job_record["id"]
    print(f"  ✓ Job created: {created_job_id[:8]}… (status=queued)")

    # Create a task for this job
    task_config = {
        "task_id": "placeholder",
        "job_id": created_job_id,
        "dataset_name": "MNIST",
        "model_type": "MLP",
        "hyperparameters": {
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 64,
            "hidden_layers": [64, 32],
            "activation": "relu",
        },
        "shard_index": 0,
        "shard_count": 1,
    }
    task_record = db.insert("tasks", {
        "job_id": created_job_id,
        "shard_index": 0,
        "status": "queued",
        "task_config": task_config,
    })
    task_id = task_record["id"]

    # Update task_config with the real task_id
    task_config["task_id"] = task_id
    db.update("tasks", {"task_config": task_config}, filters={"id": task_id})
    print(f"  ✓ Task created: {task_id[:8]}… (status=queued)")

    # ==================================================================
    # Step 3: Poll — simulate assigning the task to our node
    # ==================================================================
    print("\n[3/7] Polling for task (assigning to our node)...")
    queued_tasks = db.select("tasks", filters={"job_id": created_job_id, "status": "queued"})
    assert len(queued_tasks) == 1, f"Expected 1 queued task, got {len(queued_tasks)}"

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    db.update("tasks", {
        "status": "assigned",
        "node_id": created_node_id,
        "assigned_at": now,
    }, filters={"id": task_id})

    db.update("nodes", {"status": "busy"}, filters={"id": created_node_id})
    db.update("jobs", {"status": "running", "started_at": now}, filters={"id": created_job_id})

    # Verify
    task_rows = db.select("tasks", filters={"id": task_id})
    assert task_rows[0]["status"] == "assigned"
    assert task_rows[0]["node_id"] == created_node_id
    print(f"  ✓ Task assigned to node (status=assigned)")

    job_rows = db.select("jobs", filters={"id": created_job_id})
    assert job_rows[0]["status"] == "running"
    print(f"  ✓ Job status: running")

    # ==================================================================
    # Step 4: Start the task
    # ==================================================================
    print("\n[4/7] Starting task...")
    now = datetime.now(timezone.utc).isoformat()
    db.update("tasks", {
        "status": "running",
        "started_at": now,
    }, filters={"id": task_id})

    task_rows = db.select("tasks", filters={"id": task_id})
    assert task_rows[0]["status"] == "running"
    print(f"  ✓ Task status: running")

    # ==================================================================
    # Step 5: Report metrics (1 epoch)
    # ==================================================================
    print("\n[5/7] Reporting metrics (epoch 0)...")
    metrics_record = db.insert("metrics", {
        "job_id": created_job_id,
        "task_id": task_id,
        "node_id": created_node_id,
        "epoch": 0,
        "loss": 0.4523,
        "accuracy": 0.8765,
    })
    print(f"  ✓ Metrics recorded: loss=0.4523, accuracy=0.8765")

    # Verify metrics are readable
    metrics_rows = db.select("metrics", filters={"task_id": task_id})
    assert len(metrics_rows) == 1
    assert float(metrics_rows[0]["loss"]) == 0.4523
    assert float(metrics_rows[0]["accuracy"]) == 0.8765
    print(f"  ✓ Metrics read back successfully")

    # ==================================================================
    # Step 6: Complete the task
    # ==================================================================
    print("\n[6/7] Completing task...")
    now = datetime.now(timezone.utc).isoformat()
    checkpoint_path = f"{created_job_id}/{task_id}/final.pt"

    db.update("tasks", {
        "status": "completed",
        "checkpoint_path": checkpoint_path,
        "completed_at": now,
    }, filters={"id": task_id})

    # Insert artifact record
    db.insert("artifacts", {
        "job_id": created_job_id,
        "task_id": task_id,
        "node_id": created_node_id,
        "artifact_type": "checkpoint",
        "storage_path": checkpoint_path,
    })

    # Set node back to idle
    db.update("nodes", {"status": "idle"}, filters={"id": created_node_id})

    # Mark job completed with aggregated metrics
    aggregated_metrics = {
        "mean_loss": 0.4523,
        "mean_accuracy": 0.8765,
        "per_node": [{
            "node_id": created_node_id,
            "task_id": task_id,
            "loss": 0.4523,
            "accuracy": 0.8765,
        }],
    }
    db.update("jobs", {
        "status": "completed",
        "aggregated_metrics": aggregated_metrics,
        "completed_at": now,
    }, filters={"id": created_job_id})

    task_rows = db.select("tasks", filters={"id": task_id})
    assert task_rows[0]["status"] == "completed"
    assert task_rows[0]["checkpoint_path"] == checkpoint_path
    print(f"  ✓ Task status: completed")
    print(f"  ✓ Checkpoint path: {checkpoint_path}")

    # ==================================================================
    # Step 7: Verify everything is completed
    # ==================================================================
    print("\n[7/7] Verifying final state...")

    # Job is completed
    job_rows = db.select("jobs", filters={"id": created_job_id})
    assert len(job_rows) == 1
    job = job_rows[0]
    assert job["status"] == "completed", f"Expected completed, got {job['status']}"
    print(f"  ✓ Job status: {job['status']}")

    # Aggregated metrics
    agg = job["aggregated_metrics"]
    assert agg is not None, "aggregated_metrics should be set"
    assert abs(agg["mean_loss"] - 0.4523) < 0.001
    assert abs(agg["mean_accuracy"] - 0.8765) < 0.001
    print(f"  ✓ Aggregated metrics: mean_loss={agg['mean_loss']}, mean_accuracy={agg['mean_accuracy']}")

    # Per-node breakdown
    assert len(agg["per_node"]) == 1
    pn = agg["per_node"][0]
    assert pn["node_id"] == created_node_id
    print(f"  ✓ Per-node breakdown present")

    # Artifact exists
    artifacts = db.select("artifacts", filters={"job_id": created_job_id})
    assert len(artifacts) == 1
    assert artifacts[0]["artifact_type"] == "checkpoint"
    assert artifacts[0]["storage_path"] == checkpoint_path
    print(f"  ✓ Artifact record: {artifacts[0]['storage_path']}")

    # Node is back to idle
    node_rows = db.select("nodes", filters={"id": created_node_id})
    assert node_rows[0]["status"] == "idle"
    print(f"  ✓ Node status: idle (back to available)")

    # completed_at is set
    assert job.get("completed_at") is not None
    print(f"  ✓ Job completed_at: {job['completed_at']}")

    # ==================================================================
    # Done
    # ==================================================================
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)
    print(f"  Node:       live-test-{unique} ({created_node_id[:8]}…)")
    print(f"  Job:        live-test-{unique} ({created_job_id[:8]}…)")
    print(f"  Task:       {task_id[:8]}…")
    print(f"  Dataset:    MNIST")
    print(f"  Model:      MLP")
    print(f"  Loss:       0.4523")
    print(f"  Accuracy:   87.65%")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
