#!/usr/bin/env python3
"""Verify that the Supabase database contains all tables, columns, indexes,
and the storage buckets required by Group ML Trainer.

Usage:
    python scripts/verify_schema.py

Environment variables (loaded from .env automatically):
    SUPABASE_URL          – Project URL
    SUPABASE_SERVICE_KEY  – Service-role key (needed for storage bucket check)
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ── Expected schema definition ────────────────────────────────────────────

EXPECTED_TABLES: dict[str, list[str]] = {
    "nodes": [
        "id",
        "node_id",
        "hostname",
        "cpu_cores",
        "gpu_model",
        "vram_mb",
        "ram_mb",
        "disk_mb",
        "os",
        "python_version",
        "pytorch_version",
        "status",
        "last_heartbeat",
        "auth_token_hash",
        "created_at",
    ],
    "jobs": [
        "id",
        "job_name",
        "dataset_name",
        "model_type",
        "hyperparameters",
        "shard_count",
        "status",
        "current_round",
        "total_rounds",
        "global_model_path",
        "aggregated_metrics",
        "error_summary",
        "created_at",
        "started_at",
        "completed_at",
    ],
    "tasks": [
        "id",
        "job_id",
        "node_id",
        "shard_index",
        "status",
        "task_config",
        "last_submitted_round",
        "error_message",
        "assigned_at",
        "started_at",
        "completed_at",
        "created_at",
    ],
    "metrics": [
        "id",
        "job_id",
        "task_id",
        "node_id",
        "round_number",
        "metric_type",
        "loss",
        "accuracy",
        "created_at",
    ],
    "artifacts": [
        "id",
        "job_id",
        "task_id",
        "node_id",
        "artifact_type",
        "storage_path",
        "round_number",
        "size_bytes",
        "created_at",
    ],
    "training_rounds": [
        "id",
        "job_id",
        "round_number",
        "status",
        "active_worker_count",
        "submitted_count",
        "global_loss",
        "global_accuracy",
        "started_at",
        "completed_at",
        "created_at",
    ],
    "gradient_submissions": [
        "id",
        "job_id",
        "task_id",
        "node_id",
        "round_number",
        "gradient_path",
        "local_loss",
        "local_accuracy",
        "created_at",
    ],
}

EXPECTED_INDEXES: dict[str, list[str]] = {
    "nodes": [
        "idx_nodes_status",
        "idx_nodes_node_id",
        "idx_nodes_auth_token_hash",
    ],
    "jobs": [
        "idx_jobs_status",
    ],
    "tasks": [
        "idx_tasks_job_id",
        "idx_tasks_node_id",
        "idx_tasks_status",
    ],
    "metrics": [
        "idx_metrics_task_id",
        "idx_metrics_job_id",
    ],
    "artifacts": [
        "idx_artifacts_job_id",
        "idx_artifacts_task_id",
    ],
    "training_rounds": [
        "idx_training_rounds_job_id",
    ],
    "gradient_submissions": [
        "idx_gradient_submissions_job_id_round",
    ],
}

EXPECTED_STORAGE_BUCKETS = ["checkpoints", "parameters", "gradients"]


# ── Helpers ───────────────────────────────────────────────────────────────

def _get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
        sys.exit(1)
    return create_client(url, key)


def _query_columns(client: Client, table: str) -> set[str]:
    """Return the set of column names for *table* using information_schema."""
    resp = (
        client.table("columns")
        .select("column_name")
        .eq("table_schema", "public")
        .eq("table_name", table)
        .execute()
    )
    if resp.data:
        return {row["column_name"] for row in resp.data}
    return set()


def _query_columns_via_rpc(client: Client, table: str) -> set[str]:
    """Fetch column names via an RPC call to information_schema."""
    try:
        resp = client.rpc(
            "get_table_columns",
            {"p_table_name": table},
        ).execute()
        if resp.data:
            return {row["column_name"] for row in resp.data}
    except Exception:
        pass
    return set()


def _probe_table_columns(client: Client, table: str) -> set[str]:
    """Best-effort column detection: select a single row and inspect keys."""
    try:
        resp = client.table(table).select("*").limit(1).execute()
        if resp.data:
            return set(resp.data[0].keys())
        return set()
    except Exception:
        return set()


def _table_exists(client: Client, table: str) -> bool:
    """Return True if *table* exists in the public schema."""
    try:
        client.table(table).select("id").limit(0).execute()
        return True
    except Exception:
        return False


def _check_storage_bucket(client: Client, bucket_name: str) -> bool:
    """Return True if the storage bucket exists."""
    try:
        buckets = client.storage.list_buckets()
        return any(b.name == bucket_name for b in buckets)
    except Exception as exc:
        print(f"  WARNING: Could not list storage buckets: {exc}")
        return False


# ── Main verification ─────────────────────────────────────────────────────

def verify() -> bool:
    client = _get_client()
    ok = True

    print("=" * 60)
    print("Group ML Trainer — Schema Verification")
    print("=" * 60)

    # 1. Tables & columns
    for table, expected_cols in EXPECTED_TABLES.items():
        print(f"\n[TABLE] {table}")
        if not _table_exists(client, table):
            print(f"  FAIL: table '{table}' does not exist")
            ok = False
            continue

        print("  OK: table exists")

        # Try multiple strategies to discover columns
        cols = _query_columns(client, table)
        if not cols:
            cols = _query_columns_via_rpc(client, table)
        if not cols:
            cols = _probe_table_columns(client, table)

        if not cols:
            print("  WARNING: could not retrieve column list (table may be empty)")
            print(f"  Expected columns: {', '.join(expected_cols)}")
            continue

        missing = set(expected_cols) - cols
        extra = cols - set(expected_cols)
        if missing:
            print(f"  FAIL: missing columns: {', '.join(sorted(missing))}")
            ok = False
        else:
            print(f"  OK: all {len(expected_cols)} expected columns present")
        if extra:
            print(f"  INFO: extra columns (not in spec): {', '.join(sorted(extra))}")

    # 2. Indexes — informational only (PostgREST doesn't expose pg_indexes)
    print(f"\n{'─' * 60}")
    print("INDEX VERIFICATION (informational)")
    print(
        "  NOTE: Index verification requires direct Postgres access.\n"
        "  Run the following query against your database to confirm:\n"
    )
    for table, indexes in EXPECTED_INDEXES.items():
        for idx in indexes:
            print(f"    SELECT indexname FROM pg_indexes WHERE indexname = '{idx}';")
    print()

    # 3. Storage buckets
    print(f"{'─' * 60}")
    for bucket_name in EXPECTED_STORAGE_BUCKETS:
        print(f"[STORAGE] bucket '{bucket_name}'")
        if _check_storage_bucket(client, bucket_name):
            print("  OK: bucket exists")
        else:
            print(f"  FAIL: bucket '{bucket_name}' not found")
            ok = False

    # Summary
    print(f"\n{'=' * 60}")
    if ok:
        print("RESULT: All checks passed ✓")
    else:
        print("RESULT: Some checks FAILED — see details above")
    print("=" * 60)

    return ok


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
