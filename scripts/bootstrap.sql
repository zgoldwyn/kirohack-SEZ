-- ==========================================================================
-- Group ML Trainer — Full Database Schema
-- ==========================================================================
-- Run this against a Supabase Postgres instance to create all required
-- tables, indexes, and the storage bucket used by the platform.
--
-- Prerequisites:
--   * The `uuid-ossp` extension (enabled by default on Supabase).
--   * Supabase Storage enabled for the project.
-- ==========================================================================

-- --------------------------------------------------------------------------
-- 1. nodes
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nodes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id         TEXT NOT NULL UNIQUE,
    hostname        TEXT NOT NULL,
    cpu_cores       INTEGER NOT NULL CHECK (cpu_cores > 0),
    gpu_model       TEXT,
    vram_mb         INTEGER,
    ram_mb          INTEGER NOT NULL CHECK (ram_mb > 0),
    disk_mb         INTEGER NOT NULL CHECK (disk_mb > 0),
    os              TEXT NOT NULL,
    python_version  TEXT NOT NULL,
    pytorch_version TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'idle'
                        CHECK (status IN ('idle', 'busy', 'offline')),
    last_heartbeat  TIMESTAMPTZ DEFAULT now(),
    auth_token_hash TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes (status);
CREATE INDEX IF NOT EXISTS idx_nodes_node_id ON nodes (node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_auth_token_hash ON nodes (auth_token_hash);

-- --------------------------------------------------------------------------
-- 2. jobs
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name            TEXT,
    dataset_name        TEXT NOT NULL,
    model_type          TEXT NOT NULL,
    hyperparameters     JSONB NOT NULL DEFAULT '{}'::jsonb,
    shard_count         INTEGER NOT NULL CHECK (shard_count > 0),
    status              TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    aggregated_metrics  JSONB,
    error_summary       JSONB,
    created_at          TIMESTAMPTZ DEFAULT now(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);

-- --------------------------------------------------------------------------
-- 3. tasks
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id          UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    node_id         UUID REFERENCES nodes (id) ON DELETE SET NULL,
    shard_index     INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'assigned', 'running', 'completed', 'failed')),
    task_config     JSONB NOT NULL DEFAULT '{}'::jsonb,
    checkpoint_path TEXT,
    error_message   TEXT,
    assigned_at     TIMESTAMPTZ,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks (job_id);
CREATE INDEX IF NOT EXISTS idx_tasks_node_id ON tasks (node_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);

-- --------------------------------------------------------------------------
-- 4. metrics
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metrics (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id      UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    task_id     UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    node_id     UUID REFERENCES nodes (id) ON DELETE SET NULL,
    epoch       INTEGER NOT NULL CHECK (epoch >= 0),
    loss        NUMERIC,
    accuracy    NUMERIC,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_metrics_task_id ON metrics (task_id);
CREATE INDEX IF NOT EXISTS idx_metrics_job_id ON metrics (job_id);

-- --------------------------------------------------------------------------
-- 5. artifacts
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS artifacts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id          UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    task_id         UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    node_id         UUID REFERENCES nodes (id) ON DELETE SET NULL,
    artifact_type   TEXT NOT NULL
                        CHECK (artifact_type IN ('checkpoint', 'log', 'output')),
    storage_path    TEXT NOT NULL,
    epoch           INTEGER,
    size_bytes      BIGINT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON artifacts (job_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task_id ON artifacts (task_id);

-- --------------------------------------------------------------------------
-- 6. Storage bucket for checkpoints
-- --------------------------------------------------------------------------
-- Supabase Storage buckets are managed via the dashboard or the storage API,
-- not via raw SQL.  The verify_schema.py script checks for the bucket
-- programmatically.  If you need to create it manually:
--
--   INSERT INTO storage.buckets (id, name, public)
--   VALUES ('checkpoints', 'checkpoints', false)
--   ON CONFLICT (id) DO NOTHING;
--
-- The path convention is: {job_id}/{task_id}/final.pt
