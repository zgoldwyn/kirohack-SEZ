# Implementation Plan: Group ML Trainer

## Overview

This plan implements a distributed ML task orchestration platform with three components: a Coordinator (FastAPI backend), Workers (Python agents), and a Dashboard (Next.js frontend). Tasks are ordered for incremental buildability — each task is implementable given the tasks before it. The Coordinator is built first (data layer → auth → endpoints → scheduling → aggregation), then the Worker client, then the Dashboard.

Tasks 1–23 implemented the original independent-training scope and are preserved as completed. Tasks 24+ implement the collaborative distributed training rewrite using a Parameter Server pattern, where Workers compute gradients on local shards and the Coordinator aggregates them into a single Global_Model.

## Tasks

- [x] 1. Project structure and core dependencies
  - [x] 1.1 Create Coordinator project structure and install dependencies
    - Create `coordinator/` directory with `__init__.py`, `main.py` (FastAPI app entry), `requirements.txt`
    - Dependencies: fastapi, uvicorn, supabase-py, pydantic, python-dotenv
    - Set up `.env` loading for SUPABASE_URL, SUPABASE_KEY
    - _Requirements: 1.1, 3.1_

  - [x] 1.2 Create Worker project structure and install dependencies
    - Create `worker/` directory with `__init__.py`, `main.py`, `requirements.txt`
    - Dependencies: torch, torchvision, httpx, pydantic, python-dotenv (no supabase-py — worker uses plain HTTP for signed URL uploads)
    - _Requirements: 5.1_

  - [x] 1.3 Create Dashboard project structure
    - Initialize Next.js app in `dashboard/` with TypeScript, Tailwind CSS
    - Install SWR for data fetching
    - _Requirements: 9.1, 10.1_

  - [x] 1.4 Create shared constants and enums
    - Create `coordinator/constants.py` with enums/constants for:
      - Node statuses: `idle`, `busy`, `offline`
      - Job statuses: `queued`, `running`, `completed`, `failed`
      - Task statuses: `queued`, `assigned`, `running`, `completed`, `failed`
      - Supported datasets (core MVP): `MNIST`, `Fashion-MNIST`, `synthetic`
      - Supported model types: `MLP`
      - Artifact types: `checkpoint`, `log`, `output`
    - Use Python `enum.Enum` or string constants — import everywhere instead of hardcoding strings
    - _Requirements: 3.5, 3.6_

  - [x] 1.5 Create database schema verification script
    - Create `scripts/verify_schema.sql` (or `scripts/bootstrap.sql`) containing the full CREATE TABLE / CREATE INDEX statements for all 5 tables (nodes, jobs, tasks, metrics, artifacts) and the `checkpoints` storage bucket
    - Create `scripts/verify_schema.py` that connects to Supabase and verifies all required tables, columns, indexes, and the storage bucket exist
    - Document required environment setup in a `README.md` or `SETUP.md`
    - _Requirements: 1.1, 3.1_

- [x] 2. Database client and storage integration
  - [x] 2.1 Implement `coordinator/db.py` — Supabase client initialization and query helpers
    - Initialize Supabase client from environment variables
    - Create helper functions for common queries: insert, select, update with filters
    - Wrap Supabase errors into consistent application exceptions
    - _Requirements: 1.1, 3.1_

  - [x] 2.2 Implement `coordinator/storage.py` — Supabase Storage signed URL generation
    - Create function to generate time-limited signed upload URLs for the `checkpoints` bucket
    - Path convention: `{job_id}/{task_id}/final.pt`
    - Workers do not hold direct Supabase credentials; they use signed URLs
    - _Requirements: 7.1, 7.2_

- [x] 3. Pydantic models for API layer
  - [x] 3.1 Implement `coordinator/models.py` — Request and response models
    - Define `NodeRegistrationRequest`, `NodeRegistrationResponse`
    - Define `JobSubmissionRequest`, `JobSubmissionResponse`
    - Define `MetricsReportRequest`, `TaskCompleteRequest`, `TaskFailRequest`
    - Define `TaskPollResponse`, `AggregatedMetrics`
    - Define `JobConfig`, `HyperParameters`, `TaskConfig` internal models
    - Add field validators (gt=0 for cpu_cores, ram_mb, disk_mb, shard_count; ge=0 for epoch)
    - Use status enums/constants from `coordinator/constants.py` throughout
    - _Requirements: 1.1, 1.3, 3.1, 3.4, 12.1_

- [x] 4. Configuration parsing and validation
  - [x] 4.1 Implement `coordinator/config_parser.py` — Job config parsing and task config generation
    - Parse `JobSubmissionRequest` into structured `JobConfig` object
    - Validate dataset_name against supported set {MNIST, Fashion-MNIST, synthetic}
    - Validate model_type against supported set {MLP}
    - Serialize `JobConfig` into per-task `TaskConfig` payloads with shard_index
    - Implement resource requirement lookup by model type (e.g., MLP → min 512 MB RAM, no GPU required)
    - _Requirements: 3.2, 3.5, 3.6, 12.1, 12.2, 12.3, 12.4_

  - [ ]* 4.2 Write property tests for config validation (Properties 3, 5, 13, 14)
    - **Property 3: Job validation rejects unsupported dataset or model type**
    - **Property 5: Job validation rejects configs with missing required fields**
    - **Property 13: JobConfig serialization round-trip preserves data**
    - **Property 14: Config validation reports invalid field types and out-of-range values**
    - **Validates: Requirements 3.2, 3.4, 12.2, 12.3, 12.4**

- [x] 5. Auth module
  - [x] 5.1 Implement `coordinator/auth.py` — Token generation, hashing, and FastAPI dependency
    - Token generation using `secrets.token_urlsafe(32)`
    - SHA-256 hashing for storage
    - FastAPI dependency `get_current_node` that extracts token from `Authorization` header, hashes it, looks up in `nodes` table, returns node record or raises HTTP 401
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 5.2 Write property tests for auth (Properties 10, 11)
    - **Property 10: Auth token validation accepts valid tokens and rejects invalid ones**
    - **Property 11: Auth tokens are unique across all registered nodes**
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Node registration endpoint
  - [x] 7.1 Implement `POST /api/nodes/register` endpoint
    - Validate request body using `NodeRegistrationRequest` model
    - Check for duplicate `node_id` — return 409 if already registered
    - Generate auth token, hash it, store node record with status "idle"
    - Return `NodeRegistrationResponse` with `node_db_id` (database UUID) and `auth_token`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 7.2 Write property test for registration validation (Property 1)
    - **Property 1: Registration rejects requests with missing required fields**
    - **Validates: Requirements 1.3**

  - [ ]* 7.3 Write unit tests for node registration
    - Test successful registration returns token and sets status to "idle"
    - Test duplicate node_id returns 409
    - _Requirements: 1.1, 1.2, 1.4_

- [x] 8. Heartbeat endpoint and staleness monitor
  - [x] 8.1 Implement `POST /api/nodes/heartbeat` endpoint
    - Authenticate request using auth dependency
    - Update `last_heartbeat` timestamp for the authenticated node
    - If node was "offline", set status back to "idle"
    - _Requirements: 2.1, 2.3_

  - [x] 8.2 Implement `coordinator/heartbeat.py` — Background staleness monitor
    - Background task (asyncio) that runs every 10 seconds
    - Scans all nodes, marks those with `last_heartbeat` > 30 seconds ago as "offline"
    - If an offline node has tasks in "assigned" or "running" status, mark those tasks as "failed" with error "node went offline"
    - Check if failed tasks cause any jobs to transition to "failed" (no tasks queued/assigned/running remaining)
    - _Requirements: 2.2, 6.4_

  - [ ]* 8.3 Write property test for heartbeat staleness (Property 2)
    - **Property 2: Heartbeat staleness detection marks correct nodes offline**
    - **Validates: Requirements 2.2**

  - [ ]* 8.4 Write unit tests for heartbeat
    - Test offline node recovers to "idle" on heartbeat
    - Test staleness monitor marks stale nodes offline
    - Test offline node with running task → task marked failed
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 9. Job submission endpoint
  - [x] 9.1 Implement `POST /api/jobs` endpoint
    - Validate request using `JobSubmissionRequest` model
    - Validate dataset_name and model_type using config_parser
    - Count idle nodes; reject if shard_count > idle_node_count (HTTP 400)
    - Create job record with status "queued"
    - Trigger task creation via scheduler
    - Return `JobSubmissionResponse` with `job_id`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 9.2 Write property test for shard count validation (Property 4)
    - **Property 4: Job submission rejects shard count exceeding idle node count**
    - **Validates: Requirements 3.3**

  - [ ]* 9.3 Write unit tests for job submission
    - Test valid job creates record with "queued" status
    - Test unsupported dataset returns 422
    - Test shard_count > idle nodes returns 400
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 10. Task creation (scheduler)
  - [x] 10.1 Implement task creation in `coordinator/scheduler.py`
    - Given a job with shard_count N, create N task records with status "queued"
    - Each task gets a unique `shard_index` from {0, 1, ..., N-1}
    - Each task's `task_config` is generated from `config_parser.py` with the appropriate shard_index
    - All tasks reference the parent job_id
    - _Requirements: 4.1_

  - [ ]* 10.2 Write property test for task creation (Property 6)
    - **Property 6: Task creation produces correct count and shard indices**
    - **Validates: Requirements 4.1**

- [x] 11. Task polling and assignment (pull-based)
  - [x] 11.1 Implement `GET /api/tasks/poll` endpoint in `coordinator/scheduler.py`
    - Authenticate request using auth dependency
    - Check if polling node is idle (not already busy)
    - Select one eligible queued task: node's `ram_mb` >= task's minimum RAM requirement; if task requires GPU, node must have non-null `gpu_model`
    - Assign task to polling node: update task status to "assigned", set `node_id`, set `assigned_at`
    - Update node status to "busy"
    - If this is the first task assigned for the job, update job status to "running" and set `started_at`
    - Return `TaskPollResponse` with task config, or empty response if no eligible task
    - _Requirements: 4.2, 4.3, 4.4, 4.5_

  - [ ]* 11.2 Write property tests for task assignment (Properties 7, 15)
    - **Property 7: Task assignment uses distinct idle nodes and updates their status**
    - **Property 15: Resource-eligible task assignment**
    - **Validates: Requirements 4.2**

  - [ ]* 11.3 Write unit tests for task polling
    - Test assigned task returned to polling worker
    - Test no task returns empty response
    - Test job status set to "running" on first assignment
    - Test busy node gets no task
    - _Requirements: 4.3, 4.4, 4.5_

- [x] 12. Task lifecycle endpoints
  - [x] 12.1 Implement `POST /api/tasks/{id}/start` endpoint
    - Authenticate request, verify task is assigned to requesting node
    - Update task status to "running", set `started_at`
    - _Requirements: 5.1_

  - [x] 12.2 Implement `POST /api/tasks/{id}/complete` endpoint
    - Authenticate request, verify task belongs to requesting node
    - Accept `TaskCompleteRequest` with checkpoint_path and optional final metrics
    - Update task status to "completed", set `completed_at`, store `checkpoint_path`
    - Insert artifact record (type "checkpoint", storage_path, job_id, task_id, node_id)
    - Update node status to "idle"
    - Check if all tasks in job are completed → trigger aggregation
    - Check if job should be marked failed (some failed, none queued/assigned/running)
    - _Requirements: 5.3, 7.1, 7.2, 6.1, 6.4_

  - [x] 12.3 Implement `POST /api/tasks/{id}/fail` endpoint
    - Authenticate request, verify task belongs to requesting node
    - Accept `TaskFailRequest` with error_message
    - Update task status to "failed", set `error_message`, set `completed_at`
    - Update node status to "idle"
    - Check if job should be marked failed (no tasks queued/assigned/running remaining)
    - _Requirements: 5.4, 6.4_

  - [x] 12.4 Implement `POST /api/tasks/{id}/upload-url` endpoint
    - Authenticate request, verify task belongs to requesting node
    - Generate signed upload URL using `storage.py` for path `{job_id}/{task_id}/final.pt`
    - Return the signed URL
    - _Requirements: 7.1_

- [x] 13. Metrics reporting endpoint
  - [x] 13.1 Implement `POST /api/metrics` endpoint
    - Authenticate request
    - Accept `MetricsReportRequest` with task_id, epoch, loss, accuracy
    - Insert record into metrics table with job_id, task_id, node_id, epoch, loss, accuracy
    - _Requirements: 5.2_

- [x] 14. Result aggregation
  - [x] 14.1 Implement `coordinator/aggregator.py` — Metrics aggregation logic
    - When all tasks in a job are completed, compute aggregated metrics
    - Use per-epoch metrics from the metrics table as source of truth (not TaskCompleteRequest final metrics)
    - For each task, the final epoch metric is the metric row with the largest reported `epoch` number for that task in the metrics table
    - Compute `mean_loss` as arithmetic mean of all tasks' final epoch losses
    - Compute `mean_accuracy` as arithmetic mean of all tasks' final epoch accuracies
    - Build per-node breakdown (node_id, loss, accuracy per task)
    - Update job record: status "completed", `aggregated_metrics`, `completed_at`
    - Handle partial failure: if any tasks failed and none remain active, mark job "failed" with `error_summary`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 14.2 Write property tests for aggregation (Properties 8, 9)
    - **Property 8: Metrics aggregation computes correct values and completes the job**
    - **Property 9: Job failure detection based on task statuses**
    - **Validates: Requirements 6.1, 6.2, 6.4**

- [x] 15. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Dashboard read endpoints and monitoring
  - [x] 16.1 Implement Dashboard-facing read endpoints (unauthenticated, local/demo only)
    - `GET /api/nodes` — List all nodes with status, hardware info, last heartbeat
    - `GET /api/jobs` — List all jobs with status, model type, dataset, shard count, timestamps
    - `GET /api/jobs/{id}` — Job detail with tasks, per-task status, aggregated metrics
    - `GET /api/jobs/{id}/results` — Aggregated metrics + per-task checkpoint paths
    - `GET /api/jobs/{id}/artifacts` — List artifacts for a job
    - _Requirements: 2.4, 6.3, 7.3, 9.1, 10.1_

  - [x] 16.2 Implement `GET /api/monitoring/summary` endpoint
    - Return counts: online nodes, idle nodes, busy nodes, offline nodes, queued jobs, running jobs, completed jobs, failed jobs
    - _Requirements: 11.2_

  - [ ]* 16.3 Write property test for monitoring summary (Property 12)
    - **Property 12: Monitoring summary returns correct counts**
    - **Validates: Requirements 11.2**

- [x] 17. Logging
  - [x] 17.1 Implement structured logging across Coordinator
    - Configure Python `logging` with structured format (timestamp, level, event type)
    - Log node registration events, job submission events, task assignment events
    - Log task completion and failure events with task_id, node_id, job_id, error_message
    - _Requirements: 11.1, 11.3_

- [x] 18. Worker client implementation
  - [x] 18.1 Implement `worker/config.py` — Task configuration parsing
    - Parse `TaskConfig` JSON received from Coordinator into local training parameters
    - Extract dataset_name, model_type, hyperparameters, shard_index, shard_count
    - _Requirements: 5.1, 5.5_

  - [x] 18.2 Implement `worker/reporter.py` — HTTP client for Coordinator communication
    - Functions: `register()`, `heartbeat()`, `poll_task()`, `start_task()`, `report_metrics()`, `request_upload_url()`, `complete_task()`, `fail_task()`
    - Include auth token in Authorization header for all authenticated requests
    - Handle 401 response: stop all loops, surface auth error for operator action
    - Implement retry with exponential backoff for transient failures (503, network errors)
    - _Requirements: 5.2, 5.3, 5.4, 8.1_

  - [x] 18.3 Implement `worker/storage.py` — Checkpoint upload via signed URL
    - Receive signed upload URL from Coordinator (via `reporter.request_upload_url()`)
    - Upload final checkpoint file via plain HTTP PUT/POST to the signed URL using `httpx` — no Supabase client library needed
    - Return success/failure status
    - Handle upload failure: report task failure to Coordinator
    - _Requirements: 7.1_

  - [x] 18.4 Implement `worker/state.py` — Local worker state persistence
    - On successful registration, persist `auth_token`, `node_db_id`, and `coordinator_url` to a local JSON file (e.g., `~/.group-ml-trainer/worker_state.json`)
    - On startup, check for existing state file: if present and valid, skip registration and reuse stored credentials
    - If auth token is rejected (401), delete the state file and exit for operator action
    - _Requirements: 1.1, 8.1_

  - [x] 18.5 Implement `worker/datasets.py` — Dataset loading and shard partitioning
    - Load MNIST, Fashion-MNIST via torchvision
    - Implement synthetic dataset generation
    - Partition dataset by shard_index and shard_count (deterministic split)
    - CIFAR-10 is stretch goal, not required for MVP
    - _Requirements: 3.5, 5.1_

  - [x] 18.6 Implement `worker/models.py` — MLP model definition
    - Configurable MLP: input size (derived from dataset), hidden layers list, output size, activation function
    - Support relu activation (default)
    - _Requirements: 3.6, 5.1_

  - [x] 18.7 Implement `worker/trainer.py` — PyTorch training loop
    - Load dataset shard using `datasets.py`
    - Instantiate model using `models.py`
    - Train for configured epochs with configured hyperparameters (learning_rate, batch_size)
    - Report metrics (loss, accuracy) per epoch via `reporter.py`
    - Save final checkpoint to local temp file
    - Handle training exceptions (OOM, NaN loss): catch and report failure
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 18.8 Implement `worker/main.py` — Entry point with registration, heartbeat, and poll loops
    - On startup: check for existing state file (`worker/state.py`); if valid, reuse credentials; otherwise register with Coordinator
    - Store auth token and node_db_id via `worker/state.py`
    - Start heartbeat loop (every 10 seconds) in background thread
    - Start poll loop (every 5 seconds): poll for task, if received → start → train → upload → complete
    - On auth failure (401): delete state file, stop all loops, log error, exit with message for operator
    - After task completes or fails, return to polling
    - _Requirements: 1.1, 2.1, 4.3, 5.1_

- [x] 19. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. Milestone A: Single-worker end-to-end validation
  - [x] 20.1 Run single-worker end-to-end flow manually
    - Start Coordinator locally
    - Start one Worker pointing at the local Coordinator
    - Worker registers, begins heartbeat
    - Submit one job with shard_count=1 via `POST /api/jobs` (e.g., MNIST, MLP, 5 epochs)
    - Worker polls, receives task, calls /start, trains, reports metrics per epoch
    - Worker requests signed upload URL, uploads final checkpoint, calls /complete
    - Verify: job status is "completed", aggregated_metrics populated, artifact record exists, checkpoint file in storage
    - Verify: `GET /api/jobs/{id}/results` returns correct metrics and checkpoint path
    - This milestone proves the core loop works before adding multi-node, failure semantics, or dashboard polish
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1_

- [x] 21. Dashboard frontend
  - [x] 21.1 Implement system overview page (`/`)
    - Fetch monitoring summary from `GET /api/monitoring/summary`
    - Display online node count, running job count, quick links to nodes and jobs pages
    - Use SWR with 10-second refresh interval
    - _Requirements: 11.4_

  - [x] 21.2 Implement nodes page (`/nodes`)
    - Fetch node list from `GET /api/nodes`
    - Display table: node_id, hostname, GPU, RAM, CPU cores, status, last heartbeat
    - Visual status indicators: idle=green, busy=yellow, offline=red
    - Auto-refresh with SWR (10-second interval)
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 21.3 Implement jobs list page (`/jobs`)
    - Fetch job list from `GET /api/jobs`
    - Display table: job_id, job_name, status, model_type, dataset, shard_count, created_at, completed_at
    - Link each job to detail page
    - Auto-refresh with SWR
    - _Requirements: 10.1_

  - [x] 21.4 Implement job detail page (`/jobs/[id]`)
    - Fetch job detail from `GET /api/jobs/{id}`
    - Display per-task progress: assigned node, status, current epoch, latest loss, accuracy
    - Display aggregated metrics on completion (mean loss, mean accuracy, per-node breakdown)
    - Display per-task error messages on failure
    - Display artifact download links from `GET /api/jobs/{id}/artifacts`
    - Auto-refresh with SWR (5-second interval for running jobs)
    - _Requirements: 10.2, 10.3, 10.4, 6.3, 7.3_

  - [x] 21.5 Implement job submission page (`/jobs/new`)
    - Form with fields: job_name (optional), dataset selector (MNIST, Fashion-MNIST, synthetic), model_type (MLP), shard_count, hyperparameters (learning_rate, epochs, batch_size, hidden_layers)
    - Submit to `POST /api/jobs`
    - Display validation errors from API response
    - Redirect to job detail page on success
    - _Requirements: 3.1_

  - [ ]* 21.6 Write unit tests for Dashboard components
    - Test node list renders with status indicators
    - Test job list renders with correct columns
    - Test job detail shows metrics and errors
    - Test monitoring summary displays counts
    - _Requirements: 9.1, 9.3, 10.1, 10.2, 10.3, 11.4_

- [x] 22. Integration tests
  - [ ]* 22.1 Write integration test: Full job lifecycle
    - Register nodes → submit job → tasks created and assigned → workers poll → report metrics → upload checkpoints → job completes with aggregated results
    - _Requirements: 1.1, 3.1, 4.1, 4.2, 5.2, 5.3, 6.1, 6.2_

  - [ ]* 22.2 Write integration test: Failure lifecycle
    - Register nodes → submit job → one task fails → job marked failed with error summary
    - _Requirements: 5.4, 6.4_

  - [ ]* 22.3 Write integration test: Heartbeat lifecycle
    - Register node → heartbeat updates timestamp → stop heartbeat → node marked offline → resume heartbeat → node recovers
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ]* 22.4 Write integration test: Dashboard data flow
    - Submit job → verify Dashboard API returns correct data at each stage
    - _Requirements: 9.1, 10.1, 10.2_

  - [ ]* 22.5 Write integration test: Task interruption lifecycle
    - Register node → assign task → start task → stop heartbeat → node marked offline → task marked failed with "node went offline" error → job failure semantics verified
    - _Requirements: 2.2, 6.4_

- [x] 23. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 24. Database schema migration for collaborative training
  - [x] 24.1 Update `scripts/bootstrap.sql` with schema changes for collaborative training
    - Add columns to `jobs` table: `current_round` (INTEGER, nullable), `total_rounds` (INTEGER, nullable), `global_model_path` (TEXT, nullable)
    - Modify `tasks` table: add `last_submitted_round` (INTEGER, nullable), remove `checkpoint_path` column
    - Modify `metrics` table: rename `epoch` to `round_number`, add `metric_type` (TEXT, CHECK IN ('worker_local', 'global_aggregated')), make `task_id` nullable (global metrics have no task)
    - Modify `artifacts` table: make `task_id` nullable (global checkpoints have no task), rename `epoch` to `round_number`
    - Create new `training_rounds` table with columns: `id` (UUID PK), `job_id` (UUID FK), `round_number` (INTEGER), `status` (TEXT CHECK IN ('in_progress', 'aggregating', 'completed')), `active_worker_count` (INTEGER), `submitted_count` (INTEGER), `global_loss` (NUMERIC nullable), `global_accuracy` (NUMERIC nullable), `started_at` (TIMESTAMPTZ), `completed_at` (TIMESTAMPTZ nullable), `created_at` (TIMESTAMPTZ)
    - Create new `gradient_submissions` table with columns: `id` (UUID PK), `job_id` (UUID FK), `task_id` (UUID FK), `node_id` (UUID FK nullable), `round_number` (INTEGER), `gradient_path` (TEXT), `local_loss` (NUMERIC nullable), `local_accuracy` (NUMERIC nullable), `created_at` (TIMESTAMPTZ)
    - Add indexes on `training_rounds(job_id)`, `gradient_submissions(job_id, round_number)`
    - Add storage bucket entries for `parameters` and `gradients` buckets
    - _Requirements: 4.2, 5.2, 5.5, 6.2, 7.1, 13.2_

  - [x] 24.2 Update `scripts/verify_schema.py` to verify new tables and columns
    - Add verification for `training_rounds` and `gradient_submissions` tables
    - Verify new columns on `jobs`, `tasks`, `metrics`, `artifacts`
    - Verify `parameters` and `gradients` storage buckets exist
    - _Requirements: 4.2, 5.2_

- [x] 25. Coordinator Pydantic models update
  - [x] 25.1 Update `coordinator/models.py` with new and modified models
    - Add `GradientSubmissionRequest` model: `round_number` (int, ge=0), `task_id` (str), `local_loss` (float | None), `local_accuracy` (float | None)
    - Add `ParameterDownloadResponse` model: `job_id` (str), `current_round` (int), `job_status` (str)
    - Add `RoundStatus` model: `round_number` (int), `status` (str), `active_worker_count` (int), `submitted_count` (int), `global_loss` (float | None), `global_accuracy` (float | None)
    - Remove `TaskCompleteRequest` and `MetricsReportRequest` models (no longer used)
    - Update `TaskPollResponse` to include `total_rounds` (int | None) field
    - Update `TaskConfig` to include `total_rounds` (int) field
    - _Requirements: 5.1, 5.6, 13.2, 13.3_

- [x] 26. Coordinator config parser update
  - [x] 26.1 Update `coordinator/config_parser.py` to derive `total_rounds` from epochs
    - Update `JobConfig` to include `total_rounds` property derived from `hyperparameters.epochs`
    - Update `generate_task_configs` to include `total_rounds` in each `TaskConfig`
    - _Requirements: 5.5, 12.1_

- [ ] 27. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 28. Parameter Server module
  - [x] 28.1 Implement `coordinator/param_server.py` — Global Model state management
    - Implement `initialize_model(job_id, job_config)`: create initial model from config using `worker/models.py` `build_model()`, serialize `state_dict` via `torch.save` to bytes, upload to Supabase Storage at `parameters/{job_id}/current.pt`, return storage path
    - Implement `get_parameters(job_id)`: download current model parameters from `parameters/{job_id}/current.pt` in Supabase Storage, return raw bytes
    - Implement `update_parameters(job_id, new_state_dict)`: serialize updated `state_dict`, upload to `parameters/{job_id}/current.pt`, overwriting previous
    - Implement `store_checkpoint(job_id, round_number)`: copy current parameters to `checkpoints/{job_id}/round_{N}.pt`, insert artifact record in database with `task_id=NULL`, `round_number`, `artifact_type='checkpoint'`
    - Use PyTorch `torch.save`/`torch.load` with `state_dict` convention
    - Use `io.BytesIO` for in-memory serialization to avoid temp files
    - Use Supabase Storage HTTP API via httpx (similar pattern to existing `coordinator/storage.py`)
    - _Requirements: 4.2, 7.1, 7.2, 7.5, 13.4, 13.5_

  - [ ]* 28.2 Write property test for model initialization (Property 7)
    - **Property 7: Global model initialization matches job configuration**
    - Verify initialized model `state_dict` layer dimensions match specified architecture
    - **Validates: Requirements 4.2**

  - [ ]* 28.3 Write property test for parameter serialization round-trip (Property 19)
    - **Property 19: Model parameter serialization round-trip preserves tensor values**
    - Verify `torch.save` → `torch.load` produces identical `state_dict` keys, shapes, and values
    - **Validates: Requirements 13.5**

- [x] 29. Synchronization Barrier module
  - [x] 29.1 Implement `coordinator/barrier.py` — Synchronization barrier logic
    - Implement `get_active_workers(job_id)`: query `tasks` table for tasks with status in ('assigned', 'running') for the given job, return set of task IDs
    - Implement `record_submission(job_id, round_number, task_id, node_id)`: insert record into `gradient_submissions` table, increment `submitted_count` on the `training_rounds` record
    - Implement `check_barrier(job_id, round_number)`: compare `submitted_count` against `active_worker_count` in `training_rounds` table, return True if barrier is met
    - Implement `remove_worker(job_id, task_id)`: remove worker from active set by marking task as failed, decrement `active_worker_count` on current `training_rounds` record for the job
    - Implement `create_round(job_id, round_number, active_worker_count)`: insert new `training_rounds` record with status 'in_progress'
    - _Requirements: 5.2, 5.5, 6.5, 14.2_

  - [ ]* 29.2 Write property test for synchronization barrier (Property 9)
    - **Property 9: Synchronization barrier enforces all-submit-before-advance**
    - Verify barrier reports "met" iff submitted_count == active_worker_count
    - **Validates: Requirements 5.2, 5.5**

  - [ ]* 29.3 Write property test for barrier adjustment on worker removal (Property 13)
    - **Property 13: Synchronization barrier adjusts when workers are removed**
    - Verify active_worker_count decrements when worker removed, and already-submitted gradient is preserved
    - **Validates: Requirements 6.5, 14.2, 14.4**

- [x] 30. Gradient Aggregator rewrite
  - [x] 30.1 Rewrite `coordinator/aggregator.py` for gradient aggregation
    - Replace existing metrics-averaging logic with gradient aggregation logic
    - Implement `aggregate_round(job_id, round_number)`:
      - Load all gradient submissions for the round from `gradient_submissions` table
      - Download each gradient tensor payload from Supabase Storage (`gradients/{job_id}/round_{N}/node_{node_id}.pt`)
      - Compute element-wise mean of all gradient `state_dict` tensors
      - Load current Global_Model parameters via `param_server.get_parameters()`
      - Apply SGD step: `new_params[key] = old_params[key] - learning_rate * mean_gradients[key]`
      - Upload updated parameters via `param_server.update_parameters()`
      - Compute and store per-round global metrics (mean of worker-reported losses and accuracies) in `training_rounds` table and `metrics` table (with `metric_type='global_aggregated'`)
      - Store per-worker metrics in `metrics` table (with `metric_type='worker_local'`)
      - Advance job's `current_round`
      - Create next round record via `barrier.create_round()` if more rounds remain
    - Implement `complete_job(job_id)`: mark job as "completed", store final checkpoint via `param_server.store_checkpoint()`, build `aggregated_metrics` JSON from all round metrics, set `completed_at`
    - Update `check_job_failure(job_id)`: when all workers fail, mark job as "failed", store partial checkpoint via `param_server.store_checkpoint()` if any rounds completed, populate `error_summary`
    - Clean up gradient storage for completed rounds (delete from `gradients/{job_id}/round_{N}/` after aggregation)
    - _Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5, 14.3_

  - [ ]* 30.2 Write property test for gradient aggregation (Property 10)
    - **Property 10: Gradient aggregation computes correct element-wise mean**
    - Generate random gradient dicts with identical keys/shapes, verify mean computation
    - **Validates: Requirements 5.3**

  - [ ]* 30.3 Write property test for per-round metrics (Property 11)
    - **Property 11: Per-round aggregated metrics correctly computed**
    - Verify global_loss = mean(worker_losses), global_accuracy = mean(worker_accuracies)
    - **Validates: Requirements 6.2**

  - [ ]* 30.4 Write property test for job failure with partial checkpoint (Property 12)
    - **Property 12: Job failure with partial checkpoint when all workers fail**
    - Verify job marked failed with error_summary, partial checkpoint stored
    - **Validates: Requirements 6.4, 14.3**

- [ ] 31. Coordinator storage module update
  - [ ] 31.1 Update `coordinator/storage.py` for parameter and gradient storage
    - Add `upload_blob(bucket, path, data: bytes)`: upload binary data to Supabase Storage
    - Add `download_blob(bucket, path) -> bytes`: download binary data from Supabase Storage
    - Add `delete_blob(bucket, path)`: delete a blob from Supabase Storage
    - Add `list_blobs(bucket, prefix) -> list[str]`: list blobs under a prefix
    - Remove `generate_signed_upload_url()` function (Workers no longer upload directly)
    - These are used by `param_server.py` and `aggregator.py` for parameter/gradient storage
    - _Requirements: 7.1, 13.1, 13.2_

- [ ] 32. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 33. Heartbeat monitor update for collaborative training
  - [ ] 33.1 Update `coordinator/heartbeat.py` for mid-training failure handling
    - When a node goes offline with an active task in a running job:
      - Mark the task as "failed" (existing behavior)
      - Call `barrier.remove_worker(job_id, task_id)` to remove worker from active set and adjust barrier
      - After removing worker, call `barrier.check_barrier(job_id, current_round)` — if barrier is now met (remaining workers already submitted), trigger `aggregator.aggregate_round()`
      - Call `check_job_failure(job_id)` to see if all workers have failed
    - _Requirements: 2.2, 6.5, 14.1, 14.2, 14.4_

  - [ ]* 33.2 Write property test for stale worker handling (Property 21)
    - **Property 21: Stale worker with active task triggers task failure and active set removal**
    - Verify task marked failed, worker removed from active set, barrier adjusted
    - **Validates: Requirements 14.1, 14.2**

- [ ] 34. Job submission endpoint update — Global Model initialization
  - [ ] 34.1 Update `POST /api/jobs` in `coordinator/main.py` to initialize Global_Model
    - After creating job record and tasks, call `param_server.initialize_model(job_id, job_config)` to create and store initial model parameters
    - Store the returned `global_model_path` on the job record
    - Set `total_rounds` on the job record (derived from `hyperparameters.epochs`)
    - Set `current_round` to 0
    - Create initial `training_rounds` record for round 0 via `barrier.create_round()`
    - _Requirements: 4.2, 4.3_

- [ ] 35. Task poll endpoint update
  - [ ] 35.1 Update `GET /api/tasks/poll` to include `total_rounds` in response
    - Include `total_rounds` from the job's `hyperparameters.epochs` in the `TaskPollResponse`
    - _Requirements: 4.4_

- [ ] 36. Task start endpoint update
  - [ ] 36.1 Update `POST /api/tasks/{id}/start` to initialize round tracking
    - On task start, ensure the initial `training_rounds` record exists for round 0 (idempotent)
    - _Requirements: 5.1_

- [ ] 37. New API endpoints for gradient exchange
  - [ ] 37.1 Implement `GET /api/jobs/{id}/parameters` — Binary parameter download
    - Authenticate request using auth dependency
    - Verify the requesting node has an active task for this job
    - Fetch current Global_Model parameters via `param_server.get_parameters(job_id)`
    - Return binary payload with `Content-Type: application/octet-stream`
    - Include `ParameterDownloadResponse` metadata in response headers or as multipart: `job_id`, `current_round`, `job_status`
    - If job status is "completed" or "failed", return appropriate status so Worker can exit training loop
    - _Requirements: 4.4, 5.4, 13.1_

  - [ ] 37.2 Implement `POST /api/jobs/{id}/gradients` — Gradient submission
    - Authenticate request using auth dependency
    - Verify the requesting node has an active task for this job
    - Parse `GradientSubmissionRequest` metadata (round_number, task_id, local_loss, local_accuracy)
    - Validate `round_number` matches job's `current_round` — reject with 409 if mismatched
    - Validate worker hasn't already submitted for this round — reject with 409 if duplicate
    - Store gradient binary payload to Supabase Storage at `gradients/{job_id}/round_{N}/node_{node_id}.pt`
    - Record submission via `barrier.record_submission()`
    - Store per-worker metrics in `metrics` table with `metric_type='worker_local'`
    - Update task's `last_submitted_round`
    - Check if barrier is met via `barrier.check_barrier()` — if met, trigger `aggregator.aggregate_round()`
    - If this was the final round and aggregation completes, call `aggregator.complete_job()`
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 13.2, 13.3_

  - [ ]* 37.3 Write property test for round validation (Property 20)
    - **Property 20: Round validation rejects mismatched round numbers**
    - Verify submissions with wrong round_number are rejected, correct round_number accepted
    - **Validates: Requirements 13.3**

- [ ] 38. Remove deprecated endpoints
  - [ ] 38.1 Remove `POST /api/tasks/{id}/complete` and `POST /api/tasks/{id}/upload-url` from `coordinator/main.py`
    - Remove the `complete_task` endpoint handler and its `TaskCompleteRequest` import
    - Remove the `request_upload_url` endpoint handler and its `generate_signed_upload_url` import
    - Remove the `POST /api/metrics` endpoint (metrics now submitted inline with gradients)
    - These endpoints are replaced by the gradient submission flow
    - _Requirements: 5.3, 13.2_

- [ ] 39. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 40. Dashboard endpoints update for round progress
  - [ ] 40.1 Update `coordinator/dashboard.py` for collaborative training data
    - Update `GET /api/jobs` to include `current_round`, `total_rounds` in job list response
    - Update `GET /api/jobs/{id}` to include:
      - Training round progress: `current_round`, `total_rounds`
      - Per-worker contribution status from `gradient_submissions` table: worker ID, last submitted round, status (waiting/computing/submitted)
      - Per-round global metrics from `training_rounds` table for convergence chart data
    - Update `GET /api/jobs/{id}/results` to return per-round metrics history and final Global_Model checkpoint path (not per-task checkpoints)
    - Update `GET /api/monitoring/summary` to include `current_round` for each running job
    - _Requirements: 6.3, 10.1, 10.2, 10.3, 10.4, 11.2_

  - [ ]* 40.2 Write property test for monitoring summary with round progress (Property 16)
    - **Property 16: Monitoring summary returns correct counts including round progress**
    - Verify counts match actual node/job statuses, and current_round included for running jobs
    - **Validates: Requirements 11.2**

- [ ] 41. Worker config update
  - [ ] 41.1 Update `worker/config.py` to include `total_rounds`
    - Add `total_rounds` field to `TaskConfig` model (int, gt=0)
    - Update `parse_task_config` to handle the new field
    - _Requirements: 5.1_

- [ ] 42. Worker reporter update for gradient exchange
  - [ ] 42.1 Update `worker/reporter.py` with new gradient exchange methods
    - Add `download_parameters(job_id) -> tuple[bytes, dict]`: GET `/api/jobs/{id}/parameters`, return binary payload and metadata (current_round, job_status)
    - Add `submit_gradients(job_id, round_number, task_id, gradient_bytes, local_loss, local_accuracy)`: POST `/api/jobs/{id}/gradients` with binary gradient payload and metadata
    - Remove `request_upload_url()` method (Workers no longer upload checkpoints)
    - Remove `complete_task()` method (Workers no longer individually complete)
    - Keep `report_metrics()` for backward compatibility or remove if fully replaced by inline gradient metrics
    - _Requirements: 5.1, 5.6, 13.1, 13.2_

- [ ] 43. Worker trainer rewrite for collaborative training
  - [ ] 43.1 Rewrite `worker/trainer.py` for round-based gradient computation
    - Replace independent training loop with collaborative round-based loop:
      - `run_task()` enters a loop for `total_rounds` iterations
      - Each round:
        1. Call `reporter.download_parameters(job_id)` to get current Global_Model parameters
        2. Check if job_status is "completed" or "failed" — if so, exit loop
        3. Load received parameters into local model via `model.load_state_dict()`
        4. Forward pass on local data shard to compute loss and accuracy
        5. Backward pass to compute gradients (do NOT call `optimizer.step()`)
        6. Collect gradients from `model.parameters()` into a `state_dict`-like dict
        7. Serialize gradients via `torch.save` to bytes
        8. Call `reporter.submit_gradients()` with gradient bytes and local metrics
        9. Wait/poll for next round (parameter download will block until Coordinator advances)
    - Remove checkpoint saving and upload logic (Coordinator handles checkpoints)
    - Remove `optimizer.step()` call — Worker only computes gradients
    - Keep error handling: NaN loss, OOM, unexpected errors → report failure
    - _Requirements: 5.1, 5.6, 13.1, 13.2_

  - [ ]* 43.2 Write property test for gradient computation shape (Property 8)
    - **Property 8: Worker gradient computation produces correctly shaped tensors**
    - Verify gradient dict keys match model state_dict keys, tensor shapes match
    - **Validates: Requirements 5.1**

- [ ] 44. Worker storage simplification
  - [ ] 44.1 Simplify or remove `worker/storage.py`
    - Workers no longer upload checkpoints directly — the Coordinator manages all model storage
    - Remove `StorageClient` class and `upload_checkpoint` method
    - If any utility functions remain useful, keep them; otherwise remove the module
    - Update `worker/main.py` to remove `StorageClient` instantiation and usage
    - _Requirements: 7.1, 7.5_

- [ ] 45. Worker main.py update for collaborative training loop
  - [ ] 45.1 Update `worker/main.py` to use new collaborative training loop
    - Update `_execute_task()` to call the rewritten `run_task()` from `trainer.py`
    - Remove `StorageClient` creation and passing (no longer needed)
    - After training loop completes (all rounds done or job completed), return to polling
    - _Requirements: 5.1_

- [ ] 46. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 47. Dashboard frontend updates for collaborative training
  - [ ] 47.1 Update `dashboard/lib/types.ts` for collaborative training types
    - Add `current_round`, `total_rounds`, `global_model_path` to `Job` interface
    - Add `last_submitted_round` to `Task` interface, remove `checkpoint_path`
    - Add `RoundStatus` interface: `round_number`, `status`, `active_worker_count`, `submitted_count`, `global_loss`, `global_accuracy`
    - Update `MonitoringSummary` to include per-running-job round info
    - Update `AggregatedMetrics` to include per-round metrics history
    - Rename `shard_count` references to `worker_count` where user-facing
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ] 47.2 Update `dashboard/lib/api.ts` if needed for new response shapes
    - Ensure fetcher and postJSON handle binary responses if needed
    - No major changes expected — REST JSON responses
    - _Requirements: 10.1_

  - [ ] 47.3 Update `dashboard/app/jobs/page.tsx` — Job list with round progress
    - Replace "Shards" column with "Workers" column
    - Add "Round" column showing `current_round / total_rounds` for running jobs
    - _Requirements: 10.1_

  - [ ] 47.4 Update `dashboard/app/jobs/[id]/page.tsx` — Job detail with training progress
    - Add training round progress bar: `current_round / total_rounds` with visual progress indicator
    - Add per-Worker contribution status table: Worker ID, status (waiting/computing/submitted), last submitted round
    - Add convergence chart section: display global loss and accuracy across training rounds (can use a simple table or basic chart)
    - Update task table: replace "Epoch" column with "Last Round", remove "Checkpoint" references
    - Show final Global_Model checkpoint download link on completion (single checkpoint, not per-task)
    - _Requirements: 10.2, 10.3, 10.4_

  - [ ] 47.5 Update `dashboard/app/jobs/new/page.tsx` — Rename shard_count to worker_count
    - Rename "Shard Count" label to "Worker Count" in the form
    - Update hint text to reference workers instead of shards
    - Keep the field name `shard_count` in the API payload (backend still uses this name)
    - _Requirements: 3.1_

  - [ ] 47.6 Update `dashboard/app/page.tsx` — Overview with round progress
    - If monitoring summary includes per-job round info, display current round for running jobs in the overview
    - _Requirements: 11.4_

- [ ] 48. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 49. Update existing tests for collaborative training
  - [ ]* 49.1 Update `tests/test_aggregator.py` for gradient aggregation
    - Replace metrics-averaging tests with gradient aggregation tests
    - Test: load gradients → compute mean → apply SGD step → verify updated parameters
    - Test: complete_job stores final checkpoint and aggregated metrics
    - Test: check_job_failure with partial checkpoint storage
    - _Requirements: 5.3, 6.1, 6.4_

  - [ ]* 49.2 Update `tests/test_heartbeat.py` for barrier adjustment
    - Test: offline node with active task → task failed → barrier adjusted → check if barrier now met
    - _Requirements: 14.1, 14.2_

  - [ ]* 49.3 Update `tests/test_task_lifecycle.py` for collaborative training flow
    - Test: task start → gradient submission → round advance → next round
    - Test: removed endpoints return 404 or are gone
    - _Requirements: 5.1, 5.2_

  - [ ]* 49.4 Update `tests/test_task_poll.py` for total_rounds in response
    - Test: poll response includes `total_rounds` field
    - _Requirements: 4.4_

- [ ] 50. Integration tests for collaborative training
  - [ ]* 50.1 Write integration test: Full collaborative training lifecycle
    - Register 2 nodes → submit job with worker_count=2 → tasks assigned → Workers download params → compute gradients → submit gradients → barrier met → aggregation → repeat for all rounds → job completed with final checkpoint and per-round metrics
    - _Requirements: 1.1, 3.1, 4.1, 4.2, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1_

  - [ ]* 50.2 Write integration test: Worker failure mid-training
    - Register 3 nodes → submit job → start training → one worker fails at round 2 → barrier adjusts → remaining 2 workers continue → job completes
    - _Requirements: 6.5, 14.1, 14.2_

  - [ ]* 50.3 Write integration test: All workers fail
    - Register 2 nodes → submit job → both workers fail → job marked failed with partial checkpoint
    - _Requirements: 6.4, 14.3_

  - [ ]* 50.4 Write integration test: Round validation
    - Worker submits gradient for wrong round → 409 rejected → Worker re-syncs and submits for correct round
    - _Requirements: 13.3_

- [ ] 51. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks 1–23 (marked completed) implemented the original independent-training scope
- Tasks 24–51 implement the collaborative distributed training rewrite using a Parameter Server pattern
- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate the 21 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests verify end-to-end flows against a test Supabase instance
- Dashboard endpoints are unauthenticated for local/demo use only
- Workers authenticate via token in Authorization header; 401 → stop and require operator action
- Workers no longer upload checkpoints — the Coordinator manages all model storage
- Gradient exchange uses binary payloads (PyTorch torch.save/torch.load format)
- The Coordinator acts as a centralized Parameter Server, applying SGD steps after gradient aggregation
- Synchronization barriers ensure all active Workers submit before the round advances
