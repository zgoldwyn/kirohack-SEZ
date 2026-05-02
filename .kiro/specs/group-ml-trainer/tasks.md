# Implementation Plan: Group ML Trainer

## Overview

This plan implements a distributed ML task orchestration platform with three components: a Coordinator (FastAPI backend), Workers (Python agents), and a Dashboard (Next.js frontend). Tasks are ordered for incremental buildability — each task is implementable given the tasks before it. The Coordinator is built first (data layer → auth → endpoints → scheduling → aggregation), then the Worker client, then the Dashboard.

## Tasks

- [ ] 1. Project structure and core dependencies
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

  - [ ] 1.5 Create database schema verification script
    - Create `scripts/verify_schema.sql` (or `scripts/bootstrap.sql`) containing the full CREATE TABLE / CREATE INDEX statements for all 5 tables (nodes, jobs, tasks, metrics, artifacts) and the `checkpoints` storage bucket
    - Create `scripts/verify_schema.py` that connects to Supabase and verifies all required tables, columns, indexes, and the storage bucket exist
    - Document required environment setup in a `README.md` or `SETUP.md`
    - _Requirements: 1.1, 3.1_

- [ ] 2. Database client and storage integration
  - [x] 2.1 Implement `coordinator/db.py` — Supabase client initialization and query helpers
    - Initialize Supabase client from environment variables
    - Create helper functions for common queries: insert, select, update with filters
    - Wrap Supabase errors into consistent application exceptions
    - _Requirements: 1.1, 3.1_

  - [ ] 2.2 Implement `coordinator/storage.py` — Supabase Storage signed URL generation
    - Create function to generate time-limited signed upload URLs for the `checkpoints` bucket
    - Path convention: `{job_id}/{task_id}/final.pt`
    - Workers do not hold direct Supabase credentials; they use signed URLs
    - _Requirements: 7.1, 7.2_

- [ ] 3. Pydantic models for API layer
  - [x] 3.1 Implement `coordinator/models.py` — Request and response models
    - Define `NodeRegistrationRequest`, `NodeRegistrationResponse`
    - Define `JobSubmissionRequest`, `JobSubmissionResponse`
    - Define `MetricsReportRequest`, `TaskCompleteRequest`, `TaskFailRequest`
    - Define `TaskPollResponse`, `AggregatedMetrics`
    - Define `JobConfig`, `HyperParameters`, `TaskConfig` internal models
    - Add field validators (gt=0 for cpu_cores, ram_mb, disk_mb, shard_count; ge=0 for epoch)
    - Use status enums/constants from `coordinator/constants.py` throughout
    - _Requirements: 1.1, 1.3, 3.1, 3.4, 12.1_

- [ ] 4. Configuration parsing and validation
  - [ ] 4.1 Implement `coordinator/config_parser.py` — Job config parsing and task config generation
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

- [ ] 5. Auth module
  - [ ] 5.1 Implement `coordinator/auth.py` — Token generation, hashing, and FastAPI dependency
    - Token generation using `secrets.token_urlsafe(32)`
    - SHA-256 hashing for storage
    - FastAPI dependency `get_current_node` that extracts token from `Authorization` header, hashes it, looks up in `nodes` table, returns node record or raises HTTP 401
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 5.2 Write property tests for auth (Properties 10, 11)
    - **Property 10: Auth token validation accepts valid tokens and rejects invalid ones**
    - **Property 11: Auth tokens are unique across all registered nodes**
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [ ] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Node registration endpoint
  - [ ] 7.1 Implement `POST /api/nodes/register` endpoint
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

- [ ] 8. Heartbeat endpoint and staleness monitor
  - [ ] 8.1 Implement `POST /api/nodes/heartbeat` endpoint
    - Authenticate request using auth dependency
    - Update `last_heartbeat` timestamp for the authenticated node
    - If node was "offline", set status back to "idle"
    - _Requirements: 2.1, 2.3_

  - [ ] 8.2 Implement `coordinator/heartbeat.py` — Background staleness monitor
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

- [ ] 9. Job submission endpoint
  - [ ] 9.1 Implement `POST /api/jobs` endpoint
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

- [ ] 10. Task creation (scheduler)
  - [ ] 10.1 Implement task creation in `coordinator/scheduler.py`
    - Given a job with shard_count N, create N task records with status "queued"
    - Each task gets a unique `shard_index` from {0, 1, ..., N-1}
    - Each task's `task_config` is generated from `config_parser.py` with the appropriate shard_index
    - All tasks reference the parent job_id
    - _Requirements: 4.1_

  - [ ]* 10.2 Write property test for task creation (Property 6)
    - **Property 6: Task creation produces correct count and shard indices**
    - **Validates: Requirements 4.1**

- [ ] 11. Task polling and assignment (pull-based)
  - [ ] 11.1 Implement `GET /api/tasks/poll` endpoint in `coordinator/scheduler.py`
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

- [ ] 12. Task lifecycle endpoints
  - [ ] 12.1 Implement `POST /api/tasks/{id}/start` endpoint
    - Authenticate request, verify task is assigned to requesting node
    - Update task status to "running", set `started_at`
    - _Requirements: 5.1_

  - [ ] 12.2 Implement `POST /api/tasks/{id}/complete` endpoint
    - Authenticate request, verify task belongs to requesting node
    - Accept `TaskCompleteRequest` with checkpoint_path and optional final metrics
    - Update task status to "completed", set `completed_at`, store `checkpoint_path`
    - Insert artifact record (type "checkpoint", storage_path, job_id, task_id, node_id)
    - Update node status to "idle"
    - Check if all tasks in job are completed → trigger aggregation
    - Check if job should be marked failed (some failed, none queued/assigned/running)
    - _Requirements: 5.3, 7.1, 7.2, 6.1, 6.4_

  - [ ] 12.3 Implement `POST /api/tasks/{id}/fail` endpoint
    - Authenticate request, verify task belongs to requesting node
    - Accept `TaskFailRequest` with error_message
    - Update task status to "failed", set `error_message`, set `completed_at`
    - Update node status to "idle"
    - Check if job should be marked failed (no tasks queued/assigned/running remaining)
    - _Requirements: 5.4, 6.4_

  - [ ] 12.4 Implement `POST /api/tasks/{id}/upload-url` endpoint
    - Authenticate request, verify task belongs to requesting node
    - Generate signed upload URL using `storage.py` for path `{job_id}/{task_id}/final.pt`
    - Return the signed URL
    - _Requirements: 7.1_

- [ ] 13. Metrics reporting endpoint
  - [ ] 13.1 Implement `POST /api/metrics` endpoint
    - Authenticate request
    - Accept `MetricsReportRequest` with task_id, epoch, loss, accuracy
    - Insert record into metrics table with job_id, task_id, node_id, epoch, loss, accuracy
    - _Requirements: 5.2_

- [ ] 14. Result aggregation
  - [ ] 14.1 Implement `coordinator/aggregator.py` — Metrics aggregation logic
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

- [ ] 15. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Dashboard read endpoints and monitoring
  - [ ] 16.1 Implement Dashboard-facing read endpoints (unauthenticated, local/demo only)
    - `GET /api/nodes` — List all nodes with status, hardware info, last heartbeat
    - `GET /api/jobs` — List all jobs with status, model type, dataset, shard count, timestamps
    - `GET /api/jobs/{id}` — Job detail with tasks, per-task status, aggregated metrics
    - `GET /api/jobs/{id}/results` — Aggregated metrics + per-task checkpoint paths
    - `GET /api/jobs/{id}/artifacts` — List artifacts for a job
    - _Requirements: 2.4, 6.3, 7.3, 9.1, 10.1_

  - [ ] 16.2 Implement `GET /api/monitoring/summary` endpoint
    - Return counts: online nodes, idle nodes, busy nodes, offline nodes, queued jobs, running jobs, completed jobs, failed jobs
    - _Requirements: 11.2_

  - [ ]* 16.3 Write property test for monitoring summary (Property 12)
    - **Property 12: Monitoring summary returns correct counts**
    - **Validates: Requirements 11.2**

- [ ] 17. Logging
  - [ ] 17.1 Implement structured logging across Coordinator
    - Configure Python `logging` with structured format (timestamp, level, event type)
    - Log node registration events, job submission events, task assignment events
    - Log task completion and failure events with task_id, node_id, job_id, error_message
    - _Requirements: 11.1, 11.3_

- [ ] 18. Worker client implementation
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

  - [ ] 18.8 Implement `worker/main.py` — Entry point with registration, heartbeat, and poll loops
    - On startup: check for existing state file (`worker/state.py`); if valid, reuse credentials; otherwise register with Coordinator
    - Store auth token and node_db_id via `worker/state.py`
    - Start heartbeat loop (every 10 seconds) in background thread
    - Start poll loop (every 5 seconds): poll for task, if received → start → train → upload → complete
    - On auth failure (401): delete state file, stop all loops, log error, exit with message for operator
    - After task completes or fails, return to polling
    - _Requirements: 1.1, 2.1, 4.3, 5.1_

- [ ] 19. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Milestone A: Single-worker end-to-end validation
  - [ ] 20.1 Run single-worker end-to-end flow manually
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

- [ ] 22. Integration tests
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

- [ ] 23. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate the 15 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests verify end-to-end flows against a test Supabase instance
- Dashboard endpoints are unauthenticated for local/demo use only
- Workers authenticate via token in Authorization header; 401 → stop and require operator action
- Pull-based task assignment: tasks are assigned on poll, not pre-assigned at job creation
- Signed upload URLs: Workers never hold direct Supabase credentials
- Final checkpoint only for MVP: `{job_id}/{task_id}/final.pt`
