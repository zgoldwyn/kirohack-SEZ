# Group ML Trainer

A distributed ML training platform that pools networked hardware to execute PyTorch training jobs across multiple worker nodes. Submit a job, and the system splits it into shards, assigns each shard to an available worker, trains in parallel, and aggregates the results.

## How It Works

The platform has three components:

- **Coordinator** — A FastAPI server that manages worker registration, job submission, task scheduling, heartbeat monitoring, metrics aggregation, and checkpoint storage. It uses Supabase (Postgres + Storage) as its backend.
- **Worker** — A standalone Python agent that registers with the Coordinator, polls for tasks, runs PyTorch training, uploads checkpoints, and reports metrics. Multiple workers can run on different machines.
- **Dashboard** — A Next.js web UI for monitoring nodes, submitting jobs, and viewing training progress and results in real time.

### Training Flow

1. Workers register with the Coordinator and begin sending heartbeats.
2. A user submits a job (via the Dashboard or API) specifying a dataset, model type, hyperparameters, and shard count.
3. The Coordinator splits the job into tasks (one per shard) and queues them.
4. Workers poll for tasks. When a task is assigned, the worker loads its dataset shard, trains the model, reports per-epoch metrics, uploads the final checkpoint, and marks the task complete.
5. Once all tasks finish, the Coordinator aggregates metrics (mean loss, mean accuracy, per-node breakdown) and marks the job as completed.

### Supported Configurations

| | Options |
|---|---|
| **Datasets** | MNIST, Fashion-MNIST, synthetic |
| **Models** | MLP (multi-layer perceptron) |
| **Hyperparameters** | learning_rate, epochs, batch_size, hidden_layers, activation |

## Prerequisites

- Python 3.11+
- Node.js 18+ (for the Dashboard)
- A [Supabase](https://supabase.com) project with Storage enabled
- PyTorch 2.6+ (installed automatically with worker dependencies)

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_KEY=<your-service-role-key>
```

Find the service-role key in your Supabase project under **Settings → API → service_role (secret)**.

> Do not commit `.env` to version control.

### 2. Database

The schema is defined in `scripts/bootstrap.sql`. It creates five tables (`nodes`, `jobs`, `tasks`, `metrics`, `artifacts`) with indexes.

**Option A — Supabase SQL Editor:**

1. Open your Supabase project → **SQL Editor**
2. Paste the contents of `scripts/bootstrap.sql`
3. Click **Run**

**Option B — Supabase CLI:**

```bash
supabase db push --db-url "postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres" < scripts/bootstrap.sql
```

### 3. Storage Bucket

Create a private `checkpoints` bucket in your Supabase project:

1. Go to **Storage** in the Supabase Dashboard
2. Click **New bucket**
3. Name it `checkpoints`, set it to **private**

### 4. Verify Schema

```bash
pip install supabase python-dotenv
python scripts/verify_schema.py
```

This confirms all tables, columns, and the storage bucket are in place.

## Running the Platform

### Coordinator

```bash
cd coordinator
pip install -r requirements.txt
uvicorn coordinator.main:app --reload
```

The Coordinator starts on `http://localhost:8000` by default.

### Worker

```bash
cd worker
pip install -r requirements.txt
python -m worker.main
```

The worker auto-detects hardware (CPU, RAM, GPU if available), registers with the Coordinator, and begins polling for tasks. Run multiple workers on different machines (or terminals) to parallelize training.

Workers persist their credentials to `~/.group-ml-trainer/worker_state.json` so they can reconnect without re-registering after a restart.

**Worker environment variables:**

| Variable | Default | Description |
|---|---|---|
| `COORDINATOR_URL` | `http://localhost:8000` | Coordinator base URL |
| `WORKER_HEARTBEAT_INTERVAL` | `10` | Seconds between heartbeats |
| `WORKER_POLL_INTERVAL` | `5` | Seconds between task polls |
| `WORKER_LOG_LEVEL` | `INFO` | Logging level |

### Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Opens on `http://localhost:3000`. The dashboard auto-refreshes every 10 seconds and provides:

- **Overview** — Live counts of online/idle/busy/offline nodes and queued/running/completed/failed jobs.
- **Nodes** — Table of all registered workers with hardware info, status, and last heartbeat.
- **Jobs** — List of all jobs with status, model, dataset, and timestamps. Click a job to see per-task progress, per-epoch metrics, aggregated results, and artifacts.
- **New Job** — Form to submit a training job with dataset, model, hyperparameters, and shard count.

Set `NEXT_PUBLIC_API_URL` if the Coordinator is not running on `http://localhost:8000`.

## API Reference

All worker-facing endpoints require a `Authorization: Bearer <token>` header (issued at registration). Dashboard endpoints are unauthenticated.

### Worker Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/nodes/register` | Register a new worker node |
| `POST` | `/api/nodes/heartbeat` | Send a heartbeat (authenticated) |
| `POST` | `/api/jobs` | Submit a training job |
| `GET` | `/api/tasks/poll` | Poll for an available task (authenticated) |
| `POST` | `/api/tasks/{id}/start` | Mark a task as running (authenticated) |
| `POST` | `/api/tasks/{id}/complete` | Mark a task as completed (authenticated) |
| `POST` | `/api/tasks/{id}/fail` | Mark a task as failed (authenticated) |
| `POST` | `/api/tasks/{id}/upload-url` | Get a signed URL for checkpoint upload (authenticated) |
| `POST` | `/api/metrics` | Report per-epoch training metrics (authenticated) |

### Dashboard Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/nodes` | List all registered nodes |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/{id}` | Job detail with tasks and metrics |
| `GET` | `/api/jobs/{id}/results` | Aggregated metrics and checkpoint paths |
| `GET` | `/api/jobs/{id}/artifacts` | List artifacts for a job |
| `GET` | `/api/monitoring/summary` | System-wide node and job counts |
| `GET` | `/health` | Health check |

### Submitting a Job via API

```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "mnist-test",
    "dataset_name": "MNIST",
    "model_type": "MLP",
    "shard_count": 2,
    "hyperparameters": {
      "learning_rate": 0.01,
      "epochs": 5,
      "batch_size": 64,
      "hidden_layers": [128, 64]
    }
  }'
```

The `shard_count` must not exceed the number of idle workers.

## Project Structure

```
├── coordinator/          # FastAPI Coordinator server
│   ├── main.py           # API routes and app lifecycle
│   ├── db.py             # Supabase PostgREST client
│   ├── auth.py           # Token generation and validation
│   ├── scheduler.py      # Task creation and pull-based assignment
│   ├── aggregator.py     # Metrics aggregation and job completion
│   ├── heartbeat.py      # Background staleness monitor
│   ├── dashboard.py      # Read-only dashboard endpoints
│   ├── config_parser.py  # Job/task config validation
│   ├── storage.py        # Signed upload URL generation
│   ├── models.py         # Pydantic request/response models
│   ├── constants.py      # Enums and supported datasets/models
│   └── logging_config.py # Structured logging setup
├── worker/               # Standalone Worker agent
│   ├── main.py           # Worker lifecycle (register, heartbeat, poll, train)
│   ├── trainer.py        # PyTorch training loop
│   ├── datasets.py       # Dataset loading and shard partitioning
│   ├── models.py         # MLP model definition
│   ├── reporter.py       # HTTP client for Coordinator communication
│   ├── storage.py        # Checkpoint upload via signed URLs
│   ├── config.py         # Task config parsing
│   └── state.py          # Local credential persistence
├── dashboard/            # Next.js web dashboard
│   ├── app/              # App router pages (overview, jobs, nodes)
│   ├── components/       # Shared UI components
│   └── lib/              # API client and TypeScript types
├── scripts/
│   ├── bootstrap.sql     # Database schema DDL
│   └── verify_schema.py  # Schema verification script
└── tests/                # Test suite
```

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Architecture Notes

- **Pull-based scheduling** — Workers poll for tasks rather than receiving pushes. The Coordinator matches queued tasks to idle workers based on resource requirements (RAM, GPU).
- **Token auth** — Each worker gets a unique token at registration. Tokens are SHA-256 hashed before storage. All worker requests are authenticated via Bearer token.
- **Heartbeat monitoring** — A background loop checks for stale heartbeats every 10 seconds. Workers silent for 30+ seconds are marked offline, and their in-progress tasks are failed.
- **Checkpoint storage** — Workers upload checkpoints to Supabase Storage via time-limited signed URLs. Workers never hold direct Supabase credentials.
- **Retry with backoff** — The worker's HTTP client retries transient failures (503, network errors) with exponential backoff. Auth failures (401) immediately stop the worker.
- **State persistence** — Workers save credentials locally so they survive restarts without re-registering.
