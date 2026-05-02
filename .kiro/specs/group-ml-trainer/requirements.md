# Requirements Document

## Introduction

Group ML Trainer is a distributed compute orchestration platform that pools underused networked hardware to split machine learning training workloads across connected nodes. The system coordinates a central server (Coordinator) with multiple worker nodes (Workers) and a web-based dashboard (Dashboard), enabling users to submit ML training jobs that are distributed, executed, and aggregated across available hardware. The platform targets local and shared external hardware running PyTorch-based workloads, with an initial focus on MLP models trained on small benchmark datasets.

## Glossary

- **Coordinator**: The central backend server responsible for node registration, job submission, task assignment, heartbeat tracking, training state coordination, result aggregation, and logging.
- **Worker**: A networked machine that registers with the Coordinator, reports available resources, receives and executes training tasks, and returns results.
- **Job**: A complete ML training workload submitted by a user, composed of one or more Tasks.
- **Task**: A unit of work assigned to a single Worker, representing a shard of a Job (e.g., a data partition or model shard).
- **Node**: A registered Worker instance identified by a unique node ID.
- **Heartbeat**: A periodic signal sent by a Worker to the Coordinator to indicate the Worker is alive and operational.
- **Checkpoint**: A serialized snapshot of model weights and optimizer state saved at a point during training.
- **Artifact**: Any file output produced by a Job or Task, including checkpoints, logs, and model outputs.
- **Dashboard**: The web-based frontend that displays system status, node health, job progress, and metrics.
- **Auth_Token**: A secret credential issued to a Worker upon registration, used to authenticate subsequent requests to the Coordinator.
- **Shard**: A partition of a dataset or model assigned to a single Worker for parallel processing.
- **Metrics**: Numerical training measurements (e.g., loss, accuracy) reported by Workers during and after Task execution.
- **Registry**: The Coordinator's internal store of all registered Nodes and their reported resource information.

---

## Requirements

### Requirement 1: Worker Node Registration

**User Story:** As a node operator, I want my machine to register with the Coordinator, so that it can be recognized as an available compute resource.

#### Acceptance Criteria

1. WHEN a Worker sends a registration request with its node ID, hostname, CPU cores, GPU model, VRAM, RAM, available disk, OS, Python version, PyTorch version, and status, THE Coordinator SHALL store the Node's information in the Registry and return a unique Auth_Token to the Worker.
2. IF a Worker sends a registration request with a duplicate node ID, THEN THE Coordinator SHALL reject the request and return a descriptive error message indicating the node ID is already registered.
3. IF a Worker sends a registration request with missing required fields, THEN THE Coordinator SHALL reject the request and return a descriptive error message listing the missing fields.
4. THE Coordinator SHALL assign each registered Node an initial status of "idle" upon successful registration.

---

### Requirement 2: Worker Heartbeat and Health Tracking

**User Story:** As a system operator, I want the Coordinator to track whether each Worker is alive, so that offline or failed nodes can be detected and excluded from task assignment.

#### Acceptance Criteria

1. WHEN a Worker sends a Heartbeat request, THE Coordinator SHALL update the Node's last heartbeat timestamp in the Registry.
2. WHILE a Node's last heartbeat timestamp is older than 30 seconds, THE Coordinator SHALL mark the Node's status as "offline".
3. WHEN a Node that was marked "offline" sends a Heartbeat, THE Coordinator SHALL update the Node's status to "idle" and record the new timestamp.
4. THE Coordinator SHALL expose an endpoint that returns the current health status and last heartbeat timestamp for all registered Nodes.

---

### Requirement 3: Job Submission

**User Story:** As a user, I want to submit an ML training job to the Coordinator, so that the workload can be distributed across available Worker nodes.

#### Acceptance Criteria

1. WHEN a user submits a Job with a valid configuration (dataset name, model type, hyperparameters, and shard count), THE Coordinator SHALL create a Job record with status "queued" and return a unique job ID.
2. IF a user submits a Job with an unsupported dataset name or model type, THEN THE Coordinator SHALL reject the submission and return a descriptive error message listing supported options.
3. IF a user submits a Job with a shard count greater than the number of currently idle Nodes, THEN THE Coordinator SHALL reject the submission and return an error message indicating insufficient available nodes.
4. THE Coordinator SHALL validate that all required Job configuration fields are present before creating a Job record.
5. THE Coordinator SHALL support the following datasets: MNIST, Fashion-MNIST, CIFAR-10, and synthetic data.
6. THE Coordinator SHALL support the following model types: MLP.

---

### Requirement 4: Task Assignment and Distribution

**User Story:** As a system operator, I want the Coordinator to split a Job into Tasks and assign them to idle Workers, so that training is parallelized across available hardware.

#### Acceptance Criteria

1. WHEN a Job is created with status "queued", THE Coordinator SHALL split the Job into the specified number of Shards and create one Task per Shard.
2. THE Coordinator SHALL assign each Task to a distinct idle Node, updating the Node's status to "busy" upon assignment.
3. WHEN a Worker polls for work, THE Coordinator SHALL return the assigned Task configuration (dataset shard index, model type, hyperparameters, and total shard count) if a Task is assigned to that Node.
4. IF no Task is assigned to the polling Worker, THEN THE Coordinator SHALL return an empty response indicating no work is available.
5. THE Coordinator SHALL update the Job status to "running" when at least one Task has been assigned to a Worker.

---

### Requirement 5: Task Execution on Worker

**User Story:** As a node operator, I want the Worker to execute assigned training Tasks using PyTorch, so that the compute contribution of each node is utilized.

#### Acceptance Criteria

1. WHEN a Worker receives a Task configuration, THE Worker SHALL download or generate the assigned dataset Shard and begin training the specified model using PyTorch.
2. WHILE a Task is executing, THE Worker SHALL report Metrics (loss and accuracy per epoch) to the Coordinator at the end of each training epoch.
3. WHEN a Task completes successfully, THE Worker SHALL upload the resulting Checkpoint to the configured storage backend and notify the Coordinator with the Checkpoint location and final Metrics.
4. IF a Task fails due to an exception during training, THEN THE Worker SHALL report the failure to the Coordinator with a descriptive error message and set the Task status to "failed".
5. THE Worker SHALL only execute training Tasks described by a configuration object received from the Coordinator and SHALL NOT execute arbitrary code.

---

### Requirement 6: Result Aggregation

**User Story:** As a user, I want the Coordinator to aggregate results from all completed Tasks in a Job, so that I can retrieve a unified training outcome.

#### Acceptance Criteria

1. WHEN all Tasks in a Job reach status "completed", THE Coordinator SHALL aggregate the Metrics from all Tasks and update the Job status to "completed".
2. THE Coordinator SHALL store the aggregated Metrics (mean loss, mean accuracy, per-node breakdown) in the Job record upon Job completion.
3. WHEN a user requests Job results, THE Coordinator SHALL return the aggregated Metrics and the list of Checkpoint locations for the completed Job.
4. IF one or more Tasks in a Job reach status "failed" and no Tasks remain in status "running" or "queued", THEN THE Coordinator SHALL update the Job status to "failed" and include the per-Task error messages in the Job record.

---

### Requirement 7: Checkpoint and Artifact Storage

**User Story:** As a user, I want training checkpoints and artifacts to be stored reliably, so that I can retrieve model weights after training completes.

#### Acceptance Criteria

1. WHEN a Worker uploads a Checkpoint, THE Coordinator SHALL store the Checkpoint file in the configured storage backend (Supabase Storage or local disk) and record the storage path and metadata in the database.
2. THE Coordinator SHALL associate each stored Checkpoint with its originating Task ID, Job ID, Node ID, epoch number, and upload timestamp.
3. WHEN a user requests the Artifacts for a completed Job, THE Coordinator SHALL return a list of Checkpoint metadata records including storage paths for all Tasks in that Job.
4. IF a Checkpoint upload fails, THEN THE Coordinator SHALL mark the associated Task as "failed" and return a descriptive error message to the Worker.

---

### Requirement 8: Worker Authentication

**User Story:** As a system operator, I want all Worker requests to the Coordinator to be authenticated, so that unauthorized machines cannot submit results or consume job assignments.

#### Acceptance Criteria

1. WHEN a Worker sends any request to the Coordinator after registration, THE Coordinator SHALL validate the Auth_Token included in the request header.
2. IF a Worker sends a request with a missing or invalid Auth_Token, THEN THE Coordinator SHALL reject the request with an HTTP 401 response and a descriptive error message.
3. THE Coordinator SHALL issue a unique Auth_Token per registered Node and SHALL NOT reuse tokens across different Nodes.
4. WHERE token revocation is enabled, THE Coordinator SHALL reject requests from a Node whose Auth_Token has been revoked.

---

### Requirement 9: Frontend Dashboard — Node Status

**User Story:** As a system operator, I want to view the status of all connected nodes in the Dashboard, so that I can monitor available compute resources.

#### Acceptance Criteria

1. THE Dashboard SHALL display a list of all registered Nodes including node ID, hostname, GPU model, VRAM, RAM, CPU cores, OS, status (idle/busy/offline), and last heartbeat timestamp.
2. WHEN a Node's status changes, THE Dashboard SHALL reflect the updated status within 10 seconds of the change occurring on the Coordinator.
3. THE Dashboard SHALL visually distinguish between idle, busy, and offline Nodes.

---

### Requirement 10: Frontend Dashboard — Job and Training Progress

**User Story:** As a user, I want to monitor active and completed jobs in the Dashboard, so that I can track training progress and outcomes.

#### Acceptance Criteria

1. THE Dashboard SHALL display a list of all Jobs including job ID, status, model type, dataset, shard count, submission timestamp, and completion timestamp (if applicable).
2. WHEN a Job is in status "running", THE Dashboard SHALL display per-Task progress including assigned Node, current epoch, latest loss, and latest accuracy.
3. WHEN a Job reaches status "completed" or "failed", THE Dashboard SHALL display the final aggregated Metrics or per-Task error messages respectively.
4. THE Dashboard SHALL provide a view of per-Node Metrics for each running or completed Job.

---

### Requirement 11: Logging and Monitoring

**User Story:** As a system operator, I want the Coordinator to log system events and expose monitoring data, so that I can diagnose issues and track system health.

#### Acceptance Criteria

1. THE Coordinator SHALL log all node registration events, job submission events, task assignment events, task completion events, and task failure events with timestamps.
2. THE Coordinator SHALL expose an endpoint that returns the count of online nodes, idle nodes, busy nodes, queued jobs, running jobs, and completed jobs.
3. WHEN a Task fails, THE Coordinator SHALL log the Task ID, Node ID, Job ID, and the error message reported by the Worker.
4. THE Dashboard SHALL display the system summary metrics returned by the monitoring endpoint.

---

### Requirement 12: Configuration Parsing and Validation

**User Story:** As a developer, I want Job and Task configurations to be parsed and validated consistently, so that malformed configurations are caught before execution begins.

#### Acceptance Criteria

1. WHEN a Job configuration is received, THE Coordinator SHALL parse it into a structured JobConfig object containing dataset name, model type, hyperparameters, and shard count.
2. IF a Job configuration contains fields with incorrect types or out-of-range values, THEN THE Coordinator SHALL return a descriptive validation error listing each invalid field and the expected type or range.
3. THE Coordinator SHALL serialize JobConfig objects into Task configuration payloads that Workers can parse back into equivalent JobConfig objects (round-trip property).
4. FOR ALL valid JobConfig objects, serializing then deserializing SHALL produce an equivalent JobConfig object with identical field values.
