# Requirements Document

## Introduction

Group ML Trainer is a distributed collaborative training platform that pools networked hardware to train a single shared ML model across multiple Worker nodes. Unlike embarrassingly parallel approaches where each Worker trains an independent model, Group ML Trainer implements true distributed training: Workers compute gradients on their local data shards and exchange updates through a synchronized aggregation protocol, producing one global model as the final output.

The system coordinates a central server (Coordinator) with multiple Worker nodes and a web-based Dashboard. The Coordinator manages the training job lifecycle, orchestrates synchronization barriers between Workers, aggregates gradient updates into a global model state, and distributes updated parameters back to Workers each round. Workers load their assigned data shard, compute local gradients, and participate in the synchronized training loop. The Dashboard provides real-time visibility into the collaborative training progress.

The platform targets local and shared external hardware running PyTorch-based workloads, with an initial focus on MLP models trained on small benchmark datasets using a centralized parameter server pattern for gradient aggregation.

### Distributed Training Strategy

The Coordinator acts as a centralized **Parameter Server**. Each training round proceeds as follows:

1. The Coordinator broadcasts the current global model parameters to all participating Workers.
2. Each Worker computes gradients on its local data shard using the global parameters.
3. Workers submit their computed gradients back to the Coordinator.
4. The Coordinator aggregates gradients from all Workers (e.g., averaging) and applies the update to the global model.
5. The process repeats for the configured number of epochs.

This produces a single trained model, not N independent models.

### Out of Scope for MVP

Model-parallel training, decentralized peer-to-peer gradient exchange (all-reduce without a parameter server), automatic mixed precision, gradient compression, asynchronous SGD, and elastic scaling (adding/removing Workers mid-training) are out of scope.

## Glossary

- **Coordinator**: The central backend server responsible for node registration, job submission, training orchestration, global model state management, gradient aggregation, synchronization barrier enforcement, and logging.
- **Worker**: A networked machine that registers with the Coordinator, receives global model parameters, computes local gradients on its data shard, and submits gradients back to the Coordinator each training round.
- **Job**: A complete distributed training workload submitted by a user, defining the model architecture, dataset, hyperparameters, and number of Workers. A Job produces a single trained global model.
- **Task**: A persistent assignment linking a Worker to a Job for the duration of training. Each Task represents one Worker's ongoing participation in the collaborative training loop.
- **Node**: A registered Worker instance identified by a unique node ID.
- **Heartbeat**: A periodic signal sent by a Worker to the Coordinator to indicate the Worker is alive and operational.
- **Training_Round**: One complete cycle of: broadcast parameters → compute local gradients → submit gradients → aggregate and update global model. Corresponds to one epoch of training across the full distributed dataset.
- **Global_Model**: The single shared model whose parameters are maintained by the Coordinator and updated each Training_Round via aggregated gradients from all Workers.
- **Gradient_Update**: The set of parameter gradients computed by a Worker on its local data shard during a Training_Round, submitted to the Coordinator for aggregation.
- **Synchronization_Barrier**: A coordination point where the Coordinator waits for all active Workers to submit their Gradient_Updates before proceeding to aggregate and advance to the next Training_Round.
- **Parameter_Broadcast**: The act of the Coordinator distributing the current Global_Model parameters to all participating Workers at the start of each Training_Round.
- **Checkpoint**: A serialized snapshot of the Global_Model weights saved at a point during training, representing the single shared model state.
- **Artifact**: Any file output produced by a Job, including Global_Model checkpoints and training logs.
- **Dashboard**: The web-based frontend that displays system status, node health, job progress, training convergence metrics, and global model state.
- **Auth_Token**: A secret credential issued to a Worker upon registration, used to authenticate subsequent requests to the Coordinator.
- **Shard**: A partition of the training dataset assigned to a single Worker. Each Worker computes gradients only on its shard but contributes to the single Global_Model.
- **Metrics**: Numerical training measurements (e.g., loss, accuracy) computed from the Global_Model state after each Training_Round.
- **Registry**: The Coordinator's internal store of all registered Nodes and their reported resource information.

---

## Requirements

### Requirement 1: Worker Node Registration

**User Story:** As a node operator, I want my machine to register with the Coordinator, so that it can participate as a compute resource in collaborative training jobs.

#### Acceptance Criteria

1. WHEN a Worker sends a registration request with its node ID, hostname, CPU cores, GPU model, VRAM, RAM, available disk, OS, Python version, PyTorch version, and status, THE Coordinator SHALL store the Node's information in the Registry and return a unique Auth_Token to the Worker.
2. IF a Worker sends a registration request with a duplicate node ID, THEN THE Coordinator SHALL reject the request and return a descriptive error message indicating the node ID is already registered.
3. IF a Worker sends a registration request with missing required fields, THEN THE Coordinator SHALL reject the request and return a descriptive error message listing the missing fields.
4. THE Coordinator SHALL assign each registered Node an initial status of "idle" upon successful registration.

---

### Requirement 2: Worker Heartbeat and Health Tracking

**User Story:** As a system operator, I want the Coordinator to track whether each Worker is alive, so that offline or failed nodes can be detected and handled during collaborative training.

#### Acceptance Criteria

1. WHEN a Worker sends a Heartbeat request, THE Coordinator SHALL update the Node's last heartbeat timestamp in the Registry.
2. WHILE a Node's last heartbeat timestamp is older than 30 seconds, THE Coordinator SHALL mark the Node's status as "offline".
3. WHEN a Node that was marked "offline" sends a Heartbeat, THE Coordinator SHALL update the Node's status to "idle" and record the new timestamp.
4. THE Coordinator SHALL expose an endpoint that returns the current health status and last heartbeat timestamp for all registered Nodes.

---

### Requirement 3: Job Submission

**User Story:** As a user, I want to submit a distributed training job to the Coordinator, so that multiple Workers can collaboratively train a single model.

#### Acceptance Criteria

1. WHEN a user submits a Job with a valid configuration (dataset name, model type, hyperparameters, and worker count), THE Coordinator SHALL create a Job record with status "queued" and return a unique job ID.
2. IF a user submits a Job with an unsupported dataset name or model type, THEN THE Coordinator SHALL reject the submission and return a descriptive error message listing supported options.
3. IF a user submits a Job with a worker count greater than the number of currently idle Nodes, THEN THE Coordinator SHALL reject the submission and return an error message indicating insufficient available nodes.
4. THE Coordinator SHALL validate that all required Job configuration fields are present before creating a Job record.
5. THE Coordinator SHALL support the following datasets: MNIST, Fashion-MNIST, and synthetic data. CIFAR-10 is a stretch goal.
6. THE Coordinator SHALL support the following model types: MLP.

---

### Requirement 4: Task Assignment and Training Initialization

**User Story:** As a system operator, I want the Coordinator to assign Workers to a Job and initialize the collaborative training session, so that all Workers begin training from the same model state.

#### Acceptance Criteria

1. WHEN a Job is created with status "queued", THE Coordinator SHALL create one Task per requested Worker, each assigned a unique data shard index, and assign each Task to a distinct idle Node.
2. THE Coordinator SHALL initialize a Global_Model with the architecture and hyperparameters specified in the Job configuration and store the initial model parameters.
3. WHEN all Tasks for a Job have been assigned to Workers, THE Coordinator SHALL update the Job status to "running" and broadcast the initial Global_Model parameters to all assigned Workers.
4. WHEN a Worker polls for work, THE Coordinator SHALL return the Task configuration (dataset shard index, model type, hyperparameters, total worker count) and the current Global_Model parameters if a Task is assigned to that Node.
5. IF no Task is assigned to the polling Worker, THEN THE Coordinator SHALL return an empty response indicating no work is available.

---

### Requirement 5: Synchronized Training Loop

**User Story:** As a user, I want all Workers to train collaboratively in synchronized rounds, so that the system produces a single well-trained model rather than independent models.

#### Acceptance Criteria

1. WHEN a Worker receives Global_Model parameters and a Task configuration, THE Worker SHALL load its assigned dataset shard, set its local model to the received Global_Model parameters, compute gradients over the local shard, and submit the Gradient_Update to the Coordinator.
2. THE Coordinator SHALL enforce a Synchronization_Barrier at each Training_Round, waiting for all active Workers assigned to the Job to submit their Gradient_Updates before proceeding to aggregation.
3. WHEN all active Workers have submitted Gradient_Updates for a Training_Round, THE Coordinator SHALL aggregate the gradients (by computing the mean of all submitted gradients) and apply the aggregated update to the Global_Model parameters.
4. WHEN the Coordinator completes gradient aggregation for a Training_Round, THE Coordinator SHALL broadcast the updated Global_Model parameters to all active Workers for the next Training_Round.
5. THE Coordinator SHALL track the current Training_Round number for each running Job and SHALL NOT advance to the next round until all active Workers have submitted their Gradient_Updates for the current round.
6. WHILE a Training_Round is in progress, THE Worker SHALL report local Metrics (loss and accuracy computed on its shard using the current Global_Model parameters) to the Coordinator along with its Gradient_Update.

---

### Requirement 6: Training Completion and Result Aggregation

**User Story:** As a user, I want the Coordinator to finalize the trained model and provide aggregated metrics when all training rounds complete, so that I can retrieve the single trained model.

#### Acceptance Criteria

1. WHEN all configured Training_Rounds (epochs) for a Job have been completed, THE Coordinator SHALL update the Job status to "completed" and store the final Global_Model parameters as the Job's primary Checkpoint.
2. THE Coordinator SHALL store aggregated Metrics for each completed Training_Round, including the global loss and global accuracy computed from the aggregated model state, and per-Worker loss and accuracy breakdown.
3. WHEN a user requests Job results, THE Coordinator SHALL return the aggregated Metrics across all Training_Rounds and the storage location of the final Global_Model Checkpoint.
4. IF one or more Workers fail during training and no Workers remain active for the Job, THEN THE Coordinator SHALL update the Job status to "failed" and include the per-Worker error messages in the Job record.
5. IF a Worker fails during training but other Workers remain active, THEN THE Coordinator SHALL continue training with the remaining Workers, adjusting the Synchronization_Barrier to wait only for active Workers.

---

### Requirement 7: Checkpoint and Artifact Storage

**User Story:** As a user, I want the Global_Model checkpoints to be stored reliably, so that I can retrieve the trained model weights after training completes.

#### Acceptance Criteria

1. WHEN a Job completes all Training_Rounds, THE Coordinator SHALL upload the final Global_Model Checkpoint to the configured storage backend and record the storage path and metadata in the database.
2. THE Coordinator SHALL associate each stored Checkpoint with its originating Job ID, the Training_Round number, and the upload timestamp.
3. WHEN a user requests the Artifacts for a completed Job, THE Coordinator SHALL return the Checkpoint metadata record including the storage path for the final Global_Model.
4. IF a Checkpoint upload fails, THEN THE Coordinator SHALL retry the upload once and, if the retry fails, mark the Job with a warning indicating the Checkpoint was not persisted while keeping the Job status as "completed".
5. Each stored Checkpoint SHALL represent the single Global_Model state, not per-Worker model states.

---

### Requirement 8: Worker Authentication

**User Story:** As a system operator, I want all Worker requests to the Coordinator to be authenticated, so that unauthorized machines cannot participate in training or submit gradient updates.

#### Acceptance Criteria

1. WHEN a Worker sends any request to the Coordinator after registration, THE Coordinator SHALL validate the Auth_Token included in the request header.
2. IF a Worker sends a request with a missing or invalid Auth_Token, THEN THE Coordinator SHALL reject the request with an HTTP 401 response and a descriptive error message.
3. THE Coordinator SHALL issue a unique Auth_Token per registered Node and SHALL NOT reuse tokens across different Nodes.
4. WHERE token revocation is enabled, THE Coordinator SHALL reject requests from a Node whose Auth_Token has been revoked.

---

### Requirement 9: Frontend Dashboard — Node Status

**User Story:** As a system operator, I want to view the status of all connected nodes in the Dashboard, so that I can monitor available compute resources during collaborative training.

#### Acceptance Criteria

1. THE Dashboard SHALL display a list of all registered Nodes including node ID, hostname, GPU model, VRAM, RAM, CPU cores, OS, status (idle/busy/offline), and last heartbeat timestamp.
2. WHEN a Node's status changes, THE Dashboard SHALL reflect the updated status within 10 seconds of the change occurring on the Coordinator.
3. THE Dashboard SHALL visually distinguish between idle, busy, and offline Nodes.

---

### Requirement 10: Frontend Dashboard — Job and Training Progress

**User Story:** As a user, I want to monitor the collaborative training progress in the Dashboard, so that I can track model convergence and Worker contributions.

#### Acceptance Criteria

1. THE Dashboard SHALL display a list of all Jobs including job ID, status, model type, dataset, worker count, submission timestamp, and completion timestamp (if applicable).
2. WHEN a Job is in status "running", THE Dashboard SHALL display the current Training_Round number, per-Worker status (waiting/computing/submitted), and the latest global loss and accuracy.
3. WHEN a Job reaches status "completed" or "failed", THE Dashboard SHALL display the final Global_Model Metrics (loss and accuracy over all Training_Rounds) or per-Worker error messages respectively.
4. THE Dashboard SHALL display a training convergence view showing global loss and accuracy plotted across Training_Rounds for running and completed Jobs.

---

### Requirement 11: Logging and Monitoring

**User Story:** As a system operator, I want the Coordinator to log system events and expose monitoring data, so that I can diagnose issues and track system health during distributed training.

#### Acceptance Criteria

1. THE Coordinator SHALL log all node registration events, job submission events, task assignment events, Training_Round completion events, gradient aggregation events, and Worker failure events with timestamps.
2. THE Coordinator SHALL expose an endpoint that returns the count of online nodes, idle nodes, busy nodes, queued jobs, running jobs, completed jobs, and the current Training_Round for each running job.
3. WHEN a Worker fails during training, THE Coordinator SHALL log the Task ID, Node ID, Job ID, the Training_Round at which the failure occurred, and the error message reported by the Worker.
4. THE Dashboard SHALL display the system summary metrics returned by the monitoring endpoint.

---

### Requirement 12: Configuration Parsing and Validation

**User Story:** As a developer, I want Job and Task configurations to be parsed and validated consistently, so that malformed configurations are caught before training begins.

#### Acceptance Criteria

1. WHEN a Job configuration is received, THE Coordinator SHALL parse it into a structured JobConfig object containing dataset name, model type, hyperparameters, and worker count.
2. IF a Job configuration contains fields with incorrect types or out-of-range values, THEN THE Coordinator SHALL return a descriptive validation error listing each invalid field and the expected type or range.
3. THE Coordinator SHALL serialize JobConfig objects into Task configuration payloads that Workers can parse back into equivalent JobConfig objects (round-trip property).
4. FOR ALL valid JobConfig objects, serializing then deserializing SHALL produce an equivalent JobConfig object with identical field values.

---

### Requirement 13: Gradient Exchange Protocol

**User Story:** As a developer, I want a well-defined protocol for exchanging model parameters and gradients between the Coordinator and Workers, so that the distributed training loop operates correctly.

#### Acceptance Criteria

1. THE Coordinator SHALL expose an endpoint that Workers call to retrieve the current Global_Model parameters for their assigned Job, serialized as a downloadable binary payload.
2. THE Coordinator SHALL expose an endpoint that Workers call to submit their computed Gradient_Update for the current Training_Round, accepting a binary payload of serialized gradient tensors.
3. WHEN a Worker submits a Gradient_Update, THE Coordinator SHALL validate that the update corresponds to the current Training_Round and reject updates for past or future rounds with a descriptive error message.
4. THE Coordinator SHALL serialize and deserialize model parameters and gradient tensors using PyTorch's native serialization format (torch.save / torch.load with state_dict convention).
5. FOR ALL valid Global_Model parameter payloads, serializing then deserializing SHALL produce a state_dict with identical tensor values and shapes (round-trip property).

---

### Requirement 14: Worker Failure Handling During Training

**User Story:** As a system operator, I want the system to handle Worker failures gracefully during collaborative training, so that a single Worker failure does not necessarily abort the entire Job.

#### Acceptance Criteria

1. IF a Worker's heartbeat becomes stale (older than 30 seconds) while the Worker has an active Task in a running Job, THEN THE Coordinator SHALL mark the Worker's Task as "failed" and remove the Worker from the Job's active Worker set.
2. WHEN a Worker is removed from a Job's active Worker set, THE Coordinator SHALL adjust the Synchronization_Barrier for subsequent Training_Rounds to wait only for the remaining active Workers.
3. IF all Workers in a Job have failed, THEN THE Coordinator SHALL mark the Job as "failed" and store the last successfully aggregated Global_Model parameters as a partial Checkpoint.
4. IF a Worker fails after submitting its Gradient_Update for the current Training_Round but before the round completes, THEN THE Coordinator SHALL include the submitted gradient in the current round's aggregation and remove the Worker from subsequent rounds.
