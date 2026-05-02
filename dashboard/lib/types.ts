// Shared TypeScript types for the ML Trainer Dashboard

export type NodeStatus = "idle" | "busy" | "offline";
export type JobStatus = "queued" | "running" | "completed" | "failed";
export type TaskStatus = "queued" | "assigned" | "running" | "completed" | "failed";

export interface Node {
  id: string;
  node_id: string;
  hostname: string;
  cpu_cores: number;
  gpu_model: string | null;
  vram_mb: number | null;
  ram_mb: number;
  disk_mb: number;
  os: string;
  python_version: string;
  pytorch_version: string;
  status: NodeStatus;
  last_heartbeat: string;
  created_at: string;
}

export interface HyperParameters {
  learning_rate?: number;
  epochs?: number;
  batch_size?: number;
  hidden_layers?: number[];
  activation?: string;
  [key: string]: unknown;
}

export interface AggregatedMetrics {
  mean_loss: number | null;
  mean_accuracy: number | null;
  per_node: Array<{
    node_id: string;
    loss: number | null;
    accuracy: number | null;
    task_id?: string;
  }>;
}

export interface Job {
  id: string;
  job_name: string | null;
  dataset_name: string;
  model_type: string;
  hyperparameters: HyperParameters;
  shard_count: number;
  status: JobStatus;
  aggregated_metrics: AggregatedMetrics | null;
  error_summary: Record<string, string> | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface Task {
  id: string;
  job_id: string;
  node_id: string | null;
  shard_index: number;
  status: TaskStatus;
  task_config: Record<string, unknown>;
  checkpoint_path: string | null;
  error_message: string | null;
  assigned_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  // Joined fields from metrics
  current_epoch?: number | null;
  latest_loss?: number | null;
  latest_accuracy?: number | null;
  // Joined node info
  node?: Node | null;
}

export interface JobDetail extends Job {
  tasks: Task[];
}

export interface Artifact {
  id: string;
  job_id: string;
  task_id: string;
  node_id: string | null;
  artifact_type: "checkpoint" | "log" | "output";
  storage_path: string;
  epoch: number | null;
  size_bytes: number | null;
  created_at: string;
}

export interface MonitoringSummary {
  online_nodes: number;
  idle_nodes: number;
  busy_nodes: number;
  offline_nodes: number;
  queued_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
}

export interface JobSubmissionRequest {
  job_name?: string;
  dataset_name: string;
  model_type: string;
  hyperparameters: HyperParameters;
  shard_count: number;
}

export interface JobSubmissionResponse {
  job_id: string;
}

export interface ApiError {
  error?: string;
  detail?: string | Array<{ loc: string[]; msg: string; type: string }>;
  supported?: string[];
  idle_count?: number;
}
