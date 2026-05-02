"use client";

import useSWR from "swr";
import Link from "next/link";
import { use } from "react";
import { fetcher, formatDate, API_URL } from "@/lib/api";
import type { JobDetail, Artifact } from "@/lib/types";
import StatusBadge from "@/components/StatusBadge";
import ErrorMessage from "@/components/ErrorMessage";

interface JobDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function JobDetailPage({ params }: JobDetailPageProps) {
  const { id } = use(params);

  // Refresh every 5 seconds for running jobs, 30 seconds otherwise
  const { data: job, error: jobError } = useSWR<JobDetail>(
    `/api/jobs/${id}`,
    fetcher,
    {
      refreshInterval: (data) =>
        data?.status === "running" || data?.status === "queued" ? 5_000 : 30_000,
    }
  );

  const { data: artifacts, error: artifactsError } = useSWR<Artifact[]>(
    job ? `/api/jobs/${id}/artifacts` : null,
    fetcher,
    { refreshInterval: 30_000 }
  );

  if (jobError) {
    return <ErrorMessage message={jobError.message} />;
  }

  if (!job) {
    return <div className="text-sm text-gray-500">Loading job details…</div>;
  }

  const hyperparams = job.hyperparameters || {};

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900">
              {job.job_name || "Unnamed Job"}
            </h1>
            <StatusBadge status={job.status} />
          </div>
          <p className="mt-1 font-mono text-xs text-gray-400">{job.id}</p>
        </div>
        <Link
          href="/jobs"
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          ← Back to Jobs
        </Link>
      </div>

      {/* Job metadata */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-base font-semibold text-gray-700">Job Details</h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3 lg:grid-cols-4">
          <div>
            <dt className="text-gray-500">Model</dt>
            <dd className="font-medium text-gray-900">{job.model_type}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Dataset</dt>
            <dd className="font-medium text-gray-900">{job.dataset_name}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Shards</dt>
            <dd className="font-medium text-gray-900">{job.shard_count}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Created</dt>
            <dd className="font-medium text-gray-900">{formatDate(job.created_at)}</dd>
          </div>
          {job.started_at && (
            <div>
              <dt className="text-gray-500">Started</dt>
              <dd className="font-medium text-gray-900">{formatDate(job.started_at)}</dd>
            </div>
          )}
          {job.completed_at && (
            <div>
              <dt className="text-gray-500">Completed</dt>
              <dd className="font-medium text-gray-900">{formatDate(job.completed_at)}</dd>
            </div>
          )}
        </dl>

        {/* Hyperparameters */}
        {Object.keys(hyperparams).length > 0 && (
          <div className="mt-4 border-t border-gray-100 pt-4">
            <h3 className="mb-2 text-sm font-medium text-gray-600">Hyperparameters</h3>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-5">
              {Object.entries(hyperparams).map(([key, val]) => (
                <div key={key}>
                  <dt className="text-gray-500">{key}</dt>
                  <dd className="font-medium text-gray-900">
                    {Array.isArray(val) ? val.join(", ") : String(val ?? "—")}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </section>

      {/* Aggregated metrics (completed jobs) */}
      {job.status === "completed" && job.aggregated_metrics && (
        <section className="rounded-lg border border-green-200 bg-green-50 p-6 shadow-sm">
          <h2 className="mb-4 text-base font-semibold text-green-800">Aggregated Results</h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div className="rounded-md bg-white p-4 shadow-sm">
              <p className="text-xs text-gray-500">Mean Loss</p>
              <p className="mt-1 text-2xl font-bold text-gray-900">
                {job.aggregated_metrics.mean_loss != null
                  ? job.aggregated_metrics.mean_loss.toFixed(4)
                  : "—"}
              </p>
            </div>
            <div className="rounded-md bg-white p-4 shadow-sm">
              <p className="text-xs text-gray-500">Mean Accuracy</p>
              <p className="mt-1 text-2xl font-bold text-gray-900">
                {job.aggregated_metrics.mean_accuracy != null
                  ? `${(job.aggregated_metrics.mean_accuracy * 100).toFixed(2)}%`
                  : "—"}
              </p>
            </div>
          </div>

          {/* Per-node breakdown */}
          {job.aggregated_metrics.per_node && job.aggregated_metrics.per_node.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium text-green-700">Per-Node Breakdown</h3>
              <div className="overflow-x-auto rounded-md border border-green-200 bg-white">
                <table className="min-w-full divide-y divide-gray-100 text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left font-semibold text-gray-600">Node ID</th>
                      <th className="px-4 py-2 text-left font-semibold text-gray-600">Loss</th>
                      <th className="px-4 py-2 text-left font-semibold text-gray-600">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {job.aggregated_metrics.per_node.map((entry, i) => (
                      <tr key={i}>
                        <td className="px-4 py-2 font-mono text-xs text-gray-600">
                          {entry.node_id}
                        </td>
                        <td className="px-4 py-2 text-gray-700">
                          {entry.loss != null ? entry.loss.toFixed(4) : "—"}
                        </td>
                        <td className="px-4 py-2 text-gray-700">
                          {entry.accuracy != null
                            ? `${(entry.accuracy * 100).toFixed(2)}%`
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Error summary (failed jobs) */}
      {job.status === "failed" && job.error_summary && (
        <section className="rounded-lg border border-red-200 bg-red-50 p-6 shadow-sm">
          <h2 className="mb-4 text-base font-semibold text-red-800">Job Failed</h2>
          <div className="space-y-2">
            {Object.entries(job.error_summary).map(([taskId, msg]) => (
              <div key={taskId} className="rounded-md bg-white p-3 border border-red-100">
                <p className="font-mono text-xs text-gray-500">Task: {taskId}</p>
                <p className="mt-1 text-sm text-red-700">{msg}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-task progress */}
      {job.tasks && job.tasks.length > 0 && (
        <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-base font-semibold text-gray-700">
            Tasks ({job.tasks.length})
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Shard</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Status</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Assigned Node</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Epoch</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Loss</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Accuracy</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Error</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {job.tasks
                  .slice()
                  .sort((a, b) => a.shard_index - b.shard_index)
                  .map((task) => (
                    <tr key={task.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-3 font-medium text-gray-700">
                        {task.shard_index}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={task.status} />
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-gray-500">
                        {task.node_id ? task.node_id.slice(0, 8) + "…" : "—"}
                      </td>
                      <td className="px-4 py-3 text-gray-600">
                        {task.current_epoch != null ? task.current_epoch : "—"}
                      </td>
                      <td className="px-4 py-3 text-gray-600">
                        {task.latest_loss != null
                          ? task.latest_loss.toFixed(4)
                          : "—"}
                      </td>
                      <td className="px-4 py-3 text-gray-600">
                        {task.latest_accuracy != null
                          ? `${(task.latest_accuracy * 100).toFixed(2)}%`
                          : "—"}
                      </td>
                      <td className="px-4 py-3 text-red-600 text-xs max-w-[200px] truncate" title={task.error_message || undefined}>
                        {task.error_message || "—"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Artifacts */}
      {artifactsError && (
        <ErrorMessage message={`Failed to load artifacts: ${artifactsError.message}`} />
      )}

      {artifacts && artifacts.length > 0 && (
        <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-base font-semibold text-gray-700">
            Artifacts ({artifacts.length})
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Type</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Storage Path</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Task</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Epoch</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {artifacts.map((artifact) => (
                  <tr key={artifact.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3">
                      <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                        {artifact.artifact_type}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-gray-600 max-w-[300px] truncate" title={artifact.storage_path}>
                      <a
                        href={`${API_URL}/storage/${artifact.storage_path}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline"
                        title={artifact.storage_path}
                      >
                        {artifact.storage_path}
                      </a>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-gray-500">
                      {artifact.task_id.slice(0, 8)}…
                    </td>
                    <td className="px-4 py-3 text-gray-600">
                      {artifact.epoch != null ? artifact.epoch : "—"}
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">
                      {formatDate(artifact.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
