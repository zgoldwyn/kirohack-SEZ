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

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-10 w-64 rounded-xl animate-shimmer" />
      <div className="h-48 rounded-xl animate-shimmer" />
      <div className="h-64 rounded-xl animate-shimmer" />
    </div>
  );
}

export default function JobDetailPage({ params }: JobDetailPageProps) {
  const { id } = use(params);

  const { data: job, error: jobError } = useSWR<JobDetail>(
    `/api/jobs/${id}`,
    fetcher,
    {
      refreshInterval: (data) =>
        data?.status === "running" || data?.status === "queued"
          ? 5_000
          : 30_000,
    }
  );

  const { data: artifacts, error: artifactsError } = useSWR<Artifact[]>(
    job ? `/api/jobs/${id}/artifacts` : null,
    fetcher,
    { refreshInterval: 30_000 }
  );

  if (jobError) return <ErrorMessage message={jobError.message} />;
  if (!job) return <LoadingSkeleton />;

  const hyperparams = job.hyperparameters || {};

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-900">
              {job.job_name || "Unnamed Job"}
            </h1>
            <StatusBadge status={job.status} />
          </div>
          <p className="mt-1 font-mono text-xs text-slate-400">{job.id}</p>
        </div>
        <Link
          href="/jobs"
          className="text-sm font-medium text-slate-400 hover:text-slate-600 transition-colors"
        >
          ← Back to Jobs
        </Link>
      </div>

      {/* Job metadata */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Job Details
        </h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3 lg:grid-cols-4">
          <div>
            <dt className="text-slate-400">Model</dt>
            <dd className="mt-0.5 font-medium text-slate-800">
              <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium">
                {job.model_type}
              </span>
            </dd>
          </div>
          <div>
            <dt className="text-slate-400">Dataset</dt>
            <dd className="mt-0.5 font-medium text-slate-800">
              {job.dataset_name}
            </dd>
          </div>
          <div>
            <dt className="text-slate-400">Shards</dt>
            <dd className="mt-0.5 font-medium text-slate-800">
              {job.shard_count}
            </dd>
          </div>
          <div>
            <dt className="text-slate-400">Created</dt>
            <dd className="mt-0.5 font-medium text-slate-800">
              {formatDate(job.created_at)}
            </dd>
          </div>
          {job.started_at && (
            <div>
              <dt className="text-slate-400">Started</dt>
              <dd className="mt-0.5 font-medium text-slate-800">
                {formatDate(job.started_at)}
              </dd>
            </div>
          )}
          {job.completed_at && (
            <div>
              <dt className="text-slate-400">Completed</dt>
              <dd className="mt-0.5 font-medium text-slate-800">
                {formatDate(job.completed_at)}
              </dd>
            </div>
          )}
        </dl>

        {/* Hyperparameters */}
        {Object.keys(hyperparams).length > 0 && (
          <div className="mt-5 border-t border-slate-100 pt-5">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Hyperparameters
            </h3>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-5">
              {Object.entries(hyperparams).map(([key, val]) => (
                <div key={key}>
                  <dt className="text-slate-400">{key}</dt>
                  <dd className="font-mono text-sm font-medium text-slate-700">
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
        <section className="rounded-xl border border-emerald-200 bg-gradient-to-br from-emerald-50 to-white p-6 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-emerald-700">
            Aggregated Results
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div className="rounded-xl bg-white border border-emerald-100 p-5 shadow-sm">
              <p className="text-xs font-medium text-slate-400">Mean Loss</p>
              <p className="mt-2 text-3xl font-bold text-slate-800">
                {job.aggregated_metrics.mean_loss != null
                  ? job.aggregated_metrics.mean_loss.toFixed(4)
                  : "—"}
              </p>
            </div>
            <div className="rounded-xl bg-white border border-emerald-100 p-5 shadow-sm">
              <p className="text-xs font-medium text-slate-400">
                Mean Accuracy
              </p>
              <p className="mt-2 text-3xl font-bold text-emerald-600">
                {job.aggregated_metrics.mean_accuracy != null
                  ? `${(job.aggregated_metrics.mean_accuracy * 100).toFixed(2)}%`
                  : "—"}
              </p>
            </div>
          </div>

          {/* Per-node breakdown */}
          {job.aggregated_metrics.per_node &&
            job.aggregated_metrics.per_node.length > 0 && (
              <div className="mt-5">
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-emerald-600">
                  Per-Node Breakdown
                </h3>
                <div className="overflow-x-auto rounded-lg border border-emerald-100 bg-white">
                  <table className="min-w-full divide-y divide-slate-100 text-sm">
                    <thead>
                      <tr className="bg-slate-50/50">
                        <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500">
                          Node ID
                        </th>
                        <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500">
                          Loss
                        </th>
                        <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500">
                          Accuracy
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {job.aggregated_metrics.per_node.map((entry, i) => (
                        <tr key={i} className="table-row-hover">
                          <td className="px-4 py-2.5 font-mono text-xs text-slate-500">
                            {entry.node_id}
                          </td>
                          <td className="px-4 py-2.5 text-slate-700">
                            {entry.loss != null
                              ? entry.loss.toFixed(4)
                              : "—"}
                          </td>
                          <td className="px-4 py-2.5 text-slate-700">
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
        <section className="rounded-xl border border-red-200 bg-gradient-to-br from-red-50 to-white p-6 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-red-700">
            Job Failed
          </h2>
          <div className="space-y-2">
            {Object.entries(job.error_summary).map(([taskId, msg]) => (
              <div
                key={taskId}
                className="rounded-lg bg-white p-3 border border-red-100"
              >
                <p className="font-mono text-xs text-slate-400">
                  Task: {taskId}
                </p>
                <p className="mt-1 text-sm text-red-700">{msg}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-task progress */}
      {job.tasks && job.tasks.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Tasks ({job.tasks.length})
          </h2>
          <div className="overflow-x-auto rounded-lg border border-slate-100">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead>
                <tr className="bg-slate-50/80">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Shard
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Node
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Epoch
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Loss
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Accuracy
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Error
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {job.tasks
                  .slice()
                  .sort((a, b) => a.shard_index - b.shard_index)
                  .map((task) => (
                    <tr key={task.id} className="table-row-hover">
                      <td className="px-4 py-3 font-medium text-slate-700">
                        {task.shard_index}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={task.status} />
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-400">
                        {task.node_id
                          ? task.node_id.slice(0, 8) + "…"
                          : "—"}
                      </td>
                      <td className="px-4 py-3 text-slate-600">
                        {task.current_epoch != null
                          ? task.current_epoch
                          : "—"}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-600">
                        {task.latest_loss != null
                          ? task.latest_loss.toFixed(4)
                          : "—"}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-600">
                        {task.latest_accuracy != null
                          ? `${(task.latest_accuracy * 100).toFixed(2)}%`
                          : "—"}
                      </td>
                      <td
                        className="px-4 py-3 text-red-500 text-xs max-w-[200px] truncate"
                        title={task.error_message || undefined}
                      >
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
        <ErrorMessage
          message={`Failed to load artifacts: ${artifactsError.message}`}
        />
      )}

      {artifacts && artifacts.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Artifacts ({artifacts.length})
          </h2>
          <div className="overflow-x-auto rounded-lg border border-slate-100">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead>
                <tr className="bg-slate-50/80">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Storage Path
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Task
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Epoch
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {artifacts.map((artifact) => (
                  <tr key={artifact.id} className="table-row-hover">
                    <td className="px-4 py-3">
                      <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs font-medium text-slate-600">
                        {artifact.artifact_type}
                      </span>
                    </td>
                    <td
                      className="px-4 py-3 font-mono text-xs text-slate-500 max-w-[300px] truncate"
                      title={artifact.storage_path}
                    >
                      <a
                        href={`${API_URL}/storage/${artifact.storage_path}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-indigo-600 hover:text-indigo-500 hover:underline"
                        title={artifact.storage_path}
                      >
                        {artifact.storage_path}
                      </a>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-400">
                      {artifact.task_id.slice(0, 8)}…
                    </td>
                    <td className="px-4 py-3 text-slate-600">
                      {artifact.epoch != null ? artifact.epoch : "—"}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-400">
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
