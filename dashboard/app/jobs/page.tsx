"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { fetcher, formatDate, deleteRequest } from "@/lib/api";
import type { Job } from "@/lib/types";
import StatusBadge from "@/components/StatusBadge";
import ErrorMessage from "@/components/ErrorMessage";

function RoundProgress({ job }: { job: Job }) {
  if (job.status !== "running" || job.current_round == null || job.total_rounds == null) {
    return <span className="text-gray-400">—</span>;
  }
  return (
    <span className="text-gray-600">
      {job.current_round} / {job.total_rounds}
    </span>
  );
}

export default function JobsPage() {
  const { data: jobs, error, isLoading } = useSWR<Job[]>(
    "/api/jobs",
    fetcher,
    { refreshInterval: 10_000 }
  );
}

function DeleteJobButton({ job, onDeleted }: { job: Job; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const canDelete = job.status === "completed" || job.status === "failed";
  if (!canDelete) return null;

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await deleteRequest(`/api/jobs/${job.id}`);
      onDeleted();
    } catch {
      alert("Failed to delete job");
    } finally {
      setDeleting(false);
      setConfirming(false);
    }
  };

  if (confirming) {
    return (
      <div className="flex items-center gap-1.5">
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="rounded-md bg-red-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-50 transition-colors"
        >
          {deleting ? "…" : "Confirm"}
        </button>
        <button
          onClick={() => setConfirming(false)}
          className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-200 transition-colors"
        >
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-red-600 hover:bg-red-50 hover:text-red-700 transition-colors"
      title="Delete job"
    >
      Delete
    </button>
  );
}

export default function JobsPage() {
  const {
    data: jobs,
    error,
    isLoading,
    mutate,
  } = useSWR<Job[]>("/api/jobs", fetcher, { refreshInterval: 10_000 });

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Training Jobs</h1>
          <p className="mt-1 text-sm text-slate-500">
            All training jobs — refreshes every 10 seconds
          </p>
        </div>
        <Link
          href="/jobs/new"
          className="rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white shadow-md shadow-indigo-500/20 hover:bg-indigo-500 hover:shadow-lg transition-all duration-200"
        >
          + New Job
        </Link>
      </div>

      {error && <ErrorMessage message={error.message} />}

      {isLoading && !jobs && <LoadingSkeleton />}

      {jobs && jobs.length === 0 && (
        <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white p-16 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-xl text-slate-400">
            ▦
          </div>
          <p className="text-sm font-medium text-slate-600">No jobs yet</p>
          <p className="mt-1 text-xs text-slate-400">
            Submit your first training job to get started
          </p>
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Job</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Status</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Model</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Dataset</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Workers</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Round</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Created</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Completed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {jobs.map((job) => (
                <tr key={job.id} className="table-row-hover">
                  <td className="px-5 py-4">
                    {job.status === "failed" ? (
                      <span className="font-medium text-slate-600">
                        {job.job_name || (
                          <span className="font-mono text-xs text-slate-400">
                            {job.id.slice(0, 8)}…
                          </span>
                        )}
                      </span>
                    ) : (
                      <Link
                        href={`/jobs/${job.id}`}
                        className="font-medium text-indigo-600 hover:text-indigo-500 transition-colors"
                      >
                        {job.job_name || (
                          <span className="font-mono text-xs text-slate-400">
                            {job.id.slice(0, 8)}…
                          </span>
                        )}
                      </Link>
                    )}
                  </td>
                  <td className="px-5 py-4">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-3 text-gray-600">{job.model_type}</td>
                  <td className="px-4 py-3 text-gray-600">{job.dataset_name}</td>
                  <td className="px-4 py-3 text-gray-600">{job.shard_count}</td>
                  <td className="px-4 py-3">
                    <RoundProgress job={job} />
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {formatDate(job.created_at)}
                  </td>
                  <td className="px-5 py-4 text-xs text-slate-400">
                    {formatDate(job.completed_at)}
                  </td>
                  <td className="px-5 py-4 text-right">
                    <DeleteJobButton job={job} onDeleted={() => mutate()} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
