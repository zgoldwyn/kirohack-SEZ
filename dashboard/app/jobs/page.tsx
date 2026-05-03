"use client";

import useSWR from "swr";
import Link from "next/link";
import { fetcher, formatDate } from "@/lib/api";
import type { Job } from "@/lib/types";
import StatusBadge from "@/components/StatusBadge";
import ErrorMessage from "@/components/ErrorMessage";

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      {[...Array(3)].map((_, i) => (
        <div key={i} className="h-16 rounded-xl animate-shimmer" />
      ))}
    </div>
  );
}

export default function JobsPage() {
  const {
    data: jobs,
    error,
    isLoading,
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
          <Link
            href="/jobs/new"
            className="mt-4 inline-block rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white shadow-md shadow-indigo-500/20 hover:bg-indigo-500 transition-all duration-200"
          >
            Submit a Job
          </Link>
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead>
              <tr className="bg-slate-50/80">
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Job
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Status
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Model
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Dataset
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Shards
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Created
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Completed
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {jobs.map((job) => (
                <tr key={job.id} className="table-row-hover">
                  <td className="px-5 py-4">
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
                  </td>
                  <td className="px-5 py-4">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-5 py-4 text-slate-600">
                    <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium">
                      {job.model_type}
                    </span>
                  </td>
                  <td className="px-5 py-4 text-slate-600">{job.dataset_name}</td>
                  <td className="px-5 py-4 text-slate-600">{job.shard_count}</td>
                  <td className="px-5 py-4 text-xs text-slate-400">
                    {formatDate(job.created_at)}
                  </td>
                  <td className="px-5 py-4 text-xs text-slate-400">
                    {formatDate(job.completed_at)}
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
