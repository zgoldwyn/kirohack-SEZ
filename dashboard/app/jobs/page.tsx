"use client";

import useSWR from "swr";
import Link from "next/link";
import { fetcher, formatDate } from "@/lib/api";
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

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Jobs</h1>
          <p className="mt-1 text-sm text-gray-500">
            All training jobs — refreshes every 10 seconds.
          </p>
        </div>
        <Link
          href="/jobs/new"
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-500 transition-colors"
        >
          + New Job
        </Link>
      </div>

      {error && <ErrorMessage message={error.message} />}

      {isLoading && !jobs && (
        <div className="text-sm text-gray-500">Loading jobs…</div>
      )}

      {jobs && jobs.length === 0 && (
        <div className="rounded-lg border border-dashed border-gray-300 bg-white p-12 text-center">
          <p className="text-sm text-gray-500">No jobs yet.</p>
          <Link
            href="/jobs/new"
            className="mt-4 inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
          >
            Submit your first job
          </Link>
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
            <tbody className="divide-y divide-gray-100">
              {jobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3">
                    <Link
                      href={`/jobs/${job.id}`}
                      className="font-medium text-blue-600 hover:text-blue-800 hover:underline"
                    >
                      {job.job_name || (
                        <span className="font-mono text-xs text-gray-500">
                          {job.id.slice(0, 8)}…
                        </span>
                      )}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
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
                  <td className="px-4 py-3 text-gray-500 text-xs">
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
