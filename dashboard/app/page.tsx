"use client";

import useSWR from "swr";
import Link from "next/link";
import { fetcher } from "@/lib/api";
import type { MonitoringSummary } from "@/lib/types";
import ErrorMessage from "@/components/ErrorMessage";

interface StatCardProps {
  label: string;
  value: number | undefined;
  color?: string;
  href?: string;
}

function StatCard({ label, value, color = "text-gray-900", href }: StatCardProps) {
  const content = (
    <div className="rounded-lg bg-white border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p className={`mt-2 text-4xl font-bold ${color}`}>
        {value ?? "—"}
      </p>
    </div>
  );

  if (href) {
    return (
      <Link href={href} className="block">
        {content}
      </Link>
    );
  }
  return content;
}

export default function OverviewPage() {
  const { data, error, isLoading } = useSWR<MonitoringSummary>(
    "/api/monitoring/summary",
    fetcher,
    { refreshInterval: 10_000 }
  );

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">System Overview</h1>
        <p className="mt-1 text-sm text-gray-500">
          Live summary of nodes and training jobs — refreshes every 10 seconds.
        </p>
      </div>

      {error && <ErrorMessage message={error.message} />}

      {isLoading && !data && (
        <div className="text-sm text-gray-500">Loading summary…</div>
      )}

      {data && (
        <>
          <section className="mb-8">
            <h2 className="mb-4 text-lg font-semibold text-gray-700">Nodes</h2>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatCard
                label="Online Nodes"
                value={(data.idle_nodes ?? 0) + (data.busy_nodes ?? 0)}
                color="text-green-600"
                href="/nodes"
              />
              <StatCard
                label="Idle"
                value={data.idle_nodes}
                color="text-green-600"
                href="/nodes"
              />
              <StatCard
                label="Busy"
                value={data.busy_nodes}
                color="text-amber-600"
                href="/nodes"
              />
              <StatCard
                label="Offline"
                value={data.offline_nodes}
                color="text-red-600"
                href="/nodes"
              />
            </div>
          </section>

          <section className="mb-8">
            <h2 className="mb-4 text-lg font-semibold text-gray-700">Jobs</h2>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatCard
                label="Running"
                value={data.running_jobs}
                color="text-blue-600"
                href="/jobs"
              />
              <StatCard
                label="Queued"
                value={data.queued_jobs}
                color="text-gray-600"
                href="/jobs"
              />
              <StatCard
                label="Completed"
                value={data.completed_jobs}
                color="text-green-600"
                href="/jobs"
              />
              <StatCard
                label="Failed"
                value={data.failed_jobs}
                color="text-red-600"
                href="/jobs"
              />
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-lg font-semibold text-gray-700">Quick Links</h2>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/nodes"
                className="rounded-md bg-white border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 transition-colors"
              >
                View All Nodes →
              </Link>
              <Link
                href="/jobs"
                className="rounded-md bg-white border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 transition-colors"
              >
                View All Jobs →
              </Link>
              <Link
                href="/jobs/new"
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-500 transition-colors"
              >
                Submit New Job →
              </Link>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
