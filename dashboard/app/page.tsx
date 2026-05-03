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
                value={data.nodes.online}
                color="text-green-600"
                href="/nodes"
              />
              <StatCard
                label="Idle"
                value={data.nodes.idle}
                color="text-green-600"
                href="/nodes"
              />
              <StatCard
                label="Busy"
                value={data.nodes.busy}
                color="text-amber-600"
                href="/nodes"
              />
              <StatCard
                label="Offline"
                value={data.nodes.offline}
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
                value={data.jobs.running}
                color="text-blue-600"
                href="/jobs"
              />
              <StatCard
                label="Queued"
                value={data.jobs.queued}
                color="text-gray-600"
                href="/jobs"
              />
              <StatCard
                label="Completed"
                value={data.jobs.completed}
                color="text-green-600"
                href="/jobs"
              />
              <StatCard
                label="Failed"
                value={data.jobs.failed}
                color="text-red-600"
                href="/jobs"
              />
            </div>
          </section>

          {/* Running jobs with round progress */}
          {data.running_jobs && data.running_jobs.length > 0 && (
            <section className="mb-8">
              <h2 className="mb-4 text-lg font-semibold text-gray-700">Active Training</h2>
              <div className="space-y-3">
                {data.running_jobs.map((rj) => {
                  const current = rj.current_round ?? 0;
                  const total = rj.total_rounds ?? 0;
                  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
                  return (
                    <Link
                      key={rj.job_id}
                      href={`/jobs/${rj.job_id}`}
                      className="block rounded-lg border border-blue-200 bg-blue-50 p-4 shadow-sm hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-mono text-xs text-blue-700">
                          {rj.job_id.slice(0, 8)}…
                        </span>
                        <span className="font-medium text-blue-800">
                          Round {current} / {total}
                        </span>
                      </div>
                      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-blue-200">
                        <div
                          className="h-full rounded-full bg-blue-600 transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </Link>
                  );
                })}
              </div>
            </section>
          )}

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
