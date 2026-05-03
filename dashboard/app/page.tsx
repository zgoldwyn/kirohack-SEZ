"use client";

import useSWR from "swr";
import Link from "next/link";
import { fetcher } from "@/lib/api";
import type { MonitoringSummary } from "@/lib/types";
import ErrorMessage from "@/components/ErrorMessage";

interface StatCardProps {
  label: string;
  value: number | undefined;
  icon: string;
  color: "emerald" | "amber" | "red" | "blue" | "slate" | "indigo";
  href?: string;
}

const COLOR_MAP = {
  emerald: {
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    icon: "bg-emerald-100 text-emerald-600",
    value: "text-emerald-700",
  },
  amber: {
    bg: "bg-amber-50",
    border: "border-amber-200",
    icon: "bg-amber-100 text-amber-600",
    value: "text-amber-700",
  },
  red: {
    bg: "bg-red-50",
    border: "border-red-200",
    icon: "bg-red-100 text-red-600",
    value: "text-red-700",
  },
  blue: {
    bg: "bg-blue-50",
    border: "border-blue-200",
    icon: "bg-blue-100 text-blue-600",
    value: "text-blue-700",
  },
  slate: {
    bg: "bg-slate-50",
    border: "border-slate-200",
    icon: "bg-slate-100 text-slate-600",
    value: "text-slate-700",
  },
  indigo: {
    bg: "bg-indigo-50",
    border: "border-indigo-200",
    icon: "bg-indigo-100 text-indigo-600",
    value: "text-indigo-700",
  },
};

function StatCard({ label, value, icon, color, href }: StatCardProps) {
  const c = COLOR_MAP[color];
  const content = (
    <div
      className={`rounded-xl border ${c.border} ${c.bg} p-5 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5`}
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <span
          className={`flex h-8 w-8 items-center justify-center rounded-lg text-sm ${c.icon}`}
        >
          {icon}
        </span>
      </div>
      <p className={`mt-3 text-3xl font-bold ${c.value}`}>{value ?? "—"}</p>
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

function LoadingSkeleton() {
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-28 rounded-xl animate-shimmer" />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-28 rounded-xl animate-shimmer" />
        ))}
      </div>
    </div>
  );
}

export default function OverviewPage() {
  const { data, error, isLoading } = useSWR<MonitoringSummary>(
    "/api/monitoring/summary",
    fetcher,
    { refreshInterval: 10_000 }
  );

  return (
    <div>
      {/* Hero header */}
      <div className="mb-8 rounded-2xl bg-gradient-to-br from-indigo-600 via-indigo-700 to-slate-800 p-8 text-white shadow-xl shadow-indigo-500/10">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              System Overview
            </h1>
            <p className="mt-2 text-indigo-200">
              Live summary of your distributed training cluster
            </p>
            <div className="mt-3 flex items-center gap-2 text-xs text-indigo-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse-dot" />
              Auto-refreshes every 10 seconds
            </div>
          </div>
          <Link
            href="/jobs/new"
            className="hidden sm:inline-flex rounded-lg bg-white/15 px-5 py-2.5 text-sm font-medium text-white backdrop-blur-sm hover:bg-white/25 transition-all duration-200"
          >
            + Submit Job
          </Link>
        </div>
      </div>

      {error && <ErrorMessage message={error.message} />}

      {isLoading && !data && <LoadingSkeleton />}

      {data && (
        <>
          {/* Nodes section */}
          <section className="mb-8">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-800">
                Worker Nodes
              </h2>
              <Link
                href="/nodes"
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500 transition-colors"
              >
                View all →
              </Link>
            </div>
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

          {/* Jobs section */}
          <section className="mb-8">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-800">
                Training Jobs
              </h2>
              <Link
                href="/jobs"
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500 transition-colors"
              >
                View all →
              </Link>
            </div>
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
            <h2 className="mb-4 text-lg font-semibold text-slate-800">
              Quick Actions
            </h2>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/nodes"
                className="rounded-xl bg-white border border-slate-200 px-5 py-3 text-sm font-medium text-slate-700 shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200"
              >
                ⬡ View Nodes
              </Link>
              <Link
                href="/jobs"
                className="rounded-xl bg-white border border-slate-200 px-5 py-3 text-sm font-medium text-slate-700 shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200"
              >
                ▦ View Jobs
              </Link>
              <Link
                href="/jobs/new"
                className="rounded-xl bg-indigo-600 px-5 py-3 text-sm font-medium text-white shadow-md shadow-indigo-500/20 hover:bg-indigo-500 hover:shadow-lg hover:shadow-indigo-500/30 hover:-translate-y-0.5 transition-all duration-200"
              >
                + Submit New Job
              </Link>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
