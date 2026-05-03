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
                label="Online"
                value={(data.idle_nodes ?? 0) + (data.busy_nodes ?? 0)}
                icon="◉"
                color="emerald"
                href="/nodes"
              />
              <StatCard
                label="Idle"
                value={data.idle_nodes}
                icon="◎"
                color="emerald"
                href="/nodes"
              />
              <StatCard
                label="Busy"
                value={data.busy_nodes}
                icon="⟳"
                color="amber"
                href="/nodes"
              />
              <StatCard
                label="Offline"
                value={data.offline_nodes}
                icon="○"
                color="red"
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
                value={data.running_jobs}
                icon="▶"
                color="blue"
                href="/jobs"
              />
              <StatCard
                label="Queued"
                value={data.queued_jobs}
                icon="◷"
                color="slate"
                href="/jobs"
              />
              <StatCard
                label="Completed"
                value={data.completed_jobs}
                icon="✓"
                color="emerald"
                href="/jobs"
              />
              <StatCard
                label="Failed"
                value={data.failed_jobs}
                icon="✕"
                color="red"
                href="/jobs"
              />
            </div>
          </section>

          {/* Quick actions */}
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
