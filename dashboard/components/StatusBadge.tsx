"use client";

import type { NodeStatus, JobStatus, TaskStatus } from "@/lib/types";

type Status = NodeStatus | JobStatus | TaskStatus;

const STATUS_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  // Node statuses
  idle: { bg: "bg-emerald-50 border-emerald-200", text: "text-emerald-700", dot: "bg-emerald-500" },
  busy: { bg: "bg-amber-50 border-amber-200", text: "text-amber-700", dot: "bg-amber-500" },
  offline: { bg: "bg-red-50 border-red-200", text: "text-red-700", dot: "bg-red-500" },
  // Job / task statuses
  queued: { bg: "bg-slate-50 border-slate-200", text: "text-slate-600", dot: "bg-slate-400" },
  running: { bg: "bg-blue-50 border-blue-200", text: "text-blue-700", dot: "bg-blue-500" },
  completed: { bg: "bg-emerald-50 border-emerald-200", text: "text-emerald-700", dot: "bg-emerald-500" },
  failed: { bg: "bg-red-50 border-red-200", text: "text-red-700", dot: "bg-red-500" },
  assigned: { bg: "bg-violet-50 border-violet-200", text: "text-violet-700", dot: "bg-violet-500" },
};

const DEFAULT_STYLE = { bg: "bg-slate-50 border-slate-200", text: "text-slate-600", dot: "bg-slate-400" };

interface StatusBadgeProps {
  status: Status;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? DEFAULT_STYLE;
  const isLive = status === "running" || status === "busy";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${style.bg} ${style.text}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${style.dot} ${isLive ? "animate-pulse-dot" : ""}`}
        aria-hidden="true"
      />
      {status}
    </span>
  );
}
