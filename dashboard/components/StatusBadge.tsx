"use client";

import type { NodeStatus, JobStatus, TaskStatus } from "@/lib/types";

type Status = NodeStatus | JobStatus | TaskStatus;

const STATUS_STYLES: Record<string, string> = {
  // Node statuses
  idle: "bg-green-100 text-green-800",
  busy: "bg-amber-100 text-amber-800",
  offline: "bg-red-100 text-red-800",
  // Job / task statuses
  queued: "bg-gray-100 text-gray-700",
  running: "bg-blue-100 text-blue-800",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  assigned: "bg-purple-100 text-purple-800",
};

interface StatusBadgeProps {
  status: Status;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? "bg-gray-100 text-gray-700";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${style}`}
    >
      <span
        className="h-1.5 w-1.5 rounded-full bg-current opacity-70"
        aria-hidden="true"
      />
      {status}
    </span>
  );
}
