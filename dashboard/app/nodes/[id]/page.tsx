"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { use } from "react";
import { fetcher, formatDate, formatRam, deleteRequest } from "@/lib/api";
import type { Node, TaskStatus } from "@/lib/types";
import StatusBadge from "@/components/StatusBadge";
import ErrorMessage from "@/components/ErrorMessage";

interface NodeTask {
  id: string;
  job_id: string;
  job_name: string | null;
  shard_index: number;
  status: TaskStatus;
  checkpoint_path: string | null;
  error_message: string | null;
  assigned_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

interface NodeDetail extends Node {
  tasks: NodeTask[];
}

interface NodeDetailPageProps {
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

function DeleteNodeButton({ node, onDeleted }: { node: NodeDetail; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  if (node.status !== "offline") return null;

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await deleteRequest(`/api/nodes/${node.id}`);
      onDeleted();
    } catch {
      alert("Failed to delete node");
    } finally {
      setDeleting(false);
      setConfirming(false);
    }
  };

  if (confirming) {
    return (
      <div className="flex items-center gap-2">
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50 transition-colors"
        >
          {deleting ? "Removing…" : "Confirm Remove"}
        </button>
        <button
          onClick={() => setConfirming(false)}
          className="rounded-lg bg-slate-100 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200 transition-colors"
        >
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      className="rounded-lg bg-slate-100 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 hover:text-red-700 transition-colors"
    >
      Remove Node
    </button>
  );
}

export default function NodeDetailPage({ params }: NodeDetailPageProps) {
  const { id } = use(params);
  const router = useRouter();

  const { data: node, error } = useSWR<NodeDetail>(
    `/api/nodes/${id}`,
    fetcher,
    { refreshInterval: 10_000 }
  );

  if (error) return <ErrorMessage message={error.message} />;
  if (!node) return <LoadingSkeleton />;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-900">
              {node.hostname}
            </h1>
            <StatusBadge status={node.status} />
          </div>
          <p className="mt-1 font-mono text-xs text-slate-400">{node.node_id}</p>
        </div>
        <div className="flex items-center gap-3">
          <DeleteNodeButton node={node} onDeleted={() => router.push("/nodes")} />
          <Link
            href="/nodes"
            className="text-sm font-medium text-slate-400 hover:text-slate-600 transition-colors"
          >
            ← Back to Nodes
          </Link>
        </div>
      </div>

      {/* Hardware info */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Hardware &amp; Environment
        </h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3 lg:grid-cols-4">
          <div>
            <dt className="text-slate-400">CPU Cores</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{node.cpu_cores}</dd>
          </div>
          <div>
            <dt className="text-slate-400">RAM</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{formatRam(node.ram_mb)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Disk</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{formatRam(node.disk_mb)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">GPU</dt>
            <dd className="mt-0.5 font-medium text-slate-800">
              {node.gpu_model || <span className="text-slate-300">None</span>}
            </dd>
          </div>
          {node.vram_mb && (
            <div>
              <dt className="text-slate-400">VRAM</dt>
              <dd className="mt-0.5 font-medium text-slate-800">{formatRam(node.vram_mb)}</dd>
            </div>
          )}
          <div>
            <dt className="text-slate-400">OS</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{node.os}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Python</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{node.python_version}</dd>
          </div>
          <div>
            <dt className="text-slate-400">PyTorch</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{node.pytorch_version}</dd>
          </div>
        </dl>
      </section>

      {/* Timestamps */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Activity
        </h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3">
          <div>
            <dt className="text-slate-400">Registered</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{formatDate(node.created_at)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Last Heartbeat</dt>
            <dd className="mt-0.5 font-medium text-slate-800">{formatDate(node.last_heartbeat)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Tasks Completed</dt>
            <dd className="mt-0.5 font-medium text-slate-800">
              {node.tasks.filter((t) => t.status === "completed").length}
            </dd>
          </div>
        </dl>
      </section>

      {/* Task history */}
      {node.tasks.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Task History ({node.tasks.length})
          </h2>
          <div className="overflow-x-auto rounded-lg border border-slate-100">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead>
                <tr className="bg-slate-50/80">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Job
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Shard
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Started
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Completed
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Error
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {node.tasks.map((task) => (
                  <tr key={task.id} className="table-row-hover">
                    <td className="px-4 py-3">
                      {task.status === "failed" ? (
                        <span className="font-medium text-slate-600">
                          {task.job_name || (
                            <span className="font-mono text-xs text-slate-400">
                              {task.job_id.slice(0, 8)}…
                            </span>
                          )}
                        </span>
                      ) : (
                        <Link
                          href={`/jobs/${task.job_id}`}
                          className="font-medium text-indigo-600 hover:text-indigo-500 transition-colors"
                        >
                          {task.job_name || (
                            <span className="font-mono text-xs">
                              {task.job_id.slice(0, 8)}…
                            </span>
                          )}
                        </Link>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-600">{task.shard_index}</td>
                    <td className="px-4 py-3">
                      <StatusBadge status={task.status} />
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-400">
                      {formatDate(task.started_at)}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-400">
                      {formatDate(task.completed_at)}
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

      {node.tasks.length === 0 && (
        <section className="rounded-2xl border-2 border-dashed border-slate-200 bg-white p-12 text-center">
          <p className="text-sm font-medium text-slate-600">No tasks assigned yet</p>
          <p className="mt-1 text-xs text-slate-400">
            This node hasn&apos;t been assigned any training tasks
          </p>
        </section>
      )}
    </div>
  );
}
