"use client";

import useSWR from "swr";
import { fetcher, formatDate, formatRam } from "@/lib/api";
import type { Node } from "@/lib/types";
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

export default function NodesPage() {
  const {
    data: nodes,
    error,
    isLoading,
  } = useSWR<Node[]>("/api/nodes", fetcher, { refreshInterval: 10_000 });

  const onlineCount = nodes?.filter((n) => n.status !== "offline").length ?? 0;

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Worker Nodes</h1>
          <p className="mt-1 text-sm text-slate-500">
            All registered workers — refreshes every 10 seconds
          </p>
        </div>
        {nodes && (
          <div className="flex items-center gap-3">
            <span className="rounded-lg bg-emerald-50 border border-emerald-200 px-3 py-1.5 text-xs font-semibold text-emerald-700">
              {onlineCount} online
            </span>
            <span className="rounded-lg bg-slate-50 border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600">
              {nodes.length} total
            </span>
          </div>
        )}
      </div>

      {error && <ErrorMessage message={error.message} />}

      {isLoading && !nodes && <LoadingSkeleton />}

      {nodes && nodes.length === 0 && (
        <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white p-16 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-xl text-slate-400">
            ⬡
          </div>
          <p className="text-sm font-medium text-slate-600">
            No nodes registered yet
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Start a worker to see it here
          </p>
        </div>
      )}

      {nodes && nodes.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead>
              <tr className="bg-slate-50/80">
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Node
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  GPU
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  RAM
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  CPUs
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Status
                </th>
                <th className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Last Heartbeat
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {nodes.map((node) => (
                <tr key={node.id} className="table-row-hover">
                  <td className="px-5 py-4">
                    <div>
                      <p className="font-medium text-slate-800">
                        {node.hostname}
                      </p>
                      <p className="mt-0.5 font-mono text-xs text-slate-400">
                        {node.node_id}
                      </p>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-slate-600">
                    {node.gpu_model ? (
                      <div>
                        <p className="text-sm">{node.gpu_model}</p>
                        {node.vram_mb && (
                          <p className="text-xs text-slate-400">
                            {formatRam(node.vram_mb)} VRAM
                          </p>
                        )}
                      </div>
                    ) : (
                      <span className="text-slate-300">—</span>
                    )}
                  </td>
                  <td className="px-5 py-4 text-slate-600">
                    {formatRam(node.ram_mb)}
                  </td>
                  <td className="px-5 py-4 text-slate-600">
                    {node.cpu_cores}
                  </td>
                  <td className="px-5 py-4">
                    <StatusBadge status={node.status} />
                  </td>
                  <td className="px-5 py-4 text-xs text-slate-400">
                    {formatDate(node.last_heartbeat)}
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
