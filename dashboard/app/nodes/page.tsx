"use client";

import useSWR from "swr";
import { fetcher, formatDate, formatRam } from "@/lib/api";
import type { Node } from "@/lib/types";
import StatusBadge from "@/components/StatusBadge";
import ErrorMessage from "@/components/ErrorMessage";

export default function NodesPage() {
  const { data: nodes, error, isLoading } = useSWR<Node[]>(
    "/api/nodes",
    fetcher,
    { refreshInterval: 10_000 }
  );

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Nodes</h1>
          <p className="mt-1 text-sm text-gray-500">
            All registered worker nodes — refreshes every 10 seconds.
          </p>
        </div>
        {nodes && (
          <span className="text-sm text-gray-500">{nodes.length} node{nodes.length !== 1 ? "s" : ""}</span>
        )}
      </div>

      {error && <ErrorMessage message={error.message} />}

      {isLoading && !nodes && (
        <div className="text-sm text-gray-500">Loading nodes…</div>
      )}

      {nodes && nodes.length === 0 && (
        <div className="rounded-lg border border-dashed border-gray-300 bg-white p-12 text-center text-sm text-gray-500">
          No nodes registered yet. Start a worker to see it here.
        </div>
      )}

      {nodes && nodes.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Node ID</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Hostname</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">GPU</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">RAM</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">CPU Cores</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Status</th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Last Heartbeat</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {nodes.map((node) => (
                <tr key={node.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3 font-mono text-xs text-gray-700 max-w-[160px] truncate" title={node.node_id}>
                    {node.node_id}
                  </td>
                  <td className="px-4 py-3 text-gray-700">{node.hostname}</td>
                  <td className="px-4 py-3 text-gray-600">
                    {node.gpu_model ? (
                      <span title={node.vram_mb ? `${formatRam(node.vram_mb)} VRAM` : undefined}>
                        {node.gpu_model}
                        {node.vram_mb && (
                          <span className="ml-1 text-xs text-gray-400">
                            ({formatRam(node.vram_mb)})
                          </span>
                        )}
                      </span>
                    ) : (
                      <span className="text-gray-400">None</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-600">{formatRam(node.ram_mb)}</td>
                  <td className="px-4 py-3 text-gray-600">{node.cpu_cores}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={node.status} />
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
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
