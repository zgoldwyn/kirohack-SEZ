/**
 * Nodes list page (/nodes)
 * Displays all registered nodes with status, hardware info, and last heartbeat.
 * Data is fetched from GET /api/nodes via SWR with 10-second refresh.
 * Requirements: 9.1, 9.2, 9.3
 */
export default function NodesPage() {
  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">Nodes</h1>
      <p className="text-gray-500">Node list coming soon.</p>
    </main>
  );
}
