/**
 * Jobs list page (/jobs)
 * Displays all jobs with status, model type, dataset, shard count, and timestamps.
 * Data is fetched from GET /api/jobs via SWR with auto-refresh.
 * Requirements: 10.1
 */
export default function JobsPage() {
  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">Jobs</h1>
      <p className="text-gray-500">Job list coming soon.</p>
    </main>
  );
}
