/**
 * System Overview page (/)
 * Displays a summary of online nodes and running jobs.
 * Data is fetched from GET /api/monitoring/summary via SWR.
 */
export default function HomePage() {
  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">Group ML Trainer — System Overview</h1>
      <p className="text-gray-500">
        Dashboard UI coming soon. See <a href="/nodes" className="underline">/nodes</a> and{" "}
        <a href="/jobs" className="underline">/jobs</a> for details.
      </p>
    </main>
  );
}
