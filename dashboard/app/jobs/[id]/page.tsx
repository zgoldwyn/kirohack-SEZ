/**
 * Job detail page (/jobs/[id])
 * Displays per-task progress, aggregated metrics, and artifact download links.
 * Data is fetched from GET /api/jobs/{id} and GET /api/jobs/{id}/artifacts via SWR.
 * Requirements: 10.2, 10.3, 10.4, 6.3, 7.3
 */
interface JobDetailPageProps {
  params: Promise<{ id: string }>;
}

export default async function JobDetailPage({ params }: JobDetailPageProps) {
  const { id } = await params;

  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">Job Detail</h1>
      <p className="text-gray-500">Job ID: {id}</p>
      <p className="text-gray-500">Job detail coming soon.</p>
    </main>
  );
}
