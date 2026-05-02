// API client utilities for the ML Trainer Dashboard

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const API_URL = API_BASE_URL;

/**
 * Generic fetcher for SWR — throws on non-OK responses so SWR surfaces errors.
 */
export async function fetcher<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: res.statusText }));
    const err = new Error(
      error?.detail
        ? typeof error.detail === "string"
          ? error.detail
          : JSON.stringify(error.detail)
        : error?.error || `HTTP ${res.status}`
    );
    (err as Error & { status: number }).status = res.status;
    throw err;
  }
  return res.json() as Promise<T>;
}

/**
 * POST helper for form submissions.
 */
export async function postJSON<TBody, TResponse>(
  path: string,
  body: TBody
): Promise<TResponse> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(
      data?.detail
        ? typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail)
        : data?.error || `HTTP ${res.status}`
    );
    (err as Error & { status: number; data: unknown }).status = res.status;
    (err as Error & { status: number; data: unknown }).data = data;
    throw err;
  }
  return data as TResponse;
}

/** Format a timestamp string for display */
export function formatDate(ts: string | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts).toLocaleString();
}

/** Format bytes to human-readable */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Format RAM in MB to human-readable */
export function formatRam(mb: number | null | undefined): string {
  if (mb == null) return "—";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}
