/**
 * Base API configuration for the Dashboard.
 * All data is fetched from the Coordinator REST API.
 */

export const COORDINATOR_URL =
  process.env.NEXT_PUBLIC_COORDINATOR_URL ?? "http://localhost:8000";

/**
 * Build a full URL for a Coordinator API path.
 * @param path - API path, e.g. "/api/nodes"
 */
export function apiUrl(path: string): string {
  return `${COORDINATOR_URL}${path}`;
}
