"""HTTP client for Coordinator communication.

Provides typed async functions for every Coordinator endpoint a Worker
needs: registration, heartbeat, task polling, task lifecycle, gradient
exchange, and parameter download.

Key behaviours
--------------
* **Auth token injection** – After registration the token is stored and
  automatically included in the ``Authorization`` header of every
  subsequent request.
* **401 handling** – Any ``HTTP 401`` response raises
  :class:`AuthenticationError` so the caller (``main.py``) can stop all
  loops and surface the error for operator action.
* **Retry with exponential back-off** – Transient failures (HTTP 503,
  network errors) are retried up to ``max_retries`` times with
  exponential back-off (base × 2^attempt, capped at ``max_backoff``
  seconds).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AuthenticationError(Exception):
    """Raised when the Coordinator rejects the auth token (HTTP 401).

    The Worker MUST stop all loops and surface this for operator action.
    """


class TransientError(Exception):
    """Raised when retries are exhausted for a transient failure."""


# ---------------------------------------------------------------------------
# Reporter configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BASE_BACKOFF = 1.0  # seconds
_DEFAULT_MAX_BACKOFF = 30.0  # seconds
_DEFAULT_TIMEOUT = 30.0  # seconds per request


@dataclass
class ReporterConfig:
    """Tuneable knobs for the HTTP reporter."""

    max_retries: int = _DEFAULT_MAX_RETRIES
    base_backoff: float = _DEFAULT_BASE_BACKOFF
    max_backoff: float = _DEFAULT_MAX_BACKOFF
    timeout: float = _DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Registration response
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Values returned by a successful ``POST /api/nodes/register``."""

    node_db_id: str
    auth_token: str


# ---------------------------------------------------------------------------
# Parameter download result
# ---------------------------------------------------------------------------


@dataclass
class ParameterDownloadResult:
    """Values returned by ``GET /api/jobs/{id}/parameters``."""

    param_bytes: bytes
    current_round: int
    job_status: str


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


@dataclass
class Reporter:
    """Async HTTP client that talks to the Coordinator on behalf of a Worker.

    Parameters
    ----------
    coordinator_url:
        Base URL of the Coordinator (e.g. ``http://localhost:8000``).
    config:
        Optional :class:`ReporterConfig` for retry / timeout tuning.
    """

    coordinator_url: str
    config: ReporterConfig = field(default_factory=ReporterConfig)

    # Set after successful registration
    _auth_token: str | None = field(default=None, init=False, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.coordinator_url,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def set_auth_token(self, token: str) -> None:
        """Manually set the auth token (e.g. restored from persisted state)."""
        self._auth_token = token

    def _auth_headers(self) -> dict[str, str]:
        if self._auth_token is None:
            return {}
        return {"Authorization": f"Bearer {self._auth_token}"}

    # ------------------------------------------------------------------
    # Low-level request with retry + 401 handling
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> httpx.Response:
        """Send an HTTP request with retry logic and auth handling.

        Parameters
        ----------
        method:
            HTTP method (GET, POST, etc.).
        path:
            URL path relative to the Coordinator base URL.
        json:
            Optional JSON body (mutually exclusive with *content*).
        content:
            Optional raw binary body (mutually exclusive with *json*).
        headers:
            Optional extra headers merged with auth headers.
        params:
            Optional query parameters.
        authenticated:
            Whether to include the auth token header.

        Raises
        ------
        AuthenticationError
            On HTTP 401 — caller must stop all loops.
        TransientError
            When retries are exhausted for 503 / network errors.
        httpx.HTTPStatusError
            For other non-2xx responses after the first attempt.
        """
        client = await self._get_client()
        req_headers = self._auth_headers() if authenticated else {}
        if headers:
            req_headers.update(headers)

        last_exc: BaseException | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.request(
                    method,
                    path,
                    json=json,
                    content=content,
                    headers=req_headers,
                    params=params,
                )

                # --- 401: auth failure — never retry -----------------
                if response.status_code == 401:
                    detail = ""
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    raise AuthenticationError(
                        f"Coordinator rejected auth token: {detail}"
                    )

                # --- 503: transient — retry --------------------------
                if response.status_code == 503:
                    logger.warning(
                        "Coordinator returned 503 (attempt %d/%d) for %s %s",
                        attempt + 1,
                        self.config.max_retries + 1,
                        method,
                        path,
                    )
                    last_exc = httpx.HTTPStatusError(
                        "503 Service Unavailable",
                        request=response.request,
                        response=response,
                    )
                    await self._backoff(attempt)
                    continue

                # --- Other errors: raise immediately -----------------
                response.raise_for_status()
                return response

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                logger.warning(
                    "Network error (attempt %d/%d) for %s %s: %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    method,
                    path,
                    exc,
                )
                last_exc = exc
                await self._backoff(attempt)
                continue

        raise TransientError(
            f"Exhausted {self.config.max_retries + 1} attempts for {method} {path}"
        ) from last_exc

    async def _backoff(self, attempt: int) -> None:
        delay = min(
            self.config.base_backoff * (2 ** attempt),
            self.config.max_backoff,
        )
        logger.debug("Backing off %.1fs before retry", delay)
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Public API — one method per Coordinator endpoint
    # ------------------------------------------------------------------

    async def register(
        self,
        *,
        node_id: str,
        hostname: str,
        cpu_cores: int,
        ram_mb: int,
        disk_mb: int,
        os: str,
        python_version: str,
        pytorch_version: str,
        gpu_model: str | None = None,
        vram_mb: int | None = None,
    ) -> RegistrationResult:
        """Register this Worker with the Coordinator.

        ``POST /api/nodes/register`` (unauthenticated)

        On success the auth token is stored internally and used for all
        subsequent requests.
        """
        payload: dict[str, Any] = {
            "node_id": node_id,
            "hostname": hostname,
            "cpu_cores": cpu_cores,
            "ram_mb": ram_mb,
            "disk_mb": disk_mb,
            "os": os,
            "python_version": python_version,
            "pytorch_version": pytorch_version,
        }
        if gpu_model is not None:
            payload["gpu_model"] = gpu_model
        if vram_mb is not None:
            payload["vram_mb"] = vram_mb

        resp = await self._request(
            "POST",
            "/api/nodes/register",
            json=payload,
            authenticated=False,
        )
        data = resp.json()
        self._auth_token = data["auth_token"]
        return RegistrationResult(
            node_db_id=data["node_db_id"],
            auth_token=data["auth_token"],
        )

    async def heartbeat(self) -> None:
        """Send a heartbeat to the Coordinator.

        ``POST /api/nodes/heartbeat`` (authenticated)
        """
        await self._request("POST", "/api/nodes/heartbeat")

    async def poll_task(self) -> dict[str, Any] | None:
        """Poll the Coordinator for an available task.

        ``GET /api/tasks/poll`` (authenticated)

        Returns the task config dict if a task was assigned, or ``None``
        if no work is available (the response has a null ``task_id``).
        """
        resp = await self._request("GET", "/api/tasks/poll")
        data = resp.json()
        if data.get("task_id") is None:
            return None
        return data

    async def start_task(self, task_id: str) -> None:
        """Notify the Coordinator that task execution has started.

        ``POST /api/tasks/{id}/start`` (authenticated)
        """
        await self._request("POST", f"/api/tasks/{task_id}/start")

    async def fail_task(self, task_id: str, *, error_message: str) -> None:
        """Report task failure.

        ``POST /api/tasks/{id}/fail`` (authenticated)
        """
        await self._request(
            "POST",
            f"/api/tasks/{task_id}/fail",
            json={"error_message": error_message},
        )

    # ------------------------------------------------------------------
    # Gradient exchange — collaborative training protocol
    # ------------------------------------------------------------------

    async def download_parameters(self, job_id: str) -> ParameterDownloadResult:
        """Download the current Global_Model parameters for a job.

        ``GET /api/jobs/{id}/parameters`` (authenticated)

        The Coordinator returns the model parameters as a binary payload
        (``application/octet-stream``) with metadata in custom response
        headers (``X-Current-Round``, ``X-Job-Status``).

        Returns a :class:`ParameterDownloadResult` containing the raw
        parameter bytes and round/status metadata.  When the job is
        completed or failed the ``param_bytes`` will be empty and the
        Worker should exit the training loop.
        """
        resp = await self._request("GET", f"/api/jobs/{job_id}/parameters")

        current_round = int(resp.headers.get("X-Current-Round", "0"))
        job_status = resp.headers.get("X-Job-Status", "running")
        param_bytes = resp.content

        return ParameterDownloadResult(
            param_bytes=param_bytes,
            current_round=current_round,
            job_status=job_status,
        )

    async def submit_gradients(
        self,
        job_id: str,
        *,
        round_number: int,
        task_id: str,
        gradient_bytes: bytes,
        local_loss: float | None = None,
        local_accuracy: float | None = None,
    ) -> None:
        """Submit a gradient update for the current training round.

        ``POST /api/jobs/{id}/gradients`` (authenticated)

        The gradient payload is sent as a raw binary body
        (``application/octet-stream``).  Metadata (round number, task
        ID, local metrics) is passed via query parameters as expected by
        the Coordinator endpoint.
        """
        query_params: dict[str, Any] = {
            "round_number": round_number,
            "task_id": task_id,
        }
        if local_loss is not None:
            query_params["local_loss"] = local_loss
        if local_accuracy is not None:
            query_params["local_accuracy"] = local_accuracy

        await self._request(
            "POST",
            f"/api/jobs/{job_id}/gradients",
            content=gradient_bytes,
            headers={"Content-Type": "application/octet-stream"},
            params=query_params,
        )
