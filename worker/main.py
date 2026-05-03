"""Group ML Trainer — Worker entry point.

Standalone Python agent that registers with the Coordinator,
maintains a heartbeat, polls for training tasks, executes
PyTorch training runs, and reports results back.

Lifecycle
---------
1. **Startup** — Check for an existing state file.  If valid, reuse
   stored credentials; otherwise register with the Coordinator.
2. **Heartbeat loop** — Background thread sends a heartbeat every 10 s.
3. **Poll loop** — Main loop polls for tasks every 5 s.  When a task is
   received: parse config → train → upload checkpoint → complete.
4. **Auth failure (401)** — Delete the state file, stop all loops, log
   the error, and exit with a message for the operator.
5. **Shutdown** — Clean up HTTP clients on exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from worker.config import parse_task_config
from worker.reporter import AuthenticationError, Reporter, TransientError
from worker.state import WorkerState, delete_state, load_state, save_state
from worker.trainer import run_task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8000")
HEARTBEAT_INTERVAL = float(os.getenv("WORKER_HEARTBEAT_INTERVAL", "10"))
POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "5"))

# ---------------------------------------------------------------------------
# Hardware detection helpers
# ---------------------------------------------------------------------------


def _detect_hardware() -> dict[str, object]:
    """Gather hardware and environment info for registration."""
    info: dict[str, object] = {
        "hostname": socket.gethostname(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_mb": _detect_ram_mb(),
        "disk_mb": _detect_disk_mb(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }

    # PyTorch version
    try:
        import torch
        info["pytorch_version"] = torch.__version__
    except ImportError:
        info["pytorch_version"] = "unknown"

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_model"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem
            info["vram_mb"] = vram // (1024 * 1024)
    except Exception:
        pass  # No GPU info — fields remain absent

    return info


def _detect_ram_mb() -> int:
    """Return total system RAM in MB, with a safe fallback."""
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        pass

    # macOS / Linux fallback via os.sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        pass

    # Windows fallback via ctypes
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return mem.ullTotalPhys // (1024 * 1024)
        except Exception:
            pass

    return 4096  # Conservative default


def _detect_disk_mb() -> int:
    """Return available disk space in MB for the current directory."""
    try:
        stat = os.statvfs(".")
        return (stat.f_bavail * stat.f_frsize) // (1024 * 1024)
    except (OSError, AttributeError):
        pass

    # Windows fallback via shutil
    try:
        import shutil
        usage = shutil.disk_usage(".")
        return usage.free // (1024 * 1024)
    except Exception:
        pass

    return 10240  # Conservative default


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class Worker:
    """Orchestrates the Worker lifecycle: register, heartbeat, poll, train.

    Parameters
    ----------
    coordinator_url:
        Base URL of the Coordinator.
    node_id:
        Unique identifier for this worker node.  Defaults to a
        hostname-based ID.
    """

    def __init__(
        self,
        coordinator_url: str = COORDINATOR_URL,
        node_id: str | None = None,
    ) -> None:
        self.coordinator_url = coordinator_url
        self.node_id = node_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.reporter = Reporter(coordinator_url=coordinator_url)

        # Lifecycle flags
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Startup — registration or credential reuse
    # ------------------------------------------------------------------

    async def _ensure_registered(self) -> None:
        """Load existing state or register with the Coordinator."""
        state = load_state()

        if state is not None and state.coordinator_url == self.coordinator_url:
            logger.info(
                "Reusing stored credentials (node_db_id=%s)",
                state.node_db_id,
            )
            self.reporter.set_auth_token(state.auth_token)
            return

        if state is not None:
            logger.warning(
                "Stored state targets a different Coordinator (%s vs %s). "
                "Re-registering.",
                state.coordinator_url,
                self.coordinator_url,
            )

        logger.info("Registering with Coordinator at %s", self.coordinator_url)
        hw = _detect_hardware()

        result = await self.reporter.register(
            node_id=self.node_id,
            hostname=str(hw["hostname"]),
            cpu_cores=int(hw["cpu_cores"]),  # type: ignore[arg-type]
            ram_mb=int(hw["ram_mb"]),  # type: ignore[arg-type]
            disk_mb=int(hw["disk_mb"]),  # type: ignore[arg-type]
            os=str(hw["os"]),
            python_version=str(hw["python_version"]),
            pytorch_version=str(hw.get("pytorch_version", "unknown")),
            gpu_model=hw.get("gpu_model"),  # type: ignore[arg-type]
            vram_mb=hw.get("vram_mb"),  # type: ignore[arg-type]
        )

        save_state(
            auth_token=result.auth_token,
            node_db_id=result.node_db_id,
            coordinator_url=self.coordinator_url,
        )
        logger.info(
            "Registered successfully (node_db_id=%s)", result.node_db_id
        )

    # ------------------------------------------------------------------
    # Heartbeat loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats at a fixed interval until stopped."""
        while self._running:
            try:
                await self.reporter.heartbeat()
                logger.debug("Heartbeat sent")
            except AuthenticationError:
                logger.error("Heartbeat rejected (401). Stopping worker.")
                self._handle_auth_failure()
                return
            except (TransientError, Exception) as exc:
                logger.warning("Heartbeat failed: %s", exc)

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Poll for tasks and execute them."""
        while self._running:
            try:
                task_data = await self.reporter.poll_task()

                if task_data is not None:
                    await self._execute_task(task_data)
                else:
                    logger.debug("No task available, will poll again")

            except AuthenticationError:
                logger.error("Poll rejected (401). Stopping worker.")
                self._handle_auth_failure()
                return
            except (TransientError, Exception) as exc:
                logger.warning("Poll failed: %s", exc)

            await asyncio.sleep(POLL_INTERVAL)

    async def _execute_task(self, task_data: dict) -> None:
        """Parse task config and run the training task.

        After the task completes (success or failure), control returns
        to the poll loop.
        """
        try:
            task_config = parse_task_config(task_data)
            logger.info(
                "Received task %s (job=%s, shard=%d/%d, dataset=%s, model=%s)",
                task_config.task_id,
                task_config.job_id,
                task_config.shard_index,
                task_config.shard_count,
                task_config.dataset_name,
                task_config.model_type,
            )

            await run_task(
                task_config=task_config,
                reporter=self.reporter,
            )

        except AuthenticationError:
            # Re-raise so the poll loop can handle it
            raise
        except Exception as exc:
            # run_task already handles its own error reporting to the
            # Coordinator, so we just log here and return to polling.
            logger.error("Task execution error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Auth failure handling
    # ------------------------------------------------------------------

    def _handle_auth_failure(self) -> None:
        """Handle a 401 auth rejection: delete state, stop loops."""
        self._running = False
        delete_state()
        logger.error(
            "Auth token rejected by Coordinator. "
            "State file deleted. "
            "Please re-register this worker."
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the Worker: register, then run heartbeat + poll loops."""
        self._running = True

        try:
            await self._ensure_registered()
        except AuthenticationError:
            self._handle_auth_failure()
            return
        except Exception as exc:
            logger.error("Registration failed: %s", exc, exc_info=True)
            return

        logger.info(
            "Worker started (node_id=%s, coordinator=%s)",
            self.node_id,
            self.coordinator_url,
        )

        # Start heartbeat as a background task
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="heartbeat",
        )

        try:
            # Run the poll loop in the foreground
            await self._poll_loop()
        finally:
            # Ensure heartbeat task is cleaned up
            self._running = False
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            await self._cleanup()

    async def _cleanup(self) -> None:
        """Close HTTP clients."""
        await self.reporter.close()
        logger.info("Worker shut down cleanly")

    # ------------------------------------------------------------------
    # Graceful shutdown via signal
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the worker to stop after the current iteration."""
        logger.info("Stop requested")
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure structured logging for the Worker."""
    log_level = os.getenv("WORKER_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


async def _async_main() -> None:
    """Async entry point that sets up signal handlers and runs the Worker."""
    worker = Worker()

    # Register signal handlers for graceful shutdown
    # add_signal_handler is not supported on Windows; fall back to
    # KeyboardInterrupt handling in main().
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, worker.stop)

    await worker.run()


def main() -> None:
    """Synchronous CLI entry point."""
    _setup_logging()
    logger.info("Starting Group ML Trainer Worker")

    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as exc:
        logger.error("Worker exited with error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
