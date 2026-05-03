"""Background heartbeat staleness monitor.

Periodically scans all registered nodes and marks those whose
``last_heartbeat`` is older than the staleness threshold as "offline".
When a node goes offline, any tasks in "assigned" or "running" status
on that node are marked as "failed" with the error message
"node went offline".  For tasks belonging to a running collaborative
training job, the monitor also removes the worker from the job's active
set (adjusting the synchronization barrier) and triggers aggregation if
the barrier is now met.  After failing tasks, the monitor checks whether
the parent job should also be marked as "failed".

Requirements: 2.2, 6.4, 6.5, 14.1, 14.2, 14.4
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from coordinator import db
from coordinator import barrier as barrier_mod
from coordinator.aggregator import aggregate_round, check_job_failure
from coordinator.constants import JobStatus, NodeStatus, TaskStatus

logger = logging.getLogger(__name__)

# How often the monitor runs (seconds)
SCAN_INTERVAL_SECONDS = 10

# A node is considered stale if its last heartbeat is older than this
STALENESS_THRESHOLD_SECONDS = 30


class HeartbeatMonitor:
    """Manages the background staleness-check loop."""

    def __init__(
        self,
        scan_interval: float = SCAN_INTERVAL_SECONDS,
        staleness_threshold: float = STALENESS_THRESHOLD_SECONDS,
    ) -> None:
        self._scan_interval = scan_interval
        self._staleness_threshold = staleness_threshold
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task[None] | None:
        """Start the background monitor loop.

        Returns the ``asyncio.Task`` so the caller can cancel it on
        shutdown.
        """
        if self._running:
            return self._task
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Heartbeat staleness monitor started")
        return self._task

    def stop(self) -> None:
        """Signal the monitor loop to stop."""
        self._running = False
        logger.info("Heartbeat staleness monitor stopping")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Run the staleness check on a fixed interval."""
        while self._running:
            try:
                self._check_stale_nodes()
            except Exception:
                logger.exception("Error during heartbeat staleness check")
            await asyncio.sleep(self._scan_interval)

    # ------------------------------------------------------------------
    # Core logic (synchronous — uses the sync db helpers)
    # ------------------------------------------------------------------

    def _check_stale_nodes(self) -> None:
        """Scan all nodes and mark stale ones as offline.

        For each node that transitions to offline, fail any tasks that
        were assigned to or running on that node.  For tasks in running
        collaborative training jobs, also remove the worker from the
        active set and trigger aggregation if the barrier is now met.
        Finally, check whether the parent jobs should be marked as
        failed.
        """
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(seconds=self._staleness_threshold)

        nodes = db.select("nodes")

        affected_job_ids: set[str] = set()

        for node in nodes:
            # Skip nodes that are already offline
            if node.get("status") == NodeStatus.OFFLINE.value:
                continue

            last_hb = node.get("last_heartbeat")
            if last_hb is None:
                # Node never sent a heartbeat — treat as stale
                pass
            else:
                # Parse the timestamp if it's a string
                if isinstance(last_hb, str):
                    last_hb = datetime.fromisoformat(last_hb)
                # Ensure timezone-aware comparison
                if last_hb.tzinfo is None:
                    last_hb = last_hb.replace(tzinfo=timezone.utc)
                if last_hb >= threshold:
                    # Node is still fresh
                    continue

            node_db_id = node["id"]

            # Mark node as offline
            db.update(
                "nodes",
                {"status": NodeStatus.OFFLINE.value},
                filters={"id": node_db_id},
            )
            logger.info(
                "event=node_offline | node_id=%s | node_db_id=%s | last_heartbeat=%s",
                node.get("node_id"),
                node_db_id,
                node.get("last_heartbeat"),
            )

            # Fail any tasks assigned to or running on this node
            job_ids = self._fail_tasks_for_node(node_db_id)
            affected_job_ids.update(job_ids)

        # For each affected job, check if it should be marked as failed
        for job_id in affected_job_ids:
            check_job_failure(job_id)

    def _fail_tasks_for_node(self, node_db_id: str) -> set[str]:
        """Mark tasks in 'assigned' or 'running' status for *node_db_id*
        as 'failed' with error ``"node went offline"``.

        For tasks belonging to a running collaborative training job,
        also removes the worker from the job's active set via
        ``barrier.remove_worker()`` and checks whether the
        synchronization barrier is now met (triggering aggregation if
        so).

        Returns the set of job IDs whose tasks were affected.
        """
        now = datetime.now(timezone.utc).isoformat()
        affected_job_ids: set[str] = set()

        tasks = db.select("tasks", filters={"node_id": node_db_id})

        for task in tasks:
            task_status = task.get("status")
            if task_status not in (TaskStatus.ASSIGNED.value, TaskStatus.RUNNING.value):
                continue

            task_id = task["id"]
            job_id = task["job_id"]

            # Mark the task as failed
            db.update(
                "tasks",
                {
                    "status": TaskStatus.FAILED.value,
                    "error_message": "node went offline",
                    "completed_at": now,
                },
                filters={"id": task_id},
            )
            logger.info(
                "event=task_failed_node_offline | task_id=%s | job_id=%s | node_db_id=%s",
                task_id,
                job_id,
                node_db_id,
            )

            affected_job_ids.add(job_id)

            # For running collaborative training jobs, update the
            # synchronization barrier and potentially trigger aggregation.
            self._handle_barrier_for_failed_task(job_id, task_id)

        return affected_job_ids

    def _handle_barrier_for_failed_task(
        self, job_id: str, task_id: str
    ) -> None:
        """Handle barrier adjustment when a task fails in a running job.

        1. Look up the job to check if it is in "running" status.
        2. Call ``barrier.remove_worker()`` to adjust the active worker
           count (the task is already marked failed, so remove_worker
           will skip re-failing it).
        3. Get the job's ``current_round`` and check if the barrier is
           now met.  If so, trigger ``aggregator.aggregate_round()``.

        Requirements: 6.5, 14.2, 14.4
        """
        job_rows = db.select("jobs", filters={"id": job_id})
        if not job_rows:
            return

        job = job_rows[0]
        if job.get("status") != JobStatus.RUNNING.value:
            return

        # Remove worker from the active set and adjust barrier count.
        # Since we already marked the task as failed above,
        # barrier.remove_worker will detect the task is already failed
        # and skip the status update, but will still decrement the
        # active_worker_count on the current training round.
        barrier_mod.remove_worker(job_id, task_id)

        # Check if the barrier is now met for the current round.
        # This handles the case where the failed worker was the last
        # one the barrier was waiting on (Req 14.4: if a worker fails
        # after other workers have already submitted, the barrier may
        # now be satisfied).
        current_round = job.get("current_round")
        if current_round is not None:
            if barrier_mod.check_barrier(job_id, current_round):
                logger.info(
                    "event=barrier_met_after_worker_removal | job_id=%s | round=%d",
                    job_id,
                    current_round,
                )
                try:
                    aggregate_round(job_id, current_round)
                except Exception:
                    logger.exception(
                        "event=aggregation_after_removal_failed | job_id=%s | round=%d",
                        job_id,
                        current_round,
                    )


# Module-level singleton used by the application lifespan
heartbeat_monitor = HeartbeatMonitor()
