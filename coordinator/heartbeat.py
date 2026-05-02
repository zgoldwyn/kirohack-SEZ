"""Background heartbeat staleness monitor.

Periodically scans all registered nodes and marks those whose
``last_heartbeat`` is older than the staleness threshold as "offline".
When a node goes offline, any tasks in "assigned" or "running" status
on that node are marked as "failed" with the error message
"node went offline".  After failing tasks, the monitor checks whether
the parent job should also be marked as "failed".

Requirements: 2.2, 6.4
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from coordinator import db
from coordinator.aggregator import check_job_failure
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
        were assigned to or running on that node, then check whether
        the parent jobs should be marked as failed.
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
                "Node %s (%s) marked offline — last heartbeat: %s",
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

        Returns the set of job IDs whose tasks were affected.
        """
        now = datetime.now(timezone.utc).isoformat()
        affected_job_ids: set[str] = set()

        tasks = db.select("tasks", filters={"node_id": node_db_id})

        for task in tasks:
            task_status = task.get("status")
            if task_status in (TaskStatus.ASSIGNED.value, TaskStatus.RUNNING.value):
                db.update(
                    "tasks",
                    {
                        "status": TaskStatus.FAILED.value,
                        "error_message": "node went offline",
                        "completed_at": now,
                    },
                    filters={"id": task["id"]},
                )
                logger.info(
                    "Task %s (job %s) marked failed — node %s went offline",
                    task["id"],
                    task.get("job_id"),
                    node_db_id,
                )
                affected_job_ids.add(task["job_id"])

        return affected_job_ids


# Module-level singleton used by the application lifespan
heartbeat_monitor = HeartbeatMonitor()
