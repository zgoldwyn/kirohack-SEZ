"""Local worker state persistence.

Persists Worker credentials (auth token, node database ID, and
Coordinator URL) to a local JSON file so that a restarted Worker can
skip re-registration and reuse its existing identity.

Default state file location: ``~/.group-ml-trainer/worker_state.json``

Lifecycle
---------
1. **After registration** — call :func:`save_state` to persist the
   credentials returned by the Coordinator.
2. **On startup** — call :func:`load_state` to check for an existing
   state file.  If present and valid, the Worker reuses the stored
   credentials instead of registering again.
3. **On auth rejection (401)** — call :func:`delete_state` to remove
   the stale state file.  The Worker should then exit for operator
   action.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location for the persisted state file
DEFAULT_STATE_DIR = Path.home() / ".group-ml-trainer"
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "worker_state.json"

# Keys expected in the state JSON
_REQUIRED_KEYS = {"auth_token", "node_db_id", "coordinator_url"}


@dataclass(frozen=True)
class WorkerState:
    """Immutable snapshot of persisted Worker credentials."""

    auth_token: str
    node_db_id: str
    coordinator_url: str


def save_state(
    auth_token: str,
    node_db_id: str,
    coordinator_url: str,
    *,
    state_file: Path = DEFAULT_STATE_FILE,
) -> None:
    """Persist Worker credentials to a local JSON file.

    Creates the parent directory if it does not exist.  The file is
    written atomically-ish by writing then flushing, which is
    sufficient for a single-process CLI tool.

    Parameters
    ----------
    auth_token:
        The auth token issued by the Coordinator at registration.
    node_db_id:
        The database UUID assigned to this node by the Coordinator.
    coordinator_url:
        The base URL of the Coordinator the Worker registered with.
    state_file:
        Path to the state file.  Defaults to
        ``~/.group-ml-trainer/worker_state.json``.
    """
    state_file = Path(state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "auth_token": auth_token,
        "node_db_id": node_db_id,
        "coordinator_url": coordinator_url,
    }

    state_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Worker state saved to %s", state_file)


def load_state(
    *,
    state_file: Path = DEFAULT_STATE_FILE,
) -> WorkerState | None:
    """Load persisted Worker credentials from disk.

    Returns ``None`` (instead of raising) when the file is missing,
    unreadable, or does not contain the required keys — the caller
    should fall through to fresh registration.

    Parameters
    ----------
    state_file:
        Path to the state file.  Defaults to
        ``~/.group-ml-trainer/worker_state.json``.

    Returns
    -------
    WorkerState | None
        The restored credentials, or ``None`` if no valid state exists.
    """
    state_file = Path(state_file)

    if not state_file.is_file():
        logger.debug("No state file found at %s", state_file)
        return None

    try:
        raw = state_file.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read state file %s: %s", state_file, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("State file %s does not contain a JSON object", state_file)
        return None

    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        logger.warning(
            "State file %s is missing required keys: %s", state_file, missing
        )
        return None

    # Validate that all values are non-empty strings
    for key in _REQUIRED_KEYS:
        if not isinstance(data[key], str) or not data[key].strip():
            logger.warning(
                "State file %s has invalid value for '%s'", state_file, key
            )
            return None

    logger.info("Loaded worker state from %s (node_db_id=%s)", state_file, data["node_db_id"])
    return WorkerState(
        auth_token=data["auth_token"],
        node_db_id=data["node_db_id"],
        coordinator_url=data["coordinator_url"],
    )


def delete_state(
    *,
    state_file: Path = DEFAULT_STATE_FILE,
) -> None:
    """Delete the persisted state file.

    Called when the Coordinator rejects the stored auth token (HTTP 401).
    After deletion the Worker should exit so the operator can
    re-register manually.

    Silently succeeds if the file does not exist.

    Parameters
    ----------
    state_file:
        Path to the state file.  Defaults to
        ``~/.group-ml-trainer/worker_state.json``.
    """
    state_file = Path(state_file)

    try:
        state_file.unlink(missing_ok=True)
        logger.info("Deleted worker state file %s", state_file)
    except OSError as exc:
        logger.warning("Failed to delete state file %s: %s", state_file, exc)
