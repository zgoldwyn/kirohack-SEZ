"""Group ML Trainer — Worker (entry point).

Standalone Python agent that registers with the Coordinator,
maintains a heartbeat, polls for training tasks, executes
PyTorch training runs, and reports results back.
"""

import os

from dotenv import load_dotenv

load_dotenv()

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8000")
