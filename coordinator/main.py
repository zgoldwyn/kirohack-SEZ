"""Group ML Trainer — Coordinator (FastAPI entry point)."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI

from coordinator.dashboard import router as dashboard_router

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable is not set")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY environment variable is not set")

app = FastAPI(
    title="Group ML Trainer — Coordinator",
    description="Distributed ML task orchestration platform coordinator service.",
    version="0.1.0",
)

# Dashboard read endpoints (unauthenticated, local/demo only)
app.include_router(dashboard_router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}
