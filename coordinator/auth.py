"""Token generation, hashing, and FastAPI authentication dependency."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from fastapi import Header, HTTPException, status

from coordinator import db


def generate_token() -> str:
    """Generate a cryptographically secure auth token."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of *token* for safe storage."""
    return hashlib.sha256(token.encode()).hexdigest()


async def get_current_node(
    authorization: str = Header(..., description="Worker auth token"),
) -> dict[str, Any]:
    """FastAPI dependency that authenticates a Worker request.

    Extracts the token from the ``Authorization`` header, hashes it, and
    looks up the corresponding node in the ``nodes`` table.

    Returns the full node record on success.
    Raises HTTP 401 if the token is missing, empty, or unrecognised.
    """
    token = authorization.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing auth token",
        )

    token_hash = hash_token(token)

    try:
        node = db.select_one(
            "nodes",
            filters={"auth_token_hash": token_hash},
        )
    except db.RecordNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth token",
        )

    return node
