"""Authentication module for Worker token validation.

Provides SHA-256 hashing for token storage and a FastAPI dependency
that extracts the Bearer token from the Authorization header, hashes
it, looks up the corresponding node in the database, and returns the
node record or raises HTTP 401.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import Depends, HTTPException, Request, status

from coordinator import db

logger = logging.getLogger(__name__)


def hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of *token*."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def get_current_node(request: Request) -> dict:
    """FastAPI dependency: authenticate the request and return the node record.

    Extracts the token from the ``Authorization: Bearer <token>`` header,
    hashes it with SHA-256, and looks up the hash in the ``nodes`` table.

    Returns the full node row dict on success.

    Raises
    ------
    HTTPException (401)
        If the header is missing, malformed, or the token hash does not
        match any registered node.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
        )

    token_hash = hash_token(token)

    try:
        rows = db.select("nodes", filters={"auth_token_hash": token_hash})
    except db.DatabaseError as exc:
        logger.error("Database error during auth lookup: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth token",
        )

    return rows[0]
