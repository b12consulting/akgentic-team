"""The one check every backend's ``load_agent_state`` runs before any I/O.

``load_agent_state`` receives its ``agent_id`` from an HTTP query string. Only a
canonical UUID string (``str(uuid.UUID(...))``) is accepted; anything else is a
miss on every backend. That one definition replaces per-backend sanitising: no
separator, dot, NUL or over-long name survives it, so the YAML store can turn the
id into a file name and Postgres can bind it without a further guard.
"""

from __future__ import annotations

import uuid

__all__ = [
    "is_canonical_agent_uuid",
]


def is_canonical_agent_uuid(agent_id: str) -> bool:
    """Return whether *agent_id* is a UUID in exactly the form ``str(uuid)`` produces.

    The round-trip equality makes it strict: braces, ``urn:uuid:``, hyphen-free
    hex and upper-case forms all parse as a UUID but are rejected.
    """
    try:
        return str(uuid.UUID(agent_id)) == agent_id
    except ValueError:
        return False
