"""``NagraEventStore`` stub — full implementation lands in story 17.2.

This module declares the class shape so the surrounding skeleton (schema
loader, ``init_db``, packaging, init-container script, test fixtures) is
verifiable end-to-end without committing to query helpers in 17.1. Every
:class:`~akgentic.team.ports.EventStore` method raises
``NotImplementedError`` with a pointer to story 17.2.

The class satisfies the ``EventStore`` protocol via structural subtyping
— it does NOT inherit from the Protocol explicitly (mirrors
``YamlEventStore`` and ``MongoEventStore``).
"""

from __future__ import annotations

import uuid

from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process

_NOT_IMPLEMENTED_MSG = "Implemented in story 17.2"


class NagraEventStore:
    """Nagra-backed ``EventStore`` stub.

    Constructor calls :func:`_ensure_schema_loaded` so instances are always
    safe to build, but does NOT call :func:`init_db` — operators must
    invoke ``python -m akgentic.team.scripts.init_db`` once per deployment.

    Args:
        conn_string: Nagra-compatible Postgres connection string used by
            the (forthcoming) query helpers.
    """

    def __init__(self, conn_string: str) -> None:
        # Local import keeps the constructor cheap and avoids a circular
        # import on package initialisation (``__init__`` re-exports this
        # class after defining ``_ensure_schema_loaded``).
        from akgentic.team.repositories.postgres import _ensure_schema_loaded

        _ensure_schema_loaded()
        self._conn_string = conn_string

    def save_event(self, event: PersistedEvent) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def load_events(self, team_id: uuid.UUID) -> list[PersistedEvent]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def save_team(self, process: Process) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def delete_team(self, team_id: uuid.UUID) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def list_teams(self) -> list[Process]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def get_max_sequence(self, team_id: uuid.UUID) -> int:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
