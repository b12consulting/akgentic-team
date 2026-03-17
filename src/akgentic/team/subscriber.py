"""PersistenceSubscriber: EventSubscriber to EventStore bridge for live event sourcing."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from akgentic.core.messages.message import Message
from akgentic.core.messages.orchestrator import StateChangedMessage
from akgentic.core.orchestrator import EventSubscriber
from akgentic.team.models import AgentStateSnapshot, PersistedEvent
from akgentic.team.ports import EventStore


class PersistenceSubscriber(EventSubscriber):
    """Bridges EventSubscriber (akgentic-core) with EventStore (akgentic-team).

    Receives all messages from the orchestrator and persists:
    1. Every message as an append-only event (for replay on resume)
    2. Agent state snapshots on StateChangedMessage (for fast restore)

    LLM conversation history is NOT persisted separately -- it is
    reconstructed from the event log during restore.

    All types used here come from akgentic-core (Message, StateChangedMessage,
    BaseState) -- no imports from akgentic-llm or akgentic-agent needed.
    """

    def __init__(self, team_id: uuid.UUID, event_store: EventStore) -> None:
        """Initialize the persistence subscriber.

        Args:
            team_id: Unique identifier of the team whose events are persisted.
            event_store: Storage backend for events and agent state snapshots.
        """
        self._team_id = team_id
        self._event_store = event_store
        self._sequence = 0
        self._restoring = False

    def on_message(self, msg: Message) -> None:
        """Persist a message as a PersistedEvent and optionally save agent state.

        If the restoring flag is True, skips all persistence to avoid duplicate
        writes during event replay.

        Args:
            msg: The message flowing through the orchestrator.
        """
        if self._restoring:
            return

        self._sequence += 1
        event = PersistedEvent(
            team_id=self._team_id,
            sequence=self._sequence,
            event=msg,
            timestamp=datetime.now(UTC),
        )
        self._event_store.save_event(event)

        if isinstance(msg, StateChangedMessage):
            agent_id = msg.sender.name  # type: ignore[union-attr]
            snapshot = AgentStateSnapshot(
                team_id=self._team_id,
                agent_id=agent_id,
                state=msg.state.serializable_copy(),
                updated_at=datetime.now(UTC),
            )
            self._event_store.save_agent_state(snapshot)

    def set_restoring(self, restoring: bool) -> None:
        """Set the restoring flag to skip or resume persistence.

        Args:
            restoring: If True, on_message will skip all persistence.
        """
        self._restoring = restoring

    def on_stop(self) -> None:
        """No-op: required by EventSubscriber protocol."""
        pass
