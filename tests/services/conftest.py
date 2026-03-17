"""Test fixtures for team service tests."""

from __future__ import annotations

import uuid

from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process


class InMemoryEventStore:
    """Dict-backed EventStore for test isolation.

    Satisfies the EventStore protocol via structural subtyping.
    """

    def __init__(self) -> None:
        self.events: list[PersistedEvent] = []
        self.teams: dict[uuid.UUID, Process] = {}
        self.agent_states: dict[tuple[uuid.UUID, str], AgentStateSnapshot] = {}

    def save_event(self, event: PersistedEvent) -> None:
        """Persist a single domain event."""
        self.events.append(event)

    def load_events(self, team_id: uuid.UUID) -> list[PersistedEvent]:
        """Load all persisted events for a team."""
        return [e for e in self.events if e.team_id == team_id]

    def save_team(self, process: Process) -> None:
        """Persist team process snapshot."""
        self.teams[process.team_id] = process

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Load a team process snapshot by ID."""
        return self.teams.get(team_id)

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete all persisted data for a team."""
        self.teams.pop(team_id, None)
        self.events = [e for e in self.events if e.team_id != team_id]
        self.agent_states = {
            k: v for k, v in self.agent_states.items() if k[0] != team_id
        }

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        """Persist an agent state snapshot."""
        self.agent_states[(snapshot.team_id, snapshot.agent_id)] = snapshot

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Load all agent state snapshots for a team."""
        return [v for k, v in self.agent_states.items() if k[0] == team_id]
