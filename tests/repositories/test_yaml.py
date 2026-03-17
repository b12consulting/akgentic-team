"""Tests for YamlEventStore — file-based EventStore implementation.

Validates all seven EventStore protocol methods including round-trip
serialization of polymorphic Message and BaseState fields.

Acceptance Criteria: AC1-AC12 from Story 2.3.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from akgentic.core.messages.message import UserMessage
from akgentic.team.models import TeamStatus
from akgentic.team.repositories.yaml import YamlEventStore

from tests.models.conftest import (
    SampleAgentState,
    make_agent_state_snapshot,
    make_persisted_event,
    make_process,
)


@pytest.fixture
def yaml_store(tmp_path: Path) -> YamlEventStore:
    """Create a YamlEventStore backed by a temporary directory."""
    return YamlEventStore(tmp_path)


class TestYamlEventStore:
    """Tests for YamlEventStore covering all EventStore protocol methods (AC1-AC12)."""

    # --- save_team / load_team ---

    def test_save_and_load_team_round_trip(self, yaml_store: YamlEventStore) -> None:
        """AC2, AC3: save_team writes team.yaml; load_team deserializes it back."""
        process = make_process()
        yaml_store.save_team(process)
        loaded = yaml_store.load_team(process.team_id)

        assert loaded is not None
        assert loaded.team_id == process.team_id
        assert loaded.status == process.status
        assert loaded.team_card.name == process.team_card.name
        assert loaded.created_at == process.created_at

    def test_load_team_returns_none_for_nonexistent(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC3: load_team returns None when no team.yaml exists."""
        result = yaml_store.load_team(uuid.uuid4())
        assert result is None

    def test_save_team_overwrites_on_second_call(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC2: save_team overwrites existing team.yaml on subsequent calls."""
        process = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(process)

        updated = make_process(team_id=process.team_id, status=TeamStatus.STOPPED)
        yaml_store.save_team(updated)

        loaded = yaml_store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.status == TeamStatus.STOPPED

    # --- save_event / load_events ---

    def test_save_and_load_events_round_trip(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC4, AC5: save_event appends; load_events returns ordered by sequence."""
        team_id = uuid.uuid4()
        events = [
            make_persisted_event(team_id=team_id, sequence=3),
            make_persisted_event(team_id=team_id, sequence=1),
            make_persisted_event(team_id=team_id, sequence=2),
        ]
        for event in events:
            yaml_store.save_event(event)

        loaded = yaml_store.load_events(team_id)
        assert len(loaded) == 3
        assert [e.sequence for e in loaded] == [1, 2, 3]

    def test_save_event_append_only(self, yaml_store: YamlEventStore) -> None:
        """AC4: save_event appends without overwriting previous events."""
        team_id = uuid.uuid4()
        yaml_store.save_event(make_persisted_event(team_id=team_id, sequence=1))
        yaml_store.save_event(make_persisted_event(team_id=team_id, sequence=2))

        loaded = yaml_store.load_events(team_id)
        assert len(loaded) == 2

    def test_load_events_returns_empty_for_nonexistent(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC5: load_events returns [] when no events.yaml exists."""
        result = yaml_store.load_events(uuid.uuid4())
        assert result == []

    # --- save_agent_state / load_agent_states ---

    def test_save_and_load_agent_states_round_trip(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC6, AC7: save_agent_state writes; load_agent_states reads all."""
        team_id = uuid.uuid4()
        snap1 = make_agent_state_snapshot(team_id=team_id, agent_id="agent-a")
        snap2 = make_agent_state_snapshot(team_id=team_id, agent_id="agent-b")
        yaml_store.save_agent_state(snap1)
        yaml_store.save_agent_state(snap2)

        loaded = yaml_store.load_agent_states(team_id)
        assert len(loaded) == 2
        agent_ids = {s.agent_id for s in loaded}
        assert agent_ids == {"agent-a", "agent-b"}

    def test_save_agent_state_overwrites_for_same_agent(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC6: save_agent_state overwrites for the same agent_id."""
        team_id = uuid.uuid4()
        snap1 = make_agent_state_snapshot(
            team_id=team_id, agent_id="agent-a", state=SampleAgentState(task_count=1)
        )
        yaml_store.save_agent_state(snap1)

        snap2 = make_agent_state_snapshot(
            team_id=team_id, agent_id="agent-a", state=SampleAgentState(task_count=99)
        )
        yaml_store.save_agent_state(snap2)

        loaded = yaml_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 99

    def test_load_agent_states_returns_empty_for_nonexistent(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC7: load_agent_states returns [] when no states/ dir exists."""
        result = yaml_store.load_agent_states(uuid.uuid4())
        assert result == []

    # --- delete_team ---

    def test_delete_team_removes_directory_tree(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC8: delete_team removes the entire team directory and all contents."""
        process = make_process()
        yaml_store.save_team(process)
        yaml_store.save_event(make_persisted_event(team_id=process.team_id, sequence=1))
        yaml_store.save_agent_state(
            make_agent_state_snapshot(team_id=process.team_id, agent_id="a")
        )

        yaml_store.delete_team(process.team_id)

        assert yaml_store.load_team(process.team_id) is None
        assert yaml_store.load_events(process.team_id) == []
        assert yaml_store.load_agent_states(process.team_id) == []

    def test_delete_team_noop_for_nonexistent(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC8: delete_team is a no-op for non-existent team (no error)."""
        yaml_store.delete_team(uuid.uuid4())  # should not raise

    # --- Polymorphic round-trip ---

    def test_polymorphic_event_round_trip(self, yaml_store: YamlEventStore) -> None:
        """AC11: PersistedEvent with UserMessage survives YAML round-trip."""
        team_id = uuid.uuid4()
        msg = UserMessage(content="hello from polymorphic test")
        event = make_persisted_event(team_id=team_id, sequence=1, event=msg)
        yaml_store.save_event(event)

        loaded = yaml_store.load_events(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].event, UserMessage)
        assert loaded[0].event.content == "hello from polymorphic test"

    def test_polymorphic_agent_state_round_trip(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC11: AgentStateSnapshot with SampleAgentState survives YAML round-trip."""
        team_id = uuid.uuid4()
        state = SampleAgentState(task_count=5)
        snap = make_agent_state_snapshot(team_id=team_id, agent_id="poly-agent", state=state)
        yaml_store.save_agent_state(snap)

        loaded = yaml_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 5

    # --- Directory creation ---

    def test_directory_creation_is_automatic(
        self, yaml_store: YamlEventStore
    ) -> None:
        """AC10: Directories are created on demand, not eagerly."""
        team_id = uuid.uuid4()
        # Save event without pre-creating any dirs
        event = make_persisted_event(team_id=team_id, sequence=1)
        yaml_store.save_event(event)  # should not raise

        loaded = yaml_store.load_events(team_id)
        assert len(loaded) == 1
