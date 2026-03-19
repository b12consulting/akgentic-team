"""Tests for MongoEventStore -- MongoDB-backed EventStore implementation.

Validates all seven EventStore protocol methods including round-trip
serialization of polymorphic Message and BaseState fields.

Acceptance Criteria: AC1-AC13 from Story 5.1.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from unittest.mock import patch

from akgentic.core.messages.message import UserMessage

from akgentic.team.models import TeamStatus
from akgentic.team.repositories.mongo import MongoEventStore

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

from tests.models.conftest import (
    SampleAgentState,
    make_agent_state_snapshot,
    make_persisted_event,
    make_process,
)


class TestMongoEventStore:
    """Tests for MongoEventStore covering all EventStore protocol methods (AC1-AC13)."""

    # --- save_team / load_team (AC2, AC3) ---

    def test_save_and_load_team_round_trip(self, mongo_store: MongoEventStore) -> None:
        """AC2: save_team upserts into teams collection; load_team deserializes it back."""
        process = make_process()
        mongo_store.save_team(process)
        loaded = mongo_store.load_team(process.team_id)

        assert loaded is not None
        assert loaded.team_id == process.team_id
        assert loaded.status == process.status
        assert loaded.team_card.name == process.team_card.name
        assert loaded.created_at == process.created_at

    def test_load_team_returns_none_for_nonexistent(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC3: load_team returns None when no document exists."""
        result = mongo_store.load_team(uuid.uuid4())
        assert result is None

    def test_save_team_upserts_on_second_call(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC2: save_team upserts on subsequent calls for the same team_id."""
        process = make_process(status=TeamStatus.RUNNING)
        mongo_store.save_team(process)

        updated = make_process(team_id=process.team_id, status=TeamStatus.STOPPED)
        mongo_store.save_team(updated)

        loaded = mongo_store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.status == TeamStatus.STOPPED

    # --- save_event / load_events (AC4, AC5) ---

    def test_save_and_load_events_round_trip(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC4, AC5: save_event inserts; load_events returns ordered by sequence."""
        team_id = uuid.uuid4()
        events = [
            make_persisted_event(team_id=team_id, sequence=3),
            make_persisted_event(team_id=team_id, sequence=1),
            make_persisted_event(team_id=team_id, sequence=2),
        ]
        for event in events:
            mongo_store.save_event(event)

        loaded = mongo_store.load_events(team_id)
        assert len(loaded) == 3
        assert [e.sequence for e in loaded] == [1, 2, 3]

    def test_load_events_returns_empty_for_nonexistent(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC5: load_events returns [] for a team with no events."""
        result = mongo_store.load_events(uuid.uuid4())
        assert result == []

    # --- save_agent_state / load_agent_states (AC6, AC7) ---

    def test_save_and_load_agent_states_round_trip(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC6, AC7: save_agent_state writes; load_agent_states reads all."""
        team_id = uuid.uuid4()
        snap1 = make_agent_state_snapshot(team_id=team_id, agent_id="agent-a")
        snap2 = make_agent_state_snapshot(team_id=team_id, agent_id="agent-b")
        mongo_store.save_agent_state(snap1)
        mongo_store.save_agent_state(snap2)

        loaded = mongo_store.load_agent_states(team_id)
        assert len(loaded) == 2
        agent_ids = {s.agent_id for s in loaded}
        assert agent_ids == {"agent-a", "agent-b"}

    def test_save_agent_state_upserts_for_same_agent(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC6: save_agent_state upserts for the same agent_id."""
        team_id = uuid.uuid4()
        snap1 = make_agent_state_snapshot(
            team_id=team_id, agent_id="agent-a", state=SampleAgentState(task_count=1)
        )
        mongo_store.save_agent_state(snap1)

        snap2 = make_agent_state_snapshot(
            team_id=team_id, agent_id="agent-a", state=SampleAgentState(task_count=99)
        )
        mongo_store.save_agent_state(snap2)

        loaded = mongo_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 99

    def test_load_agent_states_returns_empty_for_nonexistent(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC7: load_agent_states returns [] for a team with no agent states."""
        result = mongo_store.load_agent_states(uuid.uuid4())
        assert result == []

    # --- delete_team (AC8) ---

    def test_delete_team_removes_all_documents(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC8: delete_team removes documents from all three collections."""
        process = make_process()
        mongo_store.save_team(process)
        mongo_store.save_event(
            make_persisted_event(team_id=process.team_id, sequence=1)
        )
        mongo_store.save_agent_state(
            make_agent_state_snapshot(team_id=process.team_id, agent_id="a")
        )

        mongo_store.delete_team(process.team_id)

        assert mongo_store.load_team(process.team_id) is None
        assert mongo_store.load_events(process.team_id) == []
        assert mongo_store.load_agent_states(process.team_id) == []

    def test_delete_team_noop_for_nonexistent(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC8: delete_team is a no-op for non-existent team (no error)."""
        mongo_store.delete_team(uuid.uuid4())  # should not raise

    # --- Polymorphic round-trip (AC11) ---

    def test_polymorphic_event_round_trip(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC11: PersistedEvent with UserMessage survives MongoDB round-trip."""
        team_id = uuid.uuid4()
        msg = UserMessage(content="hello from polymorphic test")
        event = make_persisted_event(team_id=team_id, sequence=1, event=msg)
        mongo_store.save_event(event)

        loaded = mongo_store.load_events(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].event, UserMessage)
        assert loaded[0].event.content == "hello from polymorphic test"

    def test_polymorphic_agent_state_round_trip(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC11: AgentStateSnapshot with SampleAgentState survives MongoDB round-trip."""
        team_id = uuid.uuid4()
        state = SampleAgentState(task_count=5)
        snap = make_agent_state_snapshot(
            team_id=team_id, agent_id="poly-agent", state=state
        )
        mongo_store.save_agent_state(snap)

        loaded = mongo_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 5

    # --- Corrupted data resilience ---

    def test_load_team_returns_none_for_corrupted_document(
        self, mongo_store: MongoEventStore, mongo_db: object
    ) -> None:
        """Corrupted team document returns None instead of raising."""
        from mongomock import Database as MockDB

        db: MockDB = mongo_db  # type: ignore[assignment]
        db["teams"].insert_one({"team_id": "bad-uuid", "status": "INVALID", "_bogus": True})
        result = mongo_store.load_team(uuid.UUID("00000000-0000-0000-0000-000000000000"))
        # No document matches, so None
        assert result is None

        # Insert a document with a valid team_id but corrupted fields
        team_id = uuid.uuid4()
        db["teams"].insert_one({"team_id": str(team_id), "not_a_valid_field": 123})
        result = mongo_store.load_team(team_id)
        assert result is None

    def test_load_events_skips_corrupted_documents(
        self, mongo_store: MongoEventStore, mongo_db: object
    ) -> None:
        """Corrupted event documents are skipped; valid ones still loaded."""
        from mongomock import Database as MockDB

        db: MockDB = mongo_db  # type: ignore[assignment]
        team_id = uuid.uuid4()

        # Insert a valid event
        valid_event = make_persisted_event(team_id=team_id, sequence=1)
        mongo_store.save_event(valid_event)

        # Insert a corrupted event document directly
        db["events"].insert_one(
            {"team_id": str(team_id), "sequence": 2, "corrupted": True}
        )

        loaded = mongo_store.load_events(team_id)
        assert len(loaded) == 1
        assert loaded[0].sequence == 1

    def test_load_agent_states_skips_corrupted_documents(
        self, mongo_store: MongoEventStore, mongo_db: object
    ) -> None:
        """Corrupted agent state documents are skipped; valid ones still loaded."""
        from mongomock import Database as MockDB

        db: MockDB = mongo_db  # type: ignore[assignment]
        team_id = uuid.uuid4()

        # Save a valid snapshot
        snap = make_agent_state_snapshot(team_id=team_id, agent_id="good-agent")
        mongo_store.save_agent_state(snap)

        # Insert a corrupted agent state document directly
        db["agent_states"].insert_one(
            {"team_id": str(team_id), "agent_id": "bad-agent", "corrupted": True}
        )

        loaded = mongo_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert loaded[0].agent_id == "good-agent"

    # --- Protocol compliance (AC9) ---

    def test_satisfies_event_store_protocol(
        self, mongo_store: MongoEventStore
    ) -> None:
        """AC9: MongoEventStore satisfies EventStore Protocol via structural subtyping."""
        store: EventStore = mongo_store
        assert store is not None

    # --- Import guard (AC10) ---

    def test_import_succeeds_when_pymongo_installed(self) -> None:
        """AC10: MongoEventStore is importable when pymongo is available."""
        from akgentic.team.repositories.mongo import MongoEventStore as Imported

        assert Imported is MongoEventStore

    def test_import_guard_when_pymongo_missing(self) -> None:
        """AC10: Conditional import fails gracefully when pymongo is unavailable."""
        import importlib
        import sys

        import akgentic.team.repositories as repos_module

        # Save and remove cached mongo and pymongo modules
        mongo_module_key = "akgentic.team.repositories.mongo"
        saved: dict[str, object] = {}
        for key in list(sys.modules):
            if key.startswith(mongo_module_key):
                saved[key] = sys.modules.pop(key)
        pymongo_saved: dict[str, object] = {}
        for key in list(sys.modules):
            if key.startswith("pymongo"):
                pymongo_saved[key] = sys.modules.pop(key)

        try:
            # Simulate pymongo being unavailable
            with patch.dict(
                sys.modules,
                {"pymongo": None, "pymongo.database": None, "pymongo.collection": None},
            ):
                # Verify repositories __init__ removes MongoEventStore from __all__
                importlib.reload(repos_module)
                assert "MongoEventStore" not in repos_module.__all__

                # Verify direct import of mongo module raises helpful ImportError
                try:
                    importlib.import_module(mongo_module_key)
                    msg = "Expected ImportError was not raised"
                    raise AssertionError(msg)
                except ImportError as exc:
                    assert "pymongo is required" in str(exc)
                    assert "akgentic-team[mongo]" in str(exc)
        finally:
            # Restore all saved modules
            sys.modules.update(saved)  # type: ignore[arg-type]
            sys.modules.update(pymongo_saved)  # type: ignore[arg-type]
            importlib.reload(repos_module)
