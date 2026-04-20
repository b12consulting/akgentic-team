"""Tests for ``NagraEventStore`` — Postgres-backed ``EventStore``.

Validates all nine ``EventStore`` Protocol methods including:

* per-method ``Transaction`` ownership and upsert / delete semantics,
* polymorphic ``Message`` / ``BaseState`` round-trip through JSONB,
* cascading ``delete_team`` across all three tables,
* schema-drift / payload-authority invariant — promoted columns are
  routing keys only; ``data`` JSONB is the single source of truth.

Acceptance Criteria: AC #19 through AC #30 from story 17.2.

The Postgres directory is gated by ``conftest.py``'s
``pytest.importorskip("nagra")`` / ``pytest.importorskip("testcontainers.postgres")``
so this module never runs when those optional dependencies are missing.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import TYPE_CHECKING

import pytest
from akgentic.core.messages.message import UserMessage
from nagra import Transaction  # type: ignore[import-untyped]

from akgentic.team.models import TeamStatus
from akgentic.team.repositories.postgres import NagraEventStore

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

from tests.models.conftest import (
    SampleAgentState,
    make_agent_state_snapshot,
    make_persisted_event,
    make_process,
)


class TestNagraEventStore:
    """All nine EventStore Protocol methods plus payload-authority invariant."""

    # --- Protocol compliance (AC #19) -------------------------------------

    def test_satisfies_event_store_protocol(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #19: NagraEventStore satisfies EventStore Protocol structurally."""
        store: EventStore = NagraEventStore(postgres_clean_tables)
        assert store is not None

    # --- save_team / load_team round-trip (AC #20) ------------------------

    def test_save_and_load_team_round_trip(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #20: save_team upserts; load_team hydrates via JSONB."""
        store = NagraEventStore(postgres_clean_tables)
        process = make_process()
        store.save_team(process)
        loaded = store.load_team(process.team_id)

        assert loaded is not None
        assert loaded.team_id == process.team_id
        assert loaded.status == process.status
        assert loaded.team_card.name == process.team_card.name
        assert loaded.created_at == process.created_at

    def test_save_team_upserts_on_second_call(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #20: save_team upserts on subsequent calls for the same team_id."""
        store = NagraEventStore(postgres_clean_tables)
        process = make_process(status=TeamStatus.RUNNING)
        store.save_team(process)

        updated = make_process(team_id=process.team_id, status=TeamStatus.STOPPED)
        store.save_team(updated)

        loaded = store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.status == TeamStatus.STOPPED

    def test_load_team_returns_none_for_nonexistent(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #20: load_team returns None when no row matches."""
        store = NagraEventStore(postgres_clean_tables)
        assert store.load_team(uuid.uuid4()) is None

    # --- list_teams (AC #21) ----------------------------------------------

    def test_list_teams_empty(self, postgres_clean_tables: str) -> None:
        """AC #21: list_teams returns [] when no rows exist."""
        store = NagraEventStore(postgres_clean_tables)
        assert store.list_teams() == []

    def test_list_teams_returns_all(self, postgres_clean_tables: str) -> None:
        """AC #21: list_teams returns every persisted process (order-independent)."""
        store = NagraEventStore(postgres_clean_tables)
        p1 = make_process()
        p2 = make_process()
        store.save_team(p1)
        store.save_team(p2)

        result = store.list_teams()
        assert len(result) == 2
        assert {p.team_id for p in result} == {p1.team_id, p2.team_id}

    # --- save_event / load_events ordering (AC #22, #23) ------------------

    def test_load_events_returns_ordered_by_sequence(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #22: load_events returns rows sorted by sequence ASC.

        Inserts 100 events for one team with sequences shuffled and
        asserts the loaded order is the natural ascending sequence.
        """
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        sequences = random.sample(range(1, 101), 100)
        for seq in sequences:
            store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        loaded = store.load_events(team_id)
        assert [e.sequence for e in loaded] == list(range(1, 101))

    def test_load_events_empty(self, postgres_clean_tables: str) -> None:
        """AC #23: load_events returns [] for a team with no events."""
        store = NagraEventStore(postgres_clean_tables)
        assert store.load_events(uuid.uuid4()) == []

    # --- get_max_sequence (AC #24) ----------------------------------------

    def test_get_max_sequence(self, postgres_clean_tables: str) -> None:
        """AC #24: get_max_sequence returns 0 on empty, then largest seen."""
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()

        assert store.get_max_sequence(team_id) == 0

        for seq in (1, 5, 3):
            store.save_event(make_persisted_event(team_id=team_id, sequence=seq))
        assert store.get_max_sequence(team_id) == 5

        store.save_event(make_persisted_event(team_id=team_id, sequence=99))
        assert store.get_max_sequence(team_id) == 99

    # --- duplicate (team_id, sequence) propagates raw exception (AC #25) --

    def test_duplicate_sequence_raises_native_exception(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #25: composite-PK violation propagates as a raw exception.

        Asserts via ``pytest.raises(Exception)`` and inspects the message
        substring rather than pinning a specific psycopg version.
        """
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        store.save_event(make_persisted_event(team_id=team_id, sequence=1))

        with pytest.raises(Exception) as exc_info:
            store.save_event(make_persisted_event(team_id=team_id, sequence=1))

        msg = str(exc_info.value).lower()
        assert "unique" in msg or "event_entries" in msg

    # --- cascading delete_team (AC #26) -----------------------------------

    def test_delete_team_cascades_and_is_idempotent(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #26: delete_team purges all three tables for one team only.

        Other teams' rows remain. A second delete_team call is a no-op.
        """
        store = NagraEventStore(postgres_clean_tables)
        team_a = make_process()
        team_b = make_process()
        store.save_team(team_a)
        store.save_team(team_b)

        for seq in range(1, 6):
            store.save_event(
                make_persisted_event(team_id=team_a.team_id, sequence=seq)
            )
        for agent_id in ("a1", "a2", "a3"):
            store.save_agent_state(
                make_agent_state_snapshot(team_id=team_a.team_id, agent_id=agent_id)
            )

        for seq in range(1, 3):
            store.save_event(
                make_persisted_event(team_id=team_b.team_id, sequence=seq)
            )
        store.save_agent_state(
            make_agent_state_snapshot(team_id=team_b.team_id, agent_id="b1")
        )

        store.delete_team(team_a.team_id)

        assert store.load_team(team_a.team_id) is None
        assert store.load_events(team_a.team_id) == []
        assert store.load_agent_states(team_a.team_id) == []

        assert store.load_team(team_b.team_id) is not None
        assert len(store.load_events(team_b.team_id)) == 2
        assert len(store.load_agent_states(team_b.team_id)) == 1

        # Idempotent — second call must not raise.
        store.delete_team(team_a.team_id)

    # --- schema-drift / payload-authority invariant (AC #27) --------------

    def test_payload_is_authoritative_over_promoted_columns(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #27: hydrated model fields come from JSONB ``data`` only.

        Plants a row whose promoted ``team_id`` / ``sequence`` columns
        DISAGREE with the embedded payload, then asserts the hydrated
        ``PersistedEvent`` carries the payload values — proving the
        promoted columns are query keys only, never read back into the
        model. Bypasses ``save_event`` on purpose to plant the drift.
        """
        store = NagraEventStore(postgres_clean_tables)
        routing_team_id = uuid.uuid4()
        payload_team_id = uuid.uuid4()

        # Build a valid PersistedEvent payload, then mutate the dict to
        # disagree with the promoted columns we'll write.
        canonical = make_persisted_event(team_id=payload_team_id, sequence=42)
        payload_dict = canonical.model_dump()
        payload_dict["team_id"] = str(payload_team_id)
        payload_dict["sequence"] = 42

        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO event_entries (team_id, sequence, data) "
                "VALUES (%s, %s, %s)",
                (str(routing_team_id), 1, json.dumps(payload_dict)),
            )

        loaded = store.load_events(routing_team_id)
        assert len(loaded) == 1
        assert loaded[0].team_id == payload_team_id
        assert loaded[0].sequence == 42

    # --- polymorphic Message / BaseState round-trip (AC #28) --------------

    def test_polymorphic_event_round_trip(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #28: PersistedEvent carrying UserMessage survives JSONB round-trip."""
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        msg = UserMessage(content="hello")
        store.save_event(
            make_persisted_event(team_id=team_id, sequence=1, event=msg)
        )

        loaded = store.load_events(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].event, UserMessage)
        assert loaded[0].event.content == "hello"

    def test_polymorphic_agent_state_round_trip(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #28: AgentStateSnapshot carrying SampleAgentState survives round-trip."""
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        state = SampleAgentState(task_count=5)
        store.save_agent_state(
            make_agent_state_snapshot(team_id=team_id, agent_id="a", state=state)
        )

        loaded = store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 5

    # --- save_agent_state upsert (AC #29) ---------------------------------

    def test_save_agent_state_upserts_for_same_pair(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #29: save_agent_state upserts on the (team_id, agent_id) PK."""
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        store.save_agent_state(
            make_agent_state_snapshot(
                team_id=team_id,
                agent_id="agent-a",
                state=SampleAgentState(task_count=1),
            )
        )
        store.save_agent_state(
            make_agent_state_snapshot(
                team_id=team_id,
                agent_id="agent-a",
                state=SampleAgentState(task_count=99),
            )
        )

        loaded = store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 99

    # --- load_agent_states empty (AC #30) ---------------------------------

    def test_load_agent_states_empty(
        self, postgres_clean_tables: str
    ) -> None:
        """AC #30: load_agent_states returns [] for a team with no snapshots."""
        store = NagraEventStore(postgres_clean_tables)
        assert store.load_agent_states(uuid.uuid4()) == []
