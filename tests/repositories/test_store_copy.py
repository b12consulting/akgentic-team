"""Tests for :func:`akgentic.team.store_copy.copy_store`.

The source is always a YAML store (what the copy script reads). The target is
the shared parametrized ``event_store`` fixture, so every test runs once per
backend — ``[yaml]``, ``[mongo]``, ``[postgres]`` — and the copy is shown to
work through the Protocol alone.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from akgentic.core.agent import Akgent

from akgentic.team.models import Process, TeamStatus
from akgentic.team.ports import EventStore
from akgentic.team.projection import resolve_agent_cards
from akgentic.team.repositories.yaml import YamlEventStore
from akgentic.team.store_copy import copy_store
from tests.conftest import seed_agent_cards
from tests.models.conftest import (
    make_agent_state_snapshot,
    make_persisted_event,
    make_process,
    make_team_card,
)


def _seed_team(
    store: EventStore,
    *,
    status: TeamStatus = TeamStatus.RUNNING,
    events: int = 3,
    states: int = 2,
) -> Process:
    """A team as the real write path leaves it: cards, then team, events, states."""
    # Akgent, not the default MockAgent: the store loads cards back on read, and
    # a class it cannot import reads as a corrupted card.
    team_card = make_team_card(agent_class=Akgent)
    seed_agent_cards(store, team_card)
    process = make_process(team_card=team_card, status=status)
    store.save_team(process)
    for sequence in range(1, events + 1):
        store.save_event(make_persisted_event(team_id=process.team_id, sequence=sequence))
    for index in range(states):
        store.save_agent_state(
            make_agent_state_snapshot(team_id=process.team_id, agent_id=f"agent-{index}")
        )
    return process


@pytest.fixture
def source(tmp_path: Path) -> YamlEventStore:
    return YamlEventStore(tmp_path / "source")


def _all_team_ids(store: EventStore) -> set[uuid.UUID]:
    teams = [*store.list_teams(), *store.list_teams(status=TeamStatus.DELETED)]
    return {p.team_id for p in teams}


class TestCopyStore:
    """Every team arrives whole, whatever its status, on every target backend."""

    def test_copies_teams_events_states_and_cards(
        self, source: YamlEventStore, event_store: EventStore
    ) -> None:
        running = _seed_team(source, status=TeamStatus.RUNNING, events=3, states=2)
        stopped = _seed_team(source, status=TeamStatus.STOPPED, events=1, states=0)
        retired = _seed_team(source, status=TeamStatus.DELETED, events=2, states=1)

        report = copy_store(source, event_store)

        assert not report.failed
        assert (report.teams, report.events, report.agent_states) == (3, 6, 3)
        assert _all_team_ids(event_store) == {
            running.team_id,
            stopped.team_id,
            retired.team_id,
        }
        for original in (running, stopped, retired):
            copied = event_store.load_team(original.team_id)
            assert copied is not None
            assert copied.status == original.status
            assert [e.sequence for e in event_store.load_events(original.team_id)] == [
                e.sequence for e in source.load_events(original.team_id)
            ]
            assert len(event_store.load_agent_states(original.team_id)) == len(
                source.load_agent_states(original.team_id)
            )
            # The team's cards resolve on the target — restore would need them.
            assert len(resolve_agent_cards(copied.agent_cards, event_store)) == len(
                copied.agent_cards
            )

    def test_running_it_twice_copies_nothing_new(
        self, source: YamlEventStore, event_store: EventStore
    ) -> None:
        """Re-running is safe — no duplicate-sequence error on a database target."""
        team = _seed_team(source, events=3)
        copy_store(source, event_store)

        again = copy_store(source, event_store)

        assert not again.failed
        assert again.events == 0
        assert len(event_store.load_events(team.team_id)) == 3

    def test_a_partial_copy_is_finished_by_the_next_run(
        self, source: YamlEventStore, event_store: EventStore
    ) -> None:
        """Only events above the target's max sequence are written."""
        team = _seed_team(source, events=5)
        for event in source.load_events(team.team_id)[:2]:
            event_store.save_event(event)

        report = copy_store(source, event_store)

        assert report.events == 3
        assert [e.sequence for e in event_store.load_events(team.team_id)] == [1, 2, 3, 4, 5]

    def test_dry_run_counts_and_writes_nothing(
        self, source: YamlEventStore, event_store: EventStore
    ) -> None:
        team = _seed_team(source, events=4, states=2)

        report = copy_store(source, event_store, dry_run=True)

        assert report.dry_run
        assert (report.teams, report.events, report.agent_states) == (1, 4, 2)
        assert event_store.load_team(team.team_id) is None
        assert event_store.load_events(team.team_id) == []


class TestCopyStoreFailures:
    """A team that cannot be copied is reported and never listed half-copied."""

    def test_a_team_whose_card_is_missing_fails_alone(
        self, source: YamlEventStore, event_store: EventStore
    ) -> None:
        good = _seed_team(source)
        # A role no saved card has: content addressing would otherwise share
        # the good team's blob and the orphan's card would exist after all.
        orphan_card = make_team_card(
            entry_point_name="orphan", entry_point_role="Orphan", agent_class=Akgent
        )
        orphan = make_process(team_card=orphan_card)  # its cards are never saved
        source.save_team(orphan)

        report = copy_store(source, event_store)

        assert [f.team_id for f in report.failed] == [orphan.team_id]
        assert "no card" in report.failed[0].reason
        assert event_store.load_team(orphan.team_id) is None
        assert event_store.load_team(good.team_id) is not None

    def test_a_team_whose_event_log_will_not_parse_fails_alone(
        self, source: YamlEventStore, event_store: EventStore, tmp_path: Path
    ) -> None:
        good = _seed_team(source)
        broken = _seed_team(source)
        events_file = tmp_path / "source" / str(broken.team_id) / "events.yaml"
        assert events_file.exists()
        events_file.write_text("{ this: is: not yaml", encoding="utf-8")

        report = copy_store(source, event_store)

        assert [f.team_id for f in report.failed] == [broken.team_id]
        assert event_store.load_team(broken.team_id) is None
        assert event_store.load_team(good.team_id) is not None
