"""Conformance tests for the ``InMemoryEventStore`` test fake.

The fake in ``conftest.py`` stands in for a real ``EventStore`` across the
service suite, but nothing else calls its ``list_teams`` — so its filter
semantics have no other coverage, and mypy does not run over ``tests/``.
That combination is exactly how this fake drifted out of Protocol shape
between epic 19 and epic 26. These tests pin the semantics so the next
drift fails a test instead of going unnoticed.
"""

from __future__ import annotations

import time

import pytest

from akgentic.team.models import TeamStatus
from akgentic.team.projection import hash_agent_card, storable_agent_card
from tests.models.conftest import AcmeTeamMetadata, make_indexed_process, make_process
from tests.repositories.test_event_store_contract import (
    METADATA_FILTER_IDS,
    METADATA_FILTER_MATRIX,
    _make_card,
    build_metadata_fixture_set,
)
from tests.services.conftest import InMemoryEventStore


class TestInMemoryEventStoreListTeams:
    """``list_teams`` on the fake must match the Protocol it stands in for."""

    def test_no_argument_call_returns_every_team_including_deleted(self) -> None:
        """No filter means no filter — ``DELETED`` is in the default result set."""
        store = InMemoryEventStore()
        running = make_process(status=TeamStatus.RUNNING)
        deleted = make_process(status=TeamStatus.DELETED)
        store.save_team(running)
        store.save_team(deleted)

        assert {p.team_id for p in store.list_teams()} == {running.team_id, deleted.team_id}

    def test_filters_by_user_id(self) -> None:
        """``user_id`` selects only that owner's snapshots."""
        store = InMemoryEventStore()
        mine = make_process(user_id="u1")
        theirs = make_process(user_id="u2")
        store.save_team(mine)
        store.save_team(theirs)

        assert {p.team_id for p in store.list_teams(user_id="u1")} == {mine.team_id}

    def test_filters_by_status(self) -> None:
        """``status`` selects only snapshots in that lifecycle state."""
        store = InMemoryEventStore()
        running = make_process(status=TeamStatus.RUNNING)
        stopped = make_process(status=TeamStatus.STOPPED)
        store.save_team(running)
        store.save_team(stopped)

        assert {p.team_id for p in store.list_teams(status=TeamStatus.RUNNING)} == {
            running.team_id
        }

    def test_both_filters_combine_with_and(self) -> None:
        """The two filters are independent and intersect.

        Unlike the real backends — whose push-downs land in stories 26.2 /
        26.3 / 26.5 — the fake's AND semantics are final here, so they are
        asserted rather than deferred.
        """
        store = InMemoryEventStore()
        wanted = make_process(user_id="u1", status=TeamStatus.RUNNING)
        wrong_status = make_process(user_id="u1", status=TeamStatus.STOPPED)
        wrong_user = make_process(user_id="u2", status=TeamStatus.RUNNING)
        for process in (wanted, wrong_status, wrong_user):
            store.save_team(process)

        result = store.list_teams(user_id="u1", status=TeamStatus.RUNNING)
        assert {p.team_id for p in result} == {wanted.team_id}

    def test_user_id_is_accepted_positionally(self) -> None:
        """The fake keeps the same appended-parameter shape as the real backends."""
        store = InMemoryEventStore()
        mine = make_process(user_id="u1")
        store.save_team(mine)
        store.save_team(make_process(user_id="u2"))

        assert {p.team_id for p in store.list_teams("u1")} == {mine.team_id}

    def test_filters_by_metadata(self) -> None:
        """``metadata`` selects only teams carrying that key/value pair."""
        store = InMemoryEventStore()
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        contoso = make_indexed_process(AcmeTeamMetadata(tenant="contoso"))
        store.save_team(acme)
        store.save_team(contoso)

        result = store.list_teams(metadata={"tenant": ["acme"]})
        assert {p.team_id for p in result} == {acme.team_id}

    def test_metadata_and_combines_across_keys(self) -> None:
        """Two entries mean BOTH must be present — a team with one is excluded."""
        store = InMemoryEventStore()
        both = make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-1"))
        tenant_only = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        store.save_team(both)
        store.save_team(tenant_only)

        result = store.list_teams(metadata={"tenant": ["acme"], "case_ref": ["C-1"]})
        assert {p.team_id for p in result} == {both.team_id}

    def test_metadata_combines_with_user_id(self) -> None:
        """Identical metadata under two owners stays scoped to the asked-for owner."""
        store = InMemoryEventStore()
        mine = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u1")
        theirs = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u2")
        store.save_team(mine)
        store.save_team(theirs)

        result = store.list_teams(user_id="u1", metadata={"tenant": ["acme"]})
        assert {p.team_id for p in result} == {mine.team_id}

    def test_metadata_combines_with_status(self) -> None:
        """``metadata`` and ``status`` intersect rather than union."""
        store = InMemoryEventStore()
        running = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), status=TeamStatus.RUNNING
        )
        stopped = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), status=TeamStatus.STOPPED
        )
        store.save_team(running)
        store.save_team(stopped)

        result = store.list_teams(status=TeamStatus.RUNNING, metadata={"tenant": ["acme"]})
        assert {p.team_id for p in result} == {running.team_id}

    def test_all_three_filters_combine_with_and(self) -> None:
        """Every term narrows; only the team satisfying all three survives.

        Each single term matches more than one team here, so a fake that
        honoured just one of them — or ORed them — returns the wrong set.
        """
        store = InMemoryEventStore()
        wanted = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.RUNNING
        )
        wrong_metadata = make_indexed_process(
            AcmeTeamMetadata(tenant="contoso"), user_id="u1", status=TeamStatus.RUNNING
        )
        wrong_status = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.STOPPED
        )
        wrong_user = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u2", status=TeamStatus.RUNNING
        )
        for process in (wanted, wrong_metadata, wrong_status, wrong_user):
            store.save_team(process)

        result = store.list_teams(
            user_id="u1", status=TeamStatus.RUNNING, metadata={"tenant": ["acme"]}
        )
        assert {p.team_id for p in result} == {wanted.team_id}

    def test_empty_metadata_dict_matches_everything(self) -> None:
        """``metadata={}`` is an empty conjunction — identical to ``metadata=None``."""
        store = InMemoryEventStore()
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        bare = make_process()
        store.save_team(acme)
        store.save_team(bare)

        assert {p.team_id for p in store.list_teams(metadata={})} == {
            acme.team_id,
            bare.team_id,
        }
        assert {p.team_id for p in store.list_teams(metadata=None)} == {
            acme.team_id,
            bare.team_id,
        }

    def test_metadata_on_unknown_key_returns_empty(self) -> None:
        """A key no team carries — including an unindexed field — matches nothing."""
        store = InMemoryEventStore()
        store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme", department="ops")))

        assert store.list_teams(metadata={"nope": ["acme"]}) == []
        # `department` is declared but NOT indexed, so it is not matchable.
        assert store.list_teams(metadata={"department": ["ops"]}) == []


class TestInMemoryEventStoreMatchesTheSharedMatrix:
    """The fake answers the contract suite's matrix exactly as the backends do.

    Imported rather than restated, so this provably runs the SAME cases the
    three real backends run. Without it a fake left on whole-entry equality
    drifts with no failure anywhere: ``EventStore`` is a plain ``Protocol``, not
    a ``@runtime_checkable`` one, and CI mypy covers ``src/`` only.
    """

    @pytest.mark.parametrize(
        "case_label,metadata,expected_labels",
        METADATA_FILTER_MATRIX,
        ids=METADATA_FILTER_IDS,
    )
    def test_metadata_filter_matrix(
        self,
        case_label: str,
        metadata: dict[str, list[str]] | None,
        expected_labels: set[str] | frozenset[str],
    ) -> None:
        store = InMemoryEventStore()
        teams = build_metadata_fixture_set()
        for process in teams.values():
            store.save_team(process)

        found = {p.team_id for p in store.list_teams(metadata=metadata)}
        assert found == {teams[label].team_id for label in expected_labels}, case_label

    def test_a_bare_string_metadata_value_is_rejected(self) -> None:
        """The fake rejects the un-migrated call shape too, or it hides the bug."""
        store = InMemoryEventStore()
        store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))

        with pytest.raises(TypeError, match="tenant"):
            store.list_teams(metadata={"tenant": "acme"})  # type: ignore[dict-item]


class TestInMemoryEventStoreCardEnumeration:
    """The fake's card store must answer what the real backends would answer.

    The conformance sweep proves the method EXISTS; only a spec proves it does
    not lie. A fake that returned a constant — always ``None``, always ``now`` —
    would pass the sweep and could never fail a test the real backends fail,
    which is the exact shape of drift this module was written for.
    """

    def test_an_empty_store_enumerates_to_nothing(self) -> None:
        assert InMemoryEventStore().list_agent_card_entries() == []

    def test_every_saved_card_enumerates_once_with_a_tz_aware_stamp(self) -> None:
        store = InMemoryEventStore()
        cards = [_make_card("lead", "Lead"), _make_card("writer", "Writer")]

        store.save_agent_cards(cards)

        entries = store.list_agent_card_entries()
        assert {e.card_hash for e in entries} == {hash_agent_card(c) for c in cards}
        for entry in entries:
            assert entry.first_seen_at is not None
            assert entry.first_seen_at.tzinfo is not None

    def test_a_re_save_does_not_move_the_stamp(self) -> None:
        """Insert-only, as on all three real backends — ``setdefault``, not assignment.

        Mutation-verified: an assignment in ``save_agent_cards`` turns this red
        and nothing else in the suite.
        """
        store = InMemoryEventStore()
        card = _make_card("lead", "Lead")
        store.save_agent_cards([card])
        (first,) = store.list_agent_card_entries()

        time.sleep(0.01)
        store.save_agent_cards([card])
        store.save_agent_cards([card, _make_card("writer", "Writer")])

        entries = {e.card_hash: e for e in store.list_agent_card_entries()}
        assert entries[first.card_hash].first_seen_at == first.first_seen_at

    def test_a_blob_planted_without_a_stamp_enumerates_as_unknown(self) -> None:
        """The fake's version of a pre-existing blob: a key with no stamp beside it."""
        store = InMemoryEventStore()
        card = _make_card("lead", "Lead")
        store.agent_cards[hash_agent_card(card)] = storable_agent_card(card)

        (entry,) = store.list_agent_card_entries()

        assert entry.first_seen_at is None

    def test_re_saving_an_unstamped_blob_does_not_stamp_it(self) -> None:
        store = InMemoryEventStore()
        card = _make_card("lead", "Lead")
        card_hash = hash_agent_card(card)
        store.agent_cards[card_hash] = storable_agent_card(card)
        store.agent_card_first_seen[card_hash] = None

        store.save_agent_cards([card])

        (entry,) = store.list_agent_card_entries()
        assert entry.first_seen_at is None
