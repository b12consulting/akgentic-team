"""Create and restore wired through the content-addressed card store (FR11-FR14).

Backend-agnostic store behaviour lives in
``tests/repositories/test_event_store_contract.py``, which runs it against yaml,
mongo and postgres. What is here is the *wiring*: the order the create path
writes in, the number of calls the restore path makes, the single resolution
site, and the Protocol conformance of every double in this package.

Two of these are only observable through a recording store. A reversed write
order and a per-role read both leave the exact same final state as the correct
version — the sequence and the count are the only witnesses.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState

from akgentic.team.manager import TeamManager
from akgentic.team.models import AgentCardRef, Process, TeamCard, TeamCardMember
from akgentic.team.ports import AgentCardNotFoundError, EventStore
from akgentic.team.projection import (
    derive_team_projection,
    hash_agent_card,
    resolve_agent_cards,
    storable_agent_card,
)
from akgentic.team.repositories.yaml import YamlEventStore
from tests.services.conftest import InMemoryEventStore

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent so a real team can be built and torn down."""


def _make_card(name: str, role: str) -> AgentCard:
    return AgentCard(
        description=f"Test: {role}",
        skills=["testing"],
        agent_class=StubAgent,
        config=BaseConfig(name=name, role=role),
    )


def _make_team_card(
    profiles: list[AgentCard] | None = None,
    name: str = "card-store-team",
) -> TeamCard:
    return TeamCard(
        name=name,
        description="A team whose cards land in the store",
        entry_point=TeamCardMember(card=_make_card("@Lead", "Lead")),
        members=[TeamCardMember(card=_make_card("@Worker", "Worker"))],
        agent_profiles=profiles or [],
    )


class RecordingEventStore(InMemoryEventStore):
    """``InMemoryEventStore`` that records the SEQUENCE of its write calls.

    ``CountingEventStore`` in ``test_manager_metadata`` is the precedent; this
    one records order rather than a count, because "cards before the document"
    is an ordering invariant and both orders leave the same final state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def save_agent_cards(self, cards: list[AgentCard]) -> None:
        """Record the write, then delegate."""
        self.calls.append("save_agent_cards")
        super().save_agent_cards(cards)

    def save_team(self, process: Process) -> None:
        """Record the write, then delegate."""
        self.calls.append("save_team")
        super().save_team(process)


@pytest.fixture()
def actor_system() -> ActorSystem:  # type: ignore[misc]
    """Provide an ActorSystem that shuts down after each test."""
    system = ActorSystem()
    yield system  # type: ignore[misc]
    system.shutdown()


@pytest.fixture()
def event_store() -> RecordingEventStore:
    """Provide a fresh call-recording event store per test."""
    return RecordingEventStore()


@pytest.fixture()
def manager(actor_system: ActorSystem, event_store: RecordingEventStore) -> TeamManager:
    return TeamManager(actor_system=actor_system, event_store=event_store)


# ---------------------------------------------------------------------------
# The create path (FR13)
# ---------------------------------------------------------------------------


class TestCreateTeamWritesCardsFirst:
    """AC 15-16: cards are written BEFORE the Process that references them."""

    def test_cards_are_saved_before_the_process(
        self, manager: TeamManager, event_store: RecordingEventStore
    ) -> None:
        """The ORDER, not merely that both happened.

        Asserting "both were called" passes on a reversed order, which is the
        refactor this is here to catch: a stored ``Process`` must never point at
        a blob that is not there.
        """
        manager.create_team(_make_team_card())

        assert event_store.calls.index("save_agent_cards") < event_store.calls.index(
            "save_team"
        )

    def test_every_ref_on_the_stored_process_resolves(
        self, manager: TeamManager, event_store: RecordingEventStore
    ) -> None:
        runtime = manager.create_team(_make_team_card())

        process = event_store.load_team(runtime.id)
        assert process is not None
        resolved = resolve_agent_cards(process.agent_cards, event_store)
        assert [c.role for c in resolved] == [r.role for r in process.agent_cards]

    def test_two_teams_from_one_card_store_one_blob_per_role(
        self, manager: TeamManager, event_store: RecordingEventStore
    ) -> None:
        """The deduplication the whole store exists for."""
        manager.create_team(_make_team_card())
        manager.create_team(_make_team_card())

        assert len(event_store.agent_cards) == 2  # Lead + Worker, not four

    def test_hireability_does_not_fork_the_stored_blob(
        self, manager: TeamManager, event_store: RecordingEventStore
    ) -> None:
        """AC 6: two teams differing ONLY in hireability write ONE card.

        This is the failure that excluding the flag from the hash alone does not
        prevent: both teams compute the same key, so without normalising the
        bytes they write *different content* to it and last-write-wins decides
        what everyone reads back.
        """
        # The profile is the tree card itself, not a same-role look-alike. A
        # profile overrides the tree card of its role in the projection, so a
        # differing one would make the two teams differ in card CONTENT as well
        # as hireability — and this test would then pass or fail for the wrong
        # reason.
        plain = manager.create_team(_make_team_card(name="plain"))
        hireable = manager.create_team(
            _make_team_card(profiles=[_make_card("@Worker", "Worker")], name="hireable")
        )

        stored = event_store.agent_cards
        assert len(stored) == 2
        assert all(card.can_be_hired is False for card in stored.values())

        plain_ref = _ref_for(event_store, plain.id, "Worker")
        hireable_ref = _ref_for(event_store, hireable.id, "Worker")
        assert plain_ref.card_hash == hireable_ref.card_hash
        assert plain_ref.can_be_hired is False
        assert hireable_ref.can_be_hired is True


def _ref_for(store: EventStore, team_id: uuid.UUID, role: str) -> AgentCardRef:
    """Return the stored ``Process``'s ref for *role*."""
    process = store.load_team(team_id)
    assert process is not None
    return next(ref for ref in process.agent_cards if ref.role == role)


# ---------------------------------------------------------------------------
# The restore path (FR11)
# ---------------------------------------------------------------------------


class TestRestoreResolvesInOneCall:
    """AC 17: the whole card set in EXACTLY one ``load_agent_cards`` call."""

    def test_restore_issues_exactly_one_load_agent_cards_call(
        self, manager: TeamManager, event_store: RecordingEventStore, actor_system: ActorSystem
    ) -> None:
        """The call COUNT, not the result.

        A per-role read returns exactly the same cards, so the result cannot
        distinguish the N+1 this design exists to prevent. Only the count can,
        and the loop is what a refactor reaches for first.
        """
        tc = _make_team_card()
        runtime = manager.create_team(tc)
        team_id = runtime.id
        manager.stop_team(team_id)

        process = event_store.load_team(team_id)
        assert process is not None
        assert len(process.agent_cards) > 1

        event_store.load_agent_cards_calls = 0
        manager.resume_team(team_id)

        assert event_store.load_agent_cards_calls == 1


# ---------------------------------------------------------------------------
# The one resolution site (AC 19) and the loud failure (AC 20)
# ---------------------------------------------------------------------------


class TestResolveAgentCards:
    """The single function turning refs + a store into cards."""

    def test_it_issues_one_batch_call_whatever_the_ref_count(self) -> None:
        store = RecordingEventStore()
        cards = [_make_card(f"@Agent{i}", f"Role{i}") for i in range(4)]
        store.save_agent_cards(cards)
        refs = [
            AgentCardRef(role=c.role, card_hash=hash_agent_card(c)) for c in cards
        ]

        store.load_agent_cards_calls = 0
        resolved = resolve_agent_cards(refs, store)

        assert store.load_agent_cards_calls == 1
        assert [c.role for c in resolved] == [r.role for r in refs]

    def test_it_carries_each_refs_hireability_onto_the_card(self) -> None:
        store = RecordingEventStore()
        card = _make_card("@Specialist", "Specialist")
        store.save_agent_cards([card])
        card_hash = hash_agent_card(card)

        hireable = resolve_agent_cards(
            [AgentCardRef(role="Specialist", card_hash=card_hash, can_be_hired=True)], store
        )
        plain = resolve_agent_cards(
            [AgentCardRef(role="Specialist", card_hash=card_hash)], store
        )

        assert hireable[0].can_be_hired is True
        assert plain[0].can_be_hired is False

    def test_it_does_not_mutate_the_card_the_store_handed_back(self) -> None:
        """A caching backend may share the object with the next caller."""
        store = RecordingEventStore()
        card = _make_card("@Specialist", "Specialist")
        store.save_agent_cards([card])
        card_hash = hash_agent_card(card)

        resolve_agent_cards(
            [AgentCardRef(role="Specialist", card_hash=card_hash, can_be_hired=True)], store
        )

        assert store.agent_cards[card_hash].can_be_hired is False

    def test_an_empty_ref_list_resolves_to_nothing(self) -> None:
        assert resolve_agent_cards([], RecordingEventStore()) == []

    def test_a_missing_hash_raises_naming_the_role_and_the_hash(self) -> None:
        store = RecordingEventStore()
        missing = "a" * 64

        with pytest.raises(AgentCardNotFoundError) as excinfo:
            resolve_agent_cards([AgentCardRef(role="Ghost", card_hash=missing)], store)

        message = str(excinfo.value)
        assert "Ghost" in message
        assert missing in message

    def test_the_error_is_a_lookup_error_not_a_value_error(self) -> None:
        """``ValueError`` would be swallowed by the corrupted-document handlers.

        ``yaml.py`` and ``mongo.py`` both catch ``ValueError`` around document
        hydration — the same reason ``EventNotFoundError`` subclasses
        ``LookupError``.
        """
        assert issubclass(AgentCardNotFoundError, LookupError)
        assert not issubclass(AgentCardNotFoundError, ValueError)

    def test_a_restore_with_an_unresolvable_hash_fails_loudly(
        self, manager: TeamManager, event_store: RecordingEventStore
    ) -> None:
        """FR14: never a team the orchestrator cannot describe."""
        runtime = manager.create_team(_make_team_card())
        team_id = runtime.id
        manager.stop_team(team_id)
        event_store.agent_cards.clear()

        with pytest.raises(AgentCardNotFoundError):
            manager.resume_team(team_id)


# ---------------------------------------------------------------------------
# delete_team never touches the store (FR13)
# ---------------------------------------------------------------------------


class TestDeleteTeamLeavesTheStore:
    """AC 21, at the manager level; the backend level is in the contract suite."""

    def test_a_sharing_team_still_resumes_after_a_delete(
        self, manager: TeamManager, event_store: RecordingEventStore
    ) -> None:
        """Mutation-verified: purging the cards in ``delete_team`` turns this red."""
        deleted = manager.create_team(_make_team_card(name="doomed"))
        survivor = manager.create_team(_make_team_card(name="survivor"))
        manager.stop_team(deleted.id)
        manager.stop_team(survivor.id)

        manager.delete_team(deleted.id)

        resumed = manager.resume_team(survivor.id)
        assert resumed is not None


# ---------------------------------------------------------------------------
# Protocol conformance (AC 22-23) — the trap this story exists for
# ---------------------------------------------------------------------------


def _protocol_members() -> list[str]:
    """Every public member the ``EventStore`` Protocol declares."""
    return [name for name in vars(EventStore) if not name.startswith("_")]


class TestEventStoreConformance:
    """Nothing else catches a double that stops conforming.

    ``EventStore`` is a bare ``Protocol`` — only ``ServiceRegistry`` carries
    ``@runtime_checkable`` — and this package's CI runs ``mypy src/`` only. So
    ``cli/main.py``'s ``-> EventStore`` return catches the YAML and Mongo
    backends loudly, and catches ``NagraEventStore`` and every fake under
    ``tests/`` not at all: they simply stop conforming, in silence, until
    something calls the new method.

    ``isinstance`` is unavailable here by design — making the Protocol
    runtime-checkable belongs with the CI-mypy widening (``backlog.md`` row 12),
    and it would check attribute presence only, not signatures, so it would be
    no stronger than this.
    """

    def test_the_protocol_declares_the_two_card_methods(self) -> None:
        """Guards the guard: if the members vanish, the sweep below is vacuous."""
        assert {"save_agent_cards", "load_agent_cards"} <= set(_protocol_members())

    @pytest.mark.parametrize(
        "double",
        [
            pytest.param(InMemoryEventStore(), id="InMemoryEventStore"),
            pytest.param(RecordingEventStore(), id="RecordingEventStore"),
        ],
    )
    def test_every_test_double_implements_the_whole_protocol(self, double: Any) -> None:
        for member in _protocol_members():
            assert callable(getattr(double, member, None)), member

    def test_the_counting_store_from_test_manager_metadata_conforms(self) -> None:
        """It subclasses ``InMemoryEventStore`` and overrides only ``save_team``."""
        from tests.services.test_manager_metadata import CountingEventStore

        for member in _protocol_members():
            assert callable(getattr(CountingEventStore(), member, None)), member

    def test_the_yaml_backend_implements_the_whole_protocol(self, tmp_path: Any) -> None:
        store = YamlEventStore(tmp_path)
        for member in _protocol_members():
            assert callable(getattr(store, member, None)), member

    def test_the_mongo_backend_implements_the_whole_protocol(self) -> None:
        pytest.importorskip("pymongo")
        pytest.importorskip("mongomock")
        import mongomock

        from akgentic.team.repositories.mongo import MongoEventStore

        store = MongoEventStore(mongomock.MongoClient()["conformance"])
        for member in _protocol_members():
            assert callable(getattr(store, member, None)), member

    def test_the_postgres_backend_implements_the_whole_protocol(self) -> None:
        """The one no mypy site and no fixture would otherwise catch.

        ``NagraEventStore`` is never returned from an ``EventStore``-annotated
        site in ``src/``, so mypy never checks it — which is exactly why an
        unimplemented method here would surface as an ``AttributeError`` at team
        creation on a Postgres deployment rather than at review.
        """
        pytest.importorskip("nagra")
        from akgentic.team.repositories.postgres import NagraEventStore

        for member in _protocol_members():
            assert callable(getattr(NagraEventStore, member, None)), member


# ---------------------------------------------------------------------------
# The stored form (AC 6) at the function level
# ---------------------------------------------------------------------------


class TestStorableAgentCard:
    """The normalisation that makes the stored bytes a pure function of the hash."""

    def test_it_normalises_the_hireable_flag_off(self) -> None:
        hireable = _make_card("@Specialist", "Specialist").model_copy(
            update={"can_be_hired": True}
        )
        assert storable_agent_card(hireable).can_be_hired is False

    def test_it_leaves_the_hash_unchanged(self) -> None:
        hireable = _make_card("@Specialist", "Specialist").model_copy(
            update={"can_be_hired": True}
        )
        assert hash_agent_card(storable_agent_card(hireable)) == hash_agent_card(hireable)

    def test_it_does_not_mutate_its_argument(self) -> None:
        hireable = _make_card("@Specialist", "Specialist").model_copy(
            update={"can_be_hired": True}
        )
        storable_agent_card(hireable)
        assert hireable.can_be_hired is True

    def test_it_changes_nothing_else(self) -> None:
        """Start from a HIREABLE card, or the assertion holds for a card that
        was already normalised and would pass on a function returning its
        argument unchanged — proving nothing about "changes nothing else".
        """
        card = _make_card("@Lead", "Lead").model_copy(update={"can_be_hired": True})
        storable = storable_agent_card(card)

        assert storable.can_be_hired is False
        assert storable.model_dump() == card.model_dump() | {"can_be_hired": False}

    def test_the_projection_cards_keep_their_flag_for_the_other_consumer(self) -> None:
        """The store wants the flagless form; the orchestrator wants the flagged one.

        Normalising inside ``derive_team_projection`` would serve the store and
        break the registration, which is why it happens on the store's side of
        the boundary and not in the derivation.
        """
        tc = _make_team_card(profiles=[_make_card("@AnotherWorker", "Worker")])
        projection = derive_team_projection(tc)

        flags = {c.role: c.can_be_hired for c in projection.cards}
        assert flags == {"Lead": False, "Worker": True}
