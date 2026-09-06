"""Mongo-specific tests for ``MongoEventStore``.

Behavioural Protocol coverage (round-trip, upsert, list, sequencing,
max sequence, cascading delete, polymorphic round-trips) lives in the
shared ``tests/repositories/test_event_store_contract.py`` and runs
once per backend. This module retains only Mongo-specific invariants:

* Protocol structural-typing check.
* Corrupted-document resilience (mongo's analogue of YAML's corrupted-
  file resilience and the postgres payload-authority test).
* Import guards — ``MongoEventStore`` is only available with the
  ``[mongo]`` extra installed.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from pymongo.errors import OperationFailure

from akgentic.team.projection import hash_agent_card
from akgentic.team.repositories.mongo import (
    AGENT_CARDS_COLLECTION,
    MongoEventStore,
    ensure_indexes,
)

if TYPE_CHECKING:
    from pathlib import Path

    from akgentic.team.models import Process
    from akgentic.team.ports import EventStore

from akgentic.team.models import TeamStatus
from tests.models.conftest import (
    AcmeTeamMetadata,
    make_agent_card,
    make_agent_state_snapshot,
    make_indexed_process,
    make_persisted_event,
    make_process,
)


def _card_fixture() -> AgentCard:
    """An AgentCard whose ``agent_class`` resolves on the way back out of storage."""
    return make_agent_card(name="lead", role="Lead", agent_class=Akgent)


def _metadata_fixtures() -> list[Process]:
    """Six teams spanning the metadata filter matrix, in a fixed order.

    Built through ``make_indexed_process`` so ``metadata`` and its derived
    ``metadata_indexes`` are seeded together exactly as a real write path
    leaves them — an index seeded by hand would let a query-side escaping bug
    pass. Two tenants, two owning users, one ``case_ref`` holding a literal
    ``|``, and one team carrying no metadata at all: that last one must be
    excluded by every non-empty filter and returned by ``metadata=None``.

    Returns:
        The teams, positionally addressable by the parity matrix's indices.
    """
    return [
        make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-1"), user_id="u1"),
        make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C|1234"), user_id="u1"),
        make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-2"), user_id="u2"),
        make_indexed_process(AcmeTeamMetadata(tenant="contoso", case_ref="C-1"), user_id="u2"),
        make_indexed_process(AcmeTeamMetadata(tenant="contoso"), user_id="u1"),
        make_indexed_process(None, user_id="u2"),
    ]


def _assert_ungated_indexes_present(db: Any) -> None:
    """Assert the three non-teams indexes survive a teams-collection opt-out.

    ``auto_create_indexes`` / ``MONGO_TEAM_AUTO_INDEX`` gate the teams
    collection only. The ``events`` and ``agent_states`` indexes have been
    built on every boot since those backends shipped, so they already exist in
    production and the call there is a cheap no-op; gating them would be an
    unrequested behaviour change. Called from every opt-out test so a future
    refactor cannot widen the gate unnoticed.
    """
    events_info = db["events"].index_information()
    assert [("team_id", 1), ("sequence", 1)] in [e["key"] for e in events_info.values()]
    assert "events_event_id_idx" in events_info

    agent_info = db["agent_states"].index_information()
    compound = [
        e for e in agent_info.values() if e["key"] == [("team_id", 1), ("agent_id", 1)]
    ]
    assert compound, f"agent_states (team_id, agent_id) index missing: {agent_info}"
    assert compound[0].get("unique") is True


class TestMongoEventStoreMongoSpecific:
    """Mongo-only invariants — see contract suite for behavioural coverage."""

    # --- Protocol compliance ------------------------------------------------

    def test_satisfies_event_store_protocol(self, mongo_store: MongoEventStore) -> None:
        """``MongoEventStore`` satisfies ``EventStore`` Protocol structurally."""
        store: EventStore = mongo_store
        assert store is not None

    # --- Corrupted-document resilience --------------------------------------

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
        db["events"].insert_one({"team_id": str(team_id), "sequence": 2, "corrupted": True})

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

    # --- list_teams push-down -----------------------------------------------

    def test_list_teams_pushes_status_into_the_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """``status`` reaches MongoDB inside the ``find`` filter, not Python.

        An in-memory filter returns the same list, so the assertion has to
        bite on the shape of the call: exactly one ``find``, carrying
        ``{"status": "running"}``. ``wraps=`` keeps the real query running
        so the result is pinned by the same test.
        """
        mongo_store.save_team(make_process(status=TeamStatus.RUNNING))
        mongo_store.save_team(make_process(status=TeamStatus.STOPPED))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(status=TeamStatus.RUNNING)

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {"status": "running"}
        assert len(result) == 1
        assert result[0].status == TeamStatus.RUNNING

    def test_list_teams_pushes_user_id_and_status_into_one_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """Both filters travel down in a single ``find`` call, ANDed.

        Never two ``find`` calls intersected afterwards, and never a
        ``find`` plus a second pass in Python.
        """
        mine_running = make_process(user_id="u1", status=TeamStatus.RUNNING)
        mongo_store.save_team(mine_running)
        mongo_store.save_team(make_process(user_id="u1", status=TeamStatus.STOPPED))
        mongo_store.save_team(make_process(user_id="u2", status=TeamStatus.RUNNING))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(user_id="u1", status=TeamStatus.RUNNING)

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {"user_id": "u1", "status": "running"}
        assert [p.team_id for p in result] == [mine_running.team_id]

    def test_list_teams_without_filters_issues_an_empty_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """A ``None`` parameter contributes no key — the filter stays ``{}``.

        Both the no-argument call and the explicit ``user_id=None,
        status=None`` call must reach MongoDB unfiltered and return every
        lifecycle state, ``DELETED`` included.
        """
        mongo_store.save_team(make_process(status=TeamStatus.RUNNING))
        mongo_store.save_team(make_process(status=TeamStatus.DELETED))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            no_argument = mongo_store.list_teams()

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {}

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            explicit_none = mongo_store.list_teams(user_id=None, status=None)

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {}
        assert len(no_argument) == 2
        assert {p.team_id for p in explicit_none} == {p.team_id for p in no_argument}

    def test_list_teams_positional_user_id_still_reaches_the_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """``list_teams("u1")`` keeps working and still pushes ``user_id`` down.

        Guards the parameter order against the rewrite: a keyword-only
        regression would be caught at the Mongo layer too, not only in the
        shared contract suite.
        """
        mine = make_process(user_id="u1", status=TeamStatus.RUNNING)
        mongo_store.save_team(mine)
        mongo_store.save_team(make_process(user_id="u2", status=TeamStatus.RUNNING))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams("u1")

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {"user_id": "u1"}
        assert [p.team_id for p in result] == [mine.team_id]

    def test_list_teams_does_not_filter_status_in_python(
        self, mongo_store: MongoEventStore
    ) -> None:
        """No post-hydration ``status`` comparison survives in the method body.

        ``find`` is stubbed to hand back BOTH documents whatever filter the
        store asks for. Any surviving Python-side comparison would drop the
        stopped team while leaving every result-level assertion elsewhere
        green — which is exactly what this test exists to catch.
        """
        running = make_process(status=TeamStatus.RUNNING)
        stopped = make_process(status=TeamStatus.STOPPED)
        mongo_store.save_team(running)
        mongo_store.save_team(stopped)

        all_docs = list(mongo_store._teams.find({}))
        with patch.object(mongo_store._teams, "find", return_value=iter(all_docs)):
            result = mongo_store.list_teams(status=TeamStatus.RUNNING)

        assert {p.team_id for p in result} == {running.team_id, stopped.team_id}

    def test_list_teams_skips_corrupted_documents(
        self, mongo_store: MongoEventStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupted team document is skipped, logged, and does not raise.

        Seeded beside a healthy team so a ``list_teams`` broken to return
        nothing cannot pass: the healthy team must still come back.
        """
        healthy = make_process(status=TeamStatus.RUNNING)
        mongo_store.save_team(healthy)
        mongo_store._teams.insert_one(
            {"team_id": str(uuid.uuid4()), "not_a_valid_field": 123}
        )

        with caplog.at_level(logging.WARNING):
            result = mongo_store.list_teams()

        assert {p.team_id for p in result} == {healthy.team_id}
        assert "Skipping corrupted team document" in caplog.text

    # --- user_id index presence ---------------------------------------------

    def test_teams_user_id_index_created_on_init(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """``MongoEventStore.__init__`` creates ``teams_user_id_idx`` on ``teams``.

        Asserts the index is present with a single-field ascending key spec
        on ``user_id``. Backs the ``find({"user_id": ...})`` push-down in
        ``list_teams`` (ADR-16 §5).
        """
        info = mongo_db["teams"].index_information()
        assert "teams_user_id_idx" in info
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]

    def test_teams_user_id_index_creation_is_idempotent(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """Re-running the constructor against the same database does not raise.

        PyMongo's ``create_index`` returns silently when an index with the
        same key spec already exists, so redeploys against a long-lived
        MongoDB are safe (ADR-16 §5). After the second construction the
        ``teams_user_id_idx`` entry is still present exactly once with the
        same key spec.
        """
        # First construction already happened via the ``mongo_store`` fixture.
        MongoEventStore(mongo_db)  # second construction — must not raise

        info = mongo_db["teams"].index_information()
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        assert list(info.keys()).count("teams_user_id_idx") == 1

    # --- event.id index presence --------------------------------------------

    def test_events_event_id_index_created_on_init(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """``__init__`` creates ``events_event_id_idx`` on ``events``.

        Single-field ascending key on the nested ``event.id``, backing the
        anchor lookup in ``load_events(after_event_id=...)``. Deliberately
        NOT compound with ``team_id``: Cosmos for MongoDB rejects compound
        indexes on nested paths unless the account enables
        ``EnableUniqueCompoundNestedDocs``, and ``event.id`` is a uuid4 —
        already maximally selective for this equality lookup. Deliberately
        not unique: a unique index would turn a read-path ambiguity into a
        write-path ``DuplicateKeyError`` on ``save_event``.
        """
        info = mongo_db["events"].index_information()
        assert "events_event_id_idx" in info
        assert info["events_event_id_idx"]["key"] == [("event.id", 1)]
        assert not info["events_event_id_idx"].get("unique")

    def test_events_event_id_index_creation_is_idempotent(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """Re-running the constructor against the same database does not raise.

        ``create_index`` returns silently when an index with the same key
        spec already exists, so redeploys against a long-lived MongoDB are
        safe. After the second construction the entry is still present
        exactly once with the same key spec.
        """
        # First construction already happened via the ``mongo_store`` fixture.
        MongoEventStore(mongo_db)  # second construction — must not raise

        info = mongo_db["events"].index_information()
        assert info["events_event_id_idx"]["key"] == [("event.id", 1)]
        assert list(info.keys()).count("events_event_id_idx") == 1

    def test_index_rejection_is_logged_and_does_not_block_construction(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A backend that rejects the event.id spec must not stop the process.

        Cosmos for MongoDB rejects compound-on-nested by default and may
        reject other specs depending on account capabilities. The index is a
        performance optimization, not a correctness requirement — so
        construction logs and continues rather than raising, which would
        refuse to start the server on every replica.
        """
        real_create_index = type(mongo_db["events"]).create_index

        def _reject_event_id(self: Any, keys: Any, **kwargs: Any) -> Any:
            if keys == "event.id":
                raise OperationFailure("index not supported on this account")
            return real_create_index(self, keys, **kwargs)

        with patch.object(type(mongo_db["events"]), "create_index", _reject_event_id):
            with caplog.at_level(logging.WARNING):
                MongoEventStore(mongo_db)  # must NOT raise

        assert "events_event_id_idx" in caplog.text
        assert "collection scan" in caplog.text

    def test_existing_events_team_sequence_index_still_created(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """The pre-existing ``(team_id, sequence)`` index is untouched.

        It already backs the ``{"sequence": {"$gt": n}}`` range filter and
        needs no change for the cursor push-down.
        """
        specs = [entry["key"] for entry in mongo_db["events"].index_information().values()]
        assert [("team_id", 1), ("sequence", 1)] in specs

    # --- teams index provisioning -------------------------------------------

    def test_ensure_indexes_creates_every_teams_index(self, mongo_db: Any) -> None:
        """``ensure_indexes`` provisions all three teams-collection indexes.

        Each is single-field ascending — the specs backing ``list_teams``'
        ``user_id``, ``status`` and ``metadata`` push-downs, and the exact
        names deployment tooling probes for.

        ``teams_metadata_indexes_idx`` is the multikey one. MongoDB derives
        multikey automatically from the array-valued ``metadata_indexes``
        field, so the requested spec is a plain ``[("metadata_indexes", 1)]``
        and there is no separate index type to ask for.
        """
        ensure_indexes(mongo_db)

        info = mongo_db["teams"].index_information()
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        assert info["teams_status_idx"]["key"] == [("status", 1)]
        assert info["teams_metadata_indexes_idx"]["key"] == [("metadata_indexes", 1)]

    def test_ensure_indexes_is_idempotent(self, mongo_db: Any) -> None:
        """A second call does not raise; each index stays present exactly once.

        ``create_index`` returns silently for an identical name and key spec,
        which is what makes the routine safe from a constructor, an init
        container and a migration job alike. A same-name-different-spec
        re-request would raise instead, so this also pins that the second call
        asks for byte-identical specs.
        """
        ensure_indexes(mongo_db)
        ensure_indexes(mongo_db)  # second call — must NOT raise

        info = mongo_db["teams"].index_information()
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        assert info["teams_status_idx"]["key"] == [("status", 1)]
        assert info["teams_metadata_indexes_idx"]["key"] == [("metadata_indexes", 1)]
        assert list(info.keys()).count("teams_user_id_idx") == 1
        assert list(info.keys()).count("teams_status_idx") == 1
        assert list(info.keys()).count("teams_metadata_indexes_idx") == 1

    def test_ensure_indexes_guards_each_index_independently(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A rejected spec is logged at WARNING and skipped, never raised.

        All three names must appear: one ``try`` wrapping the whole loop would
        abandon ``teams_status_idx`` — the index the reconciler waits on — the
        moment ``teams_user_id_idx`` was refused.
        """
        real_create_index = type(mongo_db["teams"]).create_index
        rejected = {"teams_user_id_idx", "teams_status_idx", "teams_metadata_indexes_idx"}

        def _reject_teams(self: Any, keys: Any, **kwargs: Any) -> Any:
            if kwargs.get("name") in rejected:
                raise OperationFailure("index not supported on this account")
            return real_create_index(self, keys, **kwargs)

        with patch.object(type(mongo_db["teams"]), "create_index", _reject_teams):
            with caplog.at_level(logging.WARNING):
                assert ensure_indexes(mongo_db) is None

        info = mongo_db["teams"].index_information()
        for name in rejected:
            assert name in caplog.text
            assert name not in info

    def test_a_rejected_spec_does_not_skip_the_specs_after_it(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Refusing the FIRST spec still provisions the ones behind it.

        ``teams_user_id_idx`` leads ``_TEAM_INDEX_SPECS``, so rejecting only it
        is what actually proves the ``try`` sits inside the loop: with one
        ``try`` around the whole loop, ``teams_metadata_indexes_idx`` would
        never be requested and the metadata push-down would run unindexed
        forever on an account that merely dislikes one unrelated spec.
        """
        real_create_index = type(mongo_db["teams"]).create_index

        def _reject_user_id(self: Any, keys: Any, **kwargs: Any) -> Any:
            if kwargs.get("name") == "teams_user_id_idx":
                raise OperationFailure("index not supported on this account")
            return real_create_index(self, keys, **kwargs)

        with patch.object(type(mongo_db["teams"]), "create_index", _reject_user_id):
            with caplog.at_level(logging.WARNING):
                assert ensure_indexes(mongo_db) is None

        info = mongo_db["teams"].index_information()
        assert "teams_user_id_idx" in caplog.text
        assert "teams_user_id_idx" not in info
        assert info["teams_status_idx"]["key"] == [("status", 1)]
        assert info["teams_metadata_indexes_idx"]["key"] == [("metadata_indexes", 1)]

    def test_metadata_index_rejection_is_logged_and_construction_still_succeeds(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rejecting only the metadata spec warns by name and does not raise.

        Cosmos for MongoDB supports single-field indexes on array fields, but
        account capabilities vary. The index is an optimization — a store that
        refused to construct because one index was unavailable would turn a
        capability gap into a server that will not boot, while ``list_teams``
        would have returned correct results the whole time.

        The environment is normalised first: this test asserts on a warning
        that is only reachable when the build is actually attempted, so an
        ambient ``MONGO_TEAM_AUTO_INDEX=0`` would skip provisioning entirely
        and fail it for a reason that has nothing to do with the guard.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)
        real_create_index = type(mongo_db["teams"]).create_index

        def _reject_metadata(self: Any, keys: Any, **kwargs: Any) -> Any:
            if kwargs.get("name") == "teams_metadata_indexes_idx":
                raise OperationFailure("index not supported on this account")
            return real_create_index(self, keys, **kwargs)

        with patch.object(type(mongo_db["teams"]), "create_index", _reject_metadata):
            with caplog.at_level(logging.WARNING):
                store = MongoEventStore(mongo_db)  # must NOT raise

        assert "teams_metadata_indexes_idx" in caplog.text
        assert "collection scan" in caplog.text
        info = mongo_db["teams"].index_information()
        assert "teams_metadata_indexes_idx" not in info
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        # Still usable: the filter runs, it just runs unindexed.
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        store.save_team(acme)
        store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="contoso")))
        assert [p.team_id for p in store.list_teams(metadata={"tenant": ["acme"]})] == [
            acme.team_id
        ]

    def test_teams_index_rejection_does_not_block_construction(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend that rejects the teams specs must not stop the process.

        The wrapper refuses only the three teams indexes and delegates every
        other key: a patch that rejected everything would take construction
        down on the ungated ``(team_id, sequence)`` index and prove nothing
        about the teams guard. ``mongomock`` shares one ``Collection`` class
        across collections, so the patch is seen by all of them.

        The environment is normalised first for the same reason as the
        metadata-only rejection test above: an ambient opt-out would skip the
        build and leave nothing to warn about.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)
        real_create_index = type(mongo_db["teams"]).create_index
        rejected = {"teams_user_id_idx", "teams_status_idx", "teams_metadata_indexes_idx"}

        def _reject_teams(self: Any, keys: Any, **kwargs: Any) -> Any:
            if kwargs.get("name") in rejected:
                raise OperationFailure("index not supported on this account")
            return real_create_index(self, keys, **kwargs)

        with patch.object(type(mongo_db["teams"]), "create_index", _reject_teams):
            with caplog.at_level(logging.WARNING):
                MongoEventStore(mongo_db)  # must NOT raise

        for name in rejected:
            assert name in caplog.text
        assert "collection scan" in caplog.text

    # --- teams index opt-out -------------------------------------------------

    def test_teams_indexes_created_by_default(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No argument and no env var: construction creates every teams index.

        Parity with how ``teams_user_id_idx`` behaved before the opt-out
        existed — small deployments get the indexes for free.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)

        MongoEventStore(mongo_db)

        info = mongo_db["teams"].index_information()
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        assert info["teams_status_idx"]["key"] == [("status", 1)]
        assert info["teams_metadata_indexes_idx"]["key"] == [("metadata_indexes", 1)]

    def test_auto_create_indexes_false_creates_no_teams_index(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``auto_create_indexes=False`` suppresses every teams index.

        Including ``teams_user_id_idx``, which was unconditional before the
        opt-out existed: one provisioning path, three indexes — which is why
        the opt-out is paired with
        ``python -m akgentic.team.scripts.init_mongo``. Absence is asserted by
        name, never as ``index_information() == {}``: mongomock adds an
        ``_id_`` entry as soon as anything touches the collection.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)

        MongoEventStore(mongo_db, auto_create_indexes=False)

        info = mongo_db["teams"].index_information()
        assert "teams_user_id_idx" not in info
        assert "teams_status_idx" not in info
        assert "teams_metadata_indexes_idx" not in info
        _assert_ungated_indexes_present(mongo_db)

    @pytest.mark.parametrize("raw", ["0", "false", "no", "FALSE", "No"])
    def test_env_disabling_value_creates_no_teams_index(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """``MONGO_TEAM_AUTO_INDEX`` in {0, false, no} disables, case-insensitively.

        This is the production safety valve: an enterprise teams collection of
        ~100 GB cannot absorb a foreground index build at boot.
        """
        monkeypatch.setenv("MONGO_TEAM_AUTO_INDEX", raw)

        MongoEventStore(mongo_db)

        info = mongo_db["teams"].index_information()
        assert "teams_user_id_idx" not in info
        assert "teams_status_idx" not in info
        assert "teams_metadata_indexes_idx" not in info
        _assert_ungated_indexes_present(mongo_db)

    @pytest.mark.parametrize("raw", ["", "1", "true", "yes", "maybe"])
    def test_env_non_disabling_value_leaves_the_default_on(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """Any other value — empty included — leaves the boot-time build on.

        The variable is an explicit opt-out, not a tri-state: a typo must fail
        towards the historical behaviour rather than silently dropping indexes.
        """
        monkeypatch.setenv("MONGO_TEAM_AUTO_INDEX", raw)

        MongoEventStore(mongo_db)

        info = mongo_db["teams"].index_information()
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        assert info["teams_status_idx"]["key"] == [("status", 1)]
        assert info["teams_metadata_indexes_idx"]["key"] == [("metadata_indexes", 1)]

    def test_explicit_argument_beats_the_disabling_environment(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``auto_create_indexes=True`` overrides ``MONGO_TEAM_AUTO_INDEX=0``.

        A caller that knows its collection is small stays in control of its own
        store even under a deployment-wide opt-out. Fails under an env-first or
        truthiness-based resolution, which is the point.
        """
        monkeypatch.setenv("MONGO_TEAM_AUTO_INDEX", "0")

        MongoEventStore(mongo_db, auto_create_indexes=True)

        info = mongo_db["teams"].index_information()
        assert info["teams_user_id_idx"]["key"] == [("user_id", 1)]
        assert info["teams_status_idx"]["key"] == [("status", 1)]
        assert info["teams_metadata_indexes_idx"]["key"] == [("metadata_indexes", 1)]

    def test_auto_create_indexes_is_keyword_only(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flag cannot be passed positionally.

        Keyword-only so no positional caller can drift into it — a second
        positional argument is a ``TypeError``, not a silent opt-out.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)

        with pytest.raises(TypeError):
            MongoEventStore(mongo_db, False)  # type: ignore[misc]

    def test_list_teams_status_filter_works_without_the_status_index(
        self, mongo_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The status filter is correct against a collection with no status index.

        Correctness must never depend on the index being present — a store that
        failed closed while an operator provisioned out of band would turn a
        performance opt-out into an outage.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)
        store = MongoEventStore(mongo_db, auto_create_indexes=False)
        running = make_process(status=TeamStatus.RUNNING)
        store.save_team(running)
        store.save_team(make_process(status=TeamStatus.STOPPED))

        assert "teams_status_idx" not in mongo_db["teams"].index_information()
        assert {p.team_id for p in store.list_teams(status=TeamStatus.RUNNING)} == {
            running.team_id
        }

    # --- metadata push-down --------------------------------------------------

    def test_list_teams_pushes_metadata_into_the_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """``metadata`` reaches MongoDB as an anchored ``$regex``, not a Python pass.

        An in-memory filter returns the identical list, so the assertion bites
        on the query dict handed to ``find``. The expected pattern is written
        as a literal rather than built through the helper, so a change to the
        separator or to the key/value ordering fails here instead of passing
        symmetrically on both sides.
        """
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        mongo_store.save_team(acme)
        mongo_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="contoso")))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(metadata={"tenant": ["acme"]})

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {
            "$and": [{"$or": [{"metadata_indexes": {"$regex": "^tenant\\|acme"}}]}]
        }
        assert [p.team_id for p in result] == [acme.team_id]

    def test_the_metadata_regex_is_anchored_and_case_sensitive(
        self, mongo_store: MongoEventStore
    ) -> None:
        """No ``$options`` and no inline ``(?i)`` — the seek depends on it.

        A case-insensitive regex uses no index at all in MongoDB, which is the
        entire reason ``make_index_entry`` casefolds on write. Asserted on the
        constructed query rather than on rows, because ``$options: "i"`` returns
        the SAME rows here and would leave every result-level assertion green
        while quietly turning each query into a collection scan.
        """
        mongo_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="AzeFR")))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            mongo_store.list_teams(metadata={"tenant": ["AZE"], "case_ref": ["c"]})

        groups = spy_find.call_args.args[0]["$and"]
        assert len(groups) == 2
        arms = [arm for group in groups for arm in group["$or"]]
        assert len(arms) == 2
        for arm in arms:
            term = arm["metadata_indexes"]
            assert set(term) == {"$regex"}, f"no option key may ride along: {term}"
            assert term["$regex"].startswith("^")
            assert "(?i)" not in term["$regex"]

    def test_the_metadata_regex_matches_the_casefolded_index(
        self, mongo_store: MongoEventStore
    ) -> None:
        """The term folds on the query side too, so the anchored pattern still hits."""
        aze = make_indexed_process(AcmeTeamMetadata(tenant="AzeFR"))
        mongo_store.save_team(aze)
        mongo_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="contoso")))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(metadata={"tenant": ["AzE"]})

        assert spy_find.call_args.args[0] == {
            "$and": [{"$or": [{"metadata_indexes": {"$regex": "^tenant\\|aze"}}]}]
        }
        assert [p.team_id for p in result] == [aze.team_id]

    def test_terms_nest_as_or_inside_and(self, mongo_store: MongoEventStore) -> None:
        """One ``$or`` per KEY, those ``$and``-ed — the nesting IS the rule.

        Terms for one key OR; distinct keys AND. Flattening the two levels into
        a single ``$and`` of arms would quietly turn the tenant disjunction back
        into a conjunction and return nothing for two distinct tenants; putting
        both keys in one ``$or`` would widen instead. The nesting, not just the
        arm count, is what this pins — and a single dict cannot carry two
        ``metadata_indexes`` keys, which is why the grouping has to be explicit.
        """
        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            mongo_store.list_teams(metadata={"tenant": ["ac", "acm"], "case_ref": ["C-"]})

        assert spy_find.call_args.args[0] == {
            "$and": [
                {
                    "$or": [
                        {"metadata_indexes": {"$regex": "^tenant\\|ac"}},
                        {"metadata_indexes": {"$regex": "^tenant\\|acm"}},
                    ]
                },
                # ``re.escape`` escapes ``-`` too — it is special inside a
                # character class, so it is escaped unconditionally.
                {"$or": [{"metadata_indexes": {"$regex": "^case_ref\\|c\\-"}}]},
            ]
        }

    def test_an_emptied_key_contributes_no_arm_beside_a_real_one(
        self, mongo_store: MongoEventStore
    ) -> None:
        """A key whose terms all render away drops out — never as an empty ``$or``.

        This is the twin of the empty-``$and`` hazard and it is worse: MongoDB
        rejects ``$or: []`` outright, so the failure is a query error rather
        than a silently wrong result set. The whole-mapping-empty case is
        covered above; this is the one where a REAL key sits beside the emptied
        one, so the metadata clause survives and only the dead group must go.
        """
        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            mongo_store.list_teams(metadata={"tenant": [], "case_ref": ["C-1"], "other": [""]})

        assert spy_find.call_args.args[0] == {
            "$and": [{"$or": [{"metadata_indexes": {"$regex": "^case_ref\\|c\\-1"}}]}]
        }

    def test_list_teams_pushes_user_id_and_metadata_into_one_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """Both terms travel down in a single ``find`` call, ANDed.

        ``user_id`` is a trust boundary: it is applied server-side regardless
        of what metadata is asked for, never replaced by the more specific
        term. Two ``find`` calls intersected afterwards, or a ``user_id``
        dropped because metadata "already narrows enough", would both make a
        metadata filter a way to reach another user's teams.
        """
        mine = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u1")
        mongo_store.save_team(mine)
        mongo_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u2"))
        mongo_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="contoso"), user_id="u1")
        )

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(user_id="u1", metadata={"tenant": ["acme"]})

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {
            "user_id": "u1",
            "$and": [{"$or": [{"metadata_indexes": {"$regex": "^tenant\\|acme"}}]}],
        }
        assert [p.team_id for p in result] == [mine.team_id]

    def test_list_teams_pushes_all_three_filters_into_one_find_filter(
        self, mongo_store: MongoEventStore
    ) -> None:
        """``user_id``, ``status`` and ``metadata`` coexist in one query dict.

        The metadata term is additive: it must not displace the two filters
        that were already pushed down before it existed.
        """
        wanted = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.RUNNING
        )
        mongo_store.save_team(wanted)
        mongo_store.save_team(
            make_indexed_process(
                AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.STOPPED
            )
        )

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(
                user_id="u1", status=TeamStatus.RUNNING, metadata={"tenant": ["acme"]}
            )

        spy_find.assert_called_once()
        assert spy_find.call_args.args[0] == {
            "user_id": "u1",
            "status": "running",
            "$and": [{"$or": [{"metadata_indexes": {"$regex": "^tenant\\|acme"}}]}],
        }
        assert [p.team_id for p in result] == [wanted.team_id]

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            {},
            {"tenant": []},
            {"tenant": [""]},
            {"tenant": ["", ""]},
            {"tenant": [""], "case_ref": []},
        ],
        ids=[
            "none",
            "empty-dict",
            "empty-term-list",
            "one-blank-term",
            "two-blank-terms",
            "blank-term-and-empty-list",
        ],
    )
    def test_list_teams_no_metadata_filter_carries_no_metadata_key(
        self, mongo_store: MongoEventStore, metadata: dict[str, list[str]] | None
    ) -> None:
        """Every "no effective term" spelling leaves the query dict clean.

        None of ``metadata_indexes``, ``$and`` or ``$or`` may appear. All three
        matter and for different reasons: ``$and: []`` AND ``$or: []`` are both
        OperationFailure in MongoDB (and in mongomock), and the ``$all: []``
        shape this replaced matched ZERO documents — so "no filter" silently
        became "no results". Asserting the keys are ABSENT is what catches it;
        a result-level assertion cannot tell "no term" from "a term that
        happens to match everything", and over an empty store it cannot see
        the regression at all.

        ``{"tenant": []}`` and ``{"tenant": [""]}`` are the ones the outer
        truthiness gate misses: the mapping itself is truthy.
        """
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        plain = make_indexed_process(None)
        mongo_store.save_team(acme)
        mongo_store.save_team(plain)

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(metadata=metadata)

        spy_find.assert_called_once()
        query = spy_find.call_args.args[0]
        assert "metadata_indexes" not in query
        assert "$and" not in query, f"an empty conjunction reached the query: {query}"
        assert "$or" not in query, f"an empty disjunction reached the query: {query}"
        assert {p.team_id for p in result} == {acme.team_id, plain.team_id}

    def test_list_teams_does_not_filter_metadata_in_python(
        self, mongo_store: MongoEventStore
    ) -> None:
        """No post-hydration metadata comparison survives in the method body.

        ``find`` is stubbed to hand back BOTH documents whatever filter the
        store asks for. A surviving Python-side ``issubset`` would drop the
        contoso team while leaving every result-level assertion elsewhere
        green — which is exactly the false pin this test exists to catch.
        """
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        contoso = make_indexed_process(AcmeTeamMetadata(tenant="contoso"))
        mongo_store.save_team(acme)
        mongo_store.save_team(contoso)

        all_docs = list(mongo_store._teams.find({}))
        with patch.object(mongo_store._teams, "find", return_value=iter(all_docs)):
            result = mongo_store.list_teams(metadata={"tenant": ["acme"]})

        assert {p.team_id for p in result} == {acme.team_id, contoso.team_id}

    def test_list_teams_metadata_query_entries_carry_the_shared_escaping(
        self, mongo_store: MongoEventStore
    ) -> None:
        """A value holding a literal ``|`` is escaped in the query, as on write.

        Both sides go through ``make_index_entry``, so ``acme|corp`` becomes
        ``tenant|acme\\|corp`` and matches the entry the write path derived. A
        hand-rolled ``f"{k}|{v}"`` would build ``tenant|acme|corp``, match
        nothing, and be invisible to any test that only used pipe-free values.

        Two escaping layers meet here: the rendered prefix already carries the
        backslash the separator escaping planted, and ``re.escape`` must carry
        it into the pattern as a LITERAL backslash rather than let it pair up
        with the following ``|`` into a regex alternation escape.
        """
        piped = make_indexed_process(AcmeTeamMetadata(tenant="acme|corp"))
        mongo_store.save_team(piped)
        mongo_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))

        with patch.object(
            mongo_store._teams, "find", wraps=mongo_store._teams.find
        ) as spy_find:
            result = mongo_store.list_teams(metadata={"tenant": ["acme|corp"]})

        spy_find.assert_called_once()
        # Written raw so the backslashes are countable. The rendered prefix is
        # `tenant|acme\|corp`; `re.escape` turns each `|` into `\|` and the
        # separator escaping's own `\` into `\\`, giving `\\\|` in the middle.
        assert spy_find.call_args.args[0] == {
            "$and": [{"$or": [{"metadata_indexes": {"$regex": r"^tenant\|acme\\\|corp"}}]}]
        }
        assert [p.team_id for p in result] == [piped.team_id]

    def test_list_teams_does_not_branch_on_whether_the_index_exists(
        self, mongo_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The read path is identical with and without ``ensure_indexes()``.

        Two stores over the same fixtures, one indexed and one built with the
        opt-out, issue the **same query dict** and return the same rows. What
        this pins is that ``auto_create_indexes`` reaches provisioning only:
        neither the filter the store builds nor the hydration that follows
        consults ``index_information()`` or degrades itself when an index is
        missing. A store that fell back to a different query shape — or failed
        closed — while an index was provisioned out of band would turn a
        performance valve into an outage.

        **This does NOT demonstrate real index-independence, and must not be
        read as evidence for it.** ``mongomock`` never consults an index when
        answering a query: ``_iter_documents`` linearly scans every document
        and applies the filter to each, so dropping an index cannot change its
        results by construction and the row comparison below would hold even
        for an implementation that genuinely depended on the index. The real
        property — that a live MongoDB returns identical rows unindexed and
        merely pays a collection scan for them — is not verifiable in this
        package, which has no real-server test harness. It rests on MongoDB's
        own guarantee that an index is an access path and never a filter.
        """
        monkeypatch.delenv("MONGO_TEAM_AUTO_INDEX", raising=False)
        indexed_db = mongo_client["indexed"]
        bare_db = mongo_client["bare"]
        indexed = MongoEventStore(indexed_db)
        bare = MongoEventStore(bare_db, auto_create_indexes=False)
        fixtures = _metadata_fixtures()
        for process in fixtures:
            indexed.save_team(process)
            bare.save_team(process)

        assert "teams_metadata_indexes_idx" not in bare_db["teams"].index_information()
        assert indexed_db["teams"].index_information()["teams_metadata_indexes_idx"][
            "key"
        ] == [("metadata_indexes", 1)]

        # The load-bearing half: the query is built the same way either side of
        # the opt-out. This is a construction-level assertion, which is the
        # class of claim mongomock can actually support.
        filters = {"tenant": ["acme"]}
        with patch.object(bare._teams, "find", wraps=bare._teams.find) as bare_find:
            from_bare = {p.team_id for p in bare.list_teams(metadata=filters)}
        with patch.object(
            indexed._teams, "find", wraps=indexed._teams.find
        ) as indexed_find:
            from_indexed = {p.team_id for p in indexed.list_teams(metadata=filters)}

        bare_find.assert_called_once()
        indexed_find.assert_called_once()
        assert bare_find.call_args.args[0] == indexed_find.call_args.args[0]
        assert bare_find.call_args.args[0] == {
            "$and": [{"$or": [{"metadata_indexes": {"$regex": "^tenant\\|acme"}}]}]
        }
        assert from_bare == from_indexed
        # Pinned absolutely too, so two equally broken stores cannot pass by
        # agreeing on the empty set.
        assert from_bare == {fixtures[i].team_id for i in (0, 1, 2)}
        # Pinned absolutely too, so two equally broken stores cannot pass by
        # agreeing on the empty set.
        assert from_bare == {fixtures[i].team_id for i in (0, 1, 2)}

    # --- cross-backend parity ------------------------------------------------

    @pytest.mark.parametrize(
        "user_id,metadata,expected_indices",
        [
            (None, {"tenant": ["acme"]}, {0, 1, 2}),
            (None, {"tenant": ["acme"], "case_ref": ["C-1"]}, {0}),
            (None, {"tenant": ["nobody-has-this"]}, set()),
            (None, {"case_ref": ["C|1234"]}, {1}),
            (None, None, {0, 1, 2, 3, 4, 5}),
            ("u1", {"tenant": ["acme"]}, {0, 1}),
        ],
        ids=["one-key", "two-keys", "no-such-value", "literal-pipe", "no-filter", "with-user-id"],
    )
    def test_mongo_and_yaml_agree_for_the_same_fixtures(
        self,
        mongo_store: MongoEventStore,
        tmp_path: Path,
        user_id: str | None,
        metadata: dict[str, list[str]] | None,
        expected_indices: set[int],
    ) -> None:
        """The two backends return the same teams for the same filters.

        Compared as sets of ``team_id`` — neither backend promises an ordering
        — and both are additionally pinned against an absolute expectation, so
        the two agreeing on a wrong answer (or on nothing, for the
        no-such-value case) is not a pass.
        """
        from akgentic.team.repositories.yaml import YamlEventStore

        yaml_store = YamlEventStore(tmp_path)
        fixtures = _metadata_fixtures()
        for process in fixtures:
            mongo_store.save_team(process)
            yaml_store.save_team(process)

        expected = {fixtures[i].team_id for i in expected_indices}
        from_mongo = {p.team_id for p in mongo_store.list_teams(user_id=user_id, metadata=metadata)}
        from_yaml = {p.team_id for p in yaml_store.list_teams(user_id=user_id, metadata=metadata)}

        assert from_mongo == expected
        assert from_yaml == expected

    # --- Import guards ------------------------------------------------------

    def test_import_succeeds_when_pymongo_installed(self) -> None:
        """``MongoEventStore`` is importable when ``pymongo`` is available."""
        from akgentic.team.repositories.mongo import MongoEventStore as Imported

        assert Imported is MongoEventStore

    def test_import_guard_when_pymongo_missing(self) -> None:
        """Conditional import fails gracefully when ``pymongo`` is unavailable."""
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


class TestMongoAgentCardStore:
    """Mongo-only invariants of the content-addressed card store."""

    def test_cards_live_in_their_own_collection(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        card = _card_fixture()
        mongo_store.save_agent_cards([card])

        docs = list(mongo_db[AGENT_CARDS_COLLECTION].find({}))
        assert len(docs) == 1
        assert docs[0]["card_hash"] == hash_agent_card(card)

    def test_the_collection_carries_no_team_id(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """The store is shared. A ``team_id`` here would be the design gone."""
        mongo_store.save_agent_cards([_card_fixture()])

        doc = mongo_db[AGENT_CARDS_COLLECTION].find_one({})
        assert doc is not None
        assert "team_id" not in doc

    def test_the_card_hash_index_is_created_and_unique(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """Unique deliberately: two documents at one hash is two answers to
        "what are the bytes this hash names", not merely a slow query.

        ``save_agent_cards`` upserts on the same key, so this store's own write
        path cannot fork a hash without it — the index is what stops a writer
        that bypasses ``save_agent_cards``, and why the creation is guarded
        rather than allowed to refuse construction.
        """
        info = mongo_db[AGENT_CARDS_COLLECTION].index_information()
        assert "agent_cards_card_hash_idx" in info
        assert info["agent_cards_card_hash_idx"]["key"] == [("card_hash", 1)]
        assert info["agent_cards_card_hash_idx"].get("unique") is True

    def test_the_index_creation_is_idempotent(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        MongoEventStore(mongo_db)  # second construction — must not raise

        info = mongo_db[AGENT_CARDS_COLLECTION].index_information()
        assert list(info.keys()).count("agent_cards_card_hash_idx") == 1

    def test_a_card_index_rejection_does_not_block_construction(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The upsert already guarantees one document per hash without it."""
        real_create_index = type(mongo_db["x"]).create_index

        def selective_failure(self: Any, keys: Any, **kwargs: Any) -> Any:
            if kwargs.get("name") == "agent_cards_card_hash_idx":
                raise OperationFailure("index rejected")
            return real_create_index(self, keys, **kwargs)

        with (
            patch.object(type(mongo_db["x"]), "create_index", selective_failure),
            caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.mongo"),
        ):
            store = MongoEventStore(mongo_db)

        assert store is not None
        assert [r for r in caplog.records if "agent_cards_card_hash_idx" in r.getMessage()]

    def test_a_re_save_upserts_rather_than_duplicating(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """``insert_one`` would raise ``DuplicateKeyError`` against the index."""
        card = _card_fixture()
        mongo_store.save_agent_cards([card])
        mongo_store.save_agent_cards([card])

        assert mongo_db[AGENT_CARDS_COLLECTION].count_documents({}) == 1

    def test_delete_team_does_not_touch_the_collection(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        """FR13, asserted on the collection rather than on a later read."""
        process = make_process()
        mongo_store.save_agent_cards([_card_fixture()])
        mongo_store.save_team(process)

        mongo_store.delete_team(process.team_id)

        assert mongo_db["teams"].count_documents({}) == 0
        assert mongo_db[AGENT_CARDS_COLLECTION].count_documents({}) == 1

    def test_saving_no_cards_writes_nothing(
        self, mongo_store: MongoEventStore, mongo_db: Any
    ) -> None:
        mongo_store.save_agent_cards([])
        assert mongo_db[AGENT_CARDS_COLLECTION].count_documents({}) == 0

    def test_a_corrupted_card_document_is_skipped_not_raised(
        self,
        mongo_store: MongoEventStore,
        mongo_db: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Mongo's analogue of the YAML corrupted-file skip."""
        mongo_db[AGENT_CARDS_COLLECTION].insert_one(
            {"card_hash": "b" * 64, "card": {"not": "a card"}}
        )

        with caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.mongo"):
            loaded = mongo_store.load_agent_cards(["b" * 64])

        assert loaded == {}
        assert [r for r in caplog.records if "corrupted agent card" in r.getMessage()]
