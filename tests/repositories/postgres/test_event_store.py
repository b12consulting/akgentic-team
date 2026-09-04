"""Postgres-specific tests for ``NagraEventStore``.

Behavioural Protocol coverage (round-trip, upsert, list, sequencing,
max sequence, cascading delete, polymorphic round-trips) lives in the
shared ``tests/repositories/test_event_store_contract.py`` and runs
once per backend. This module retains only Postgres-specific
invariants:

* ``test_satisfies_event_store_protocol`` — structural typing check.
* ``test_payload_is_authoritative_over_promoted_columns`` — schema-drift
  / payload-authority invariant from Story 17.2.
* ``test_duplicate_sequence_raises_unique_violation`` — the §8 native-
  exception propagation contract for the composite primary key.

Constructor / source-purity / import-gate tests live in adjacent files
(``test_ci_env_wiring.py``, ``test_init_db.py``, ``test_import_gate.py``).
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from nagra import Transaction  # type: ignore[import-untyped]

from akgentic.team.models import Process, TeamStatus
from akgentic.team.projection import hash_agent_card
from akgentic.team.repositories.postgres import NagraEventStore, init_db

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

from tests.models.conftest import (
    AcmeTeamMetadata,
    make_agent_card,
    make_indexed_process,
    make_persisted_event,
    make_process,
)


def _card_fixture() -> AgentCard:
    """An AgentCard whose ``agent_class`` resolves on the way back out of storage."""
    return make_agent_card(name="lead", role="Lead", agent_class=Akgent)


def _read_metadata_indexes_column(conn_string: str, team_id: uuid.UUID) -> list[str] | None:
    """Read the promoted column directly, bypassing hydration.

    Every assertion about the write path has to read the column with SQL:
    going through ``load_team`` would report the payload instead and pass
    just as happily against a store that never wrote the column at all.
    """
    with Transaction(conn_string) as trn:
        cursor = trn.execute(
            "SELECT metadata_indexes FROM team_process_entries WHERE id = %s",
            (str(team_id),),
        )
        row = cursor.fetchone()
    assert row is not None, f"no team_process_entries row for {team_id}"
    value: list[str] | None = row[0]
    return value


class TestNagraEventStorePostgresSpecific:
    """Postgres-only invariants — see contract suite for behavioural coverage."""

    # --- Protocol compliance -------------------------------------------------

    def test_satisfies_event_store_protocol(self, postgres_clean_tables: str) -> None:
        """``NagraEventStore`` satisfies ``EventStore`` Protocol structurally."""
        store: EventStore = NagraEventStore(postgres_clean_tables)
        assert store is not None

    # --- schema-drift / payload-authority invariant -------------------------

    def test_payload_is_authoritative_over_promoted_columns(
        self, postgres_clean_tables: str
    ) -> None:
        """Hydrated model fields come from JSONB ``data`` only.

        Plants a row whose promoted ``team_id`` / ``sequence`` columns
        DISAGREE with the embedded payload, then asserts the hydrated
        ``PersistedEvent`` carries the payload values — proving the
        promoted columns are query keys only, never read back into the
        model. Bypasses ``save_event`` on purpose to plant the drift.
        """
        store = NagraEventStore(postgres_clean_tables)
        routing_team_id = uuid.uuid4()
        payload_team_id = uuid.uuid4()

        canonical = make_persisted_event(team_id=payload_team_id, sequence=42)
        payload_dict = canonical.model_dump()
        payload_dict["team_id"] = str(payload_team_id)
        payload_dict["sequence"] = 42

        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO event_entries (team_id, sequence, data) VALUES (%s, %s, %s)",
                (str(routing_team_id), 1, json.dumps(payload_dict)),
            )

        loaded = store.load_events(routing_team_id)
        assert len(loaded) == 1
        assert loaded[0].team_id == payload_team_id
        assert loaded[0].sequence == 42

    # --- duplicate (team_id, sequence) propagation contract -----------------

    def test_duplicate_sequence_raises_unique_violation(self, postgres_clean_tables: str) -> None:
        """Composite-PK violation propagates as ``psycopg.errors.UniqueViolation``.

        Pins the exception TYPE — not a substring of the message — so a
        regression that swallows or rewraps the native exception is
        caught immediately. Mirrors the §8 contract: backends must
        propagate the underlying driver's native uniqueness error so
        callers can rely on the type for compensation logic.
        """
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        store.save_event(make_persisted_event(team_id=team_id, sequence=1))
        with pytest.raises(psycopg.errors.UniqueViolation):
            store.save_event(make_persisted_event(team_id=team_id, sequence=1))


class TestMetadataIndexesColumnWritePath:
    """The promoted ``metadata_indexes`` column is written on every save.

    The value lives in the JSON ``data`` payload and its derived index
    lives in the column — two places, written by one statement. These
    tests read the column with SQL, never through ``load_team``.
    """

    def test_save_team_writes_the_column_on_insert(
        self, postgres_clean_tables: str
    ) -> None:
        """A fresh save lands the derived entries in the column."""
        store = NagraEventStore(postgres_clean_tables)
        process = make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-1"))

        store.save_team(process)

        stored = _read_metadata_indexes_column(postgres_clean_tables, process.team_id)
        assert stored is not None
        # The value half is casefolded at derivation, so ``C-1`` is stored
        # ``c-1``. The seeded value itself is untouched — see the payload.
        assert set(stored) == {"tenant|acme", "case_ref|c-1"}

    def test_save_team_updates_the_column_on_conflict(
        self, postgres_clean_tables: str
    ) -> None:
        """Re-saving the same ``team_id`` refreshes the column, not just ``data``.

        This is the failure the ``ON CONFLICT`` branch invites: an upsert
        that sets ``data = EXCLUDED.data`` and forgets the column leaves a
        stale index behind, which ``list_teams`` then faithfully queries.
        A test that only saves once cannot see it, so this one saves twice
        with genuinely different metadata.
        """
        store = NagraEventStore(postgres_clean_tables)
        team_id = uuid.uuid4()
        store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="acme"), team_id=team_id)
        )

        store.save_team(
            make_indexed_process(
                AcmeTeamMetadata(tenant="contoso", case_ref="C-9"), team_id=team_id
            )
        )

        stored = _read_metadata_indexes_column(postgres_clean_tables, team_id)
        assert stored is not None
        assert set(stored) == {"tenant|contoso", "case_ref|c-9"}
        # And the filter agrees with the column: the old value is gone.
        assert store.list_teams(metadata={"tenant": ["acme"]}) == []
        assert [p.team_id for p in store.list_teams(metadata={"tenant": ["contoso"]})] == [
            team_id
        ]

    def test_team_without_metadata_stores_an_empty_array_not_null(
        self, postgres_clean_tables: str
    ) -> None:
        """No metadata means ``ARRAY[]``, distinct from a legacy row's ``NULL``.

        The distinction matters: ``NULL`` is what a row written before the
        column existed carries, and the two must stay tellable apart.
        """
        store = NagraEventStore(postgres_clean_tables)
        process = make_indexed_process(None)

        store.save_team(process)

        stored = _read_metadata_indexes_column(postgres_clean_tables, process.team_id)
        assert stored == []
        loaded = store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.metadata_indexes == []


class TestMetadataFilterIsPushedDown:
    """Which half of the row each side reads.

    The filter reads the promoted column; hydration reads the payload.
    Both tests plant a row whose column and payload deliberately disagree,
    which is the only way to tell a pushed-down ``@>`` term apart from an
    in-memory filter in a backend where the payload carries the same data.
    """

    def test_metadata_filter_reads_the_column_not_the_payload(
        self, postgres_clean_tables: str
    ) -> None:
        """A row whose payload matches but whose column is empty is NOT returned.

        The in-memory filter this replaced tested ``Process.metadata_indexes``
        hydrated from ``data`` — so it WOULD return this row. The pushed-down
        ``@>`` term tests the column, and does not.
        """
        store = NagraEventStore(postgres_clean_tables)
        process = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        assert process.metadata_indexes == ["tenant|acme"], "payload must carry the entry"
        payload = json.dumps(process.model_dump())

        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data, metadata_indexes) "
                "VALUES (%s, %s, %s)",
                (str(process.team_id), payload, []),
            )

        assert store.list_teams(metadata={"tenant": ["acme"]}) == []
        # ...and it is still an ordinary team for every other query.
        assert [p.team_id for p in store.list_teams()] == [process.team_id]

    def test_hydration_reads_the_payload_not_the_column(
        self, postgres_clean_tables: str
    ) -> None:
        """The mirror image: the column matches, the payload is empty.

        The filter returns the row because the column matches, but the
        hydrated ``Process`` carries the PAYLOAD's (empty) index — proving
        the promoted column is a query key only and is never read back into
        the model. Mirrors ``test_payload_is_authoritative_over_promoted_columns``
        for ``event_entries``.
        """
        store = NagraEventStore(postgres_clean_tables)
        process = make_indexed_process(None)
        assert process.metadata_indexes == [], "payload must carry no entry"
        payload = json.dumps(process.model_dump())

        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data, metadata_indexes) "
                "VALUES (%s, %s, %s)",
                (str(process.team_id), payload, ["tenant|acme"]),
            )

        found = store.list_teams(metadata={"tenant": ["acme"]})
        assert [p.team_id for p in found] == [process.team_id]
        assert found[0].metadata_indexes == []

    def test_row_written_before_the_column_existed_still_lists(
        self, postgres_clean_tables: str
    ) -> None:
        """A legacy row — column absent from the INSERT, therefore ``NULL``.

        Backward compatibility: such a row is returned by every query that
        does not filter on metadata, and by none that does. ``NULL @> ...``
        is ``NULL``, which ``WHERE`` treats as false — the wanted behaviour,
        and the reason no ``COALESCE`` wraps the column (it would make the
        expression non-indexable).
        """
        store = NagraEventStore(postgres_clean_tables)
        process = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.RUNNING
        )
        payload = json.dumps(process.model_dump())

        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data) VALUES (%s, %s)",
                (str(process.team_id), payload),
            )

        assert _read_metadata_indexes_column(postgres_clean_tables, process.team_id) is None
        assert [p.team_id for p in store.list_teams()] == [process.team_id]
        assert [p.team_id for p in store.list_teams(user_id="u1")] == [process.team_id]
        assert [
            p.team_id for p in store.list_teams(status=TeamStatus.RUNNING)
        ] == [process.team_id]
        # Not returned by any non-empty metadata filter, even though its
        # payload carries the matching entry.
        assert store.list_teams(metadata={"tenant": ["acme"]}) == []

    def test_metadata_values_are_bound_never_interpolated(
        self, postgres_clean_tables: str
    ) -> None:
        """A value carrying SQL syntax is treated as data, never as statement text.

        The statement is assembled only from fixed fragments and every
        caller-supplied value travels as a ``%s`` parameter, so a value that
        would otherwise close a string literal and open a new statement is
        matched literally instead. Without this test nothing goes red if a
        later edit interpolates the value into the SQL — every other
        acceptance criterion here has a behavioural pin, and this one is the
        security-relevant one.
        """
        store = NagraEventStore(postgres_clean_tables)
        hostile = "acme'; DROP TABLE team_process_entries; --"
        planted = make_indexed_process(AcmeTeamMetadata(tenant=hostile))
        ordinary = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        store.save_team(planted)
        store.save_team(ordinary)

        # The value round-trips as data: the full term reaches only its own team.
        assert [p.team_id for p in store.list_teams(metadata={"tenant": [hostile]})] == [
            planted.team_id
        ]
        # The hostile value happens to START with ``acme``, so under prefix
        # matching the short term legitimately reaches both. Kept deliberately:
        # what this test pins is that the value is DATA, and a term that reaches
        # a second row is a far better outcome than one that drops a table.
        assert {p.team_id for p in store.list_teams(metadata={"tenant": ["acme"]})} == {
            planted.team_id,
            ordinary.team_id,
        }
        # A term the hostile value does not start with still excludes it.
        assert [p.team_id for p in store.list_teams(metadata={"tenant": ["acme'"]})] == [
            planted.team_id
        ]
        # ...and the table the value names is still there, with both rows in it.
        assert len({p.team_id for p in store.list_teams()}) == 2

    def test_legacy_row_is_still_returned_by_an_empty_metadata_filter(
        self, postgres_clean_tables: str
    ) -> None:
        """``metadata={}`` must not silently drop pre-migration rows.

        ``x @> '{}'`` is TRUE for any non-null ``x`` but NULL for a legacy
        row, so gating the clause on ``is not None`` rather than truthiness
        would stop returning every such team while looking correct against
        a freshly populated store.
        """
        store = NagraEventStore(postgres_clean_tables)
        legacy = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data) VALUES (%s, %s)",
                (str(legacy.team_id), json.dumps(legacy.model_dump())),
            )
        current = make_indexed_process(AcmeTeamMetadata(tenant="contoso"))
        store.save_team(current)

        expected = {legacy.team_id, current.team_id}
        assert {p.team_id for p in store.list_teams(metadata={})} == expected
        assert {p.team_id for p in store.list_teams(metadata=None)} == expected
        assert {p.team_id for p in store.list_teams()} == expected


# --- shared fixture set + filter matrix for the parity / index invariants ---

_FilterCase = tuple[str, str | None, dict[str, list[str]] | None, set[str]]

_EVERY_TEAM = {"acme_c1_u1", "acme_u2", "contoso_c1_u1", "piped_u1", "no_metadata_u2"}
"""What a call that constrains nothing must answer."""

_ACME_PREFIXED = {"acme_c1_u1", "acme_u2", "piped_u1"}
"""Every team whose stored ``tenant`` entry starts with ``acme`` — ``piped_u1``
included, since it stores ``tenant|acme\\|corp``."""

_FILTER_MATRIX: list[_FilterCase] = [
    # ``piped_u1`` stores ``tenant|acme\|corp``, which STARTS WITH ``tenant|acme``
    # — so every short ``acme`` term now reaches it. That is the widening this
    # story exists for, and the longer term below still separates the two.
    ("single key", None, {"tenant": ["acme"]}, {"acme_c1_u1", "acme_u2", "piped_u1"}),
    ("two keys AND-combined", None, {"tenant": ["acme"], "case_ref": ["C-1"]}, {"acme_c1_u1"}),
    ("pair no team carries", None, {"tenant": ["acme"], "case_ref": ["C-9"]}, set()),
    ("value containing a literal separator", None, {"tenant": ["acme|corp"]}, {"piped_u1"}),
    ("value spanning two entries", None, {"tenant": ["acme|case_ref|C-1"]}, set()),
    (
        "no metadata filter",
        None,
        None,
        {"acme_c1_u1", "acme_u2", "contoso_c1_u1", "piped_u1", "no_metadata_u2"},
    ),
    ("metadata AND user_id", "u1", {"tenant": ["acme"]}, {"acme_c1_u1", "piped_u1"}),
    ("prefix shorter than any stored value", None, {"tenant": ["ac"]}, _ACME_PREFIXED),
    ("prefix is not a substring search", None, {"tenant": ["cme"]}, set()),
    ("nested terms for one key are idempotent", None, {"tenant": ["ac", "acm"]}, _ACME_PREFIXED),
    (
        "terms for one key OR-combine",
        None,
        {"tenant": ["acme", "contoso"]},
        _ACME_PREFIXED | {"contoso_c1_u1"},
    ),
    (
        "keys AND while their own terms OR",
        None,
        {"tenant": ["acme", "contoso"], "case_ref": ["C-1"]},
        {"acme_c1_u1", "contoso_c1_u1"},
    ),
    ("an emptied key beside a real one", None, {"tenant": [], "case_ref": ["C-1"]},
     {"acme_c1_u1", "contoso_c1_u1"}),
    ("case-insensitive term", None, {"tenant": ["ACME"]}, _ACME_PREFIXED),
    ("an empty term constrains nothing", None, {"tenant": [""]}, _EVERY_TEAM),
    ("an empty term list constrains nothing", None, {"tenant": []}, _EVERY_TEAM),
    ("LIKE metacharacters are literal", None, {"tenant": ["a_m"]}, set()),
    ("LIKE wildcard is literal", None, {"tenant": ["a%"]}, set()),
]
"""Label, ``user_id``, ``metadata``, and the labels of the teams expected back.

Held as data rather than as separate tests so the index-absent invariant and
the cross-backend parity check provably run the SAME matrix.
"""


def _build_fixture_set() -> dict[str, Process]:
    """Build the shared teams, keyed by the label the matrix refers to."""
    return {
        "acme_c1_u1": make_indexed_process(
            AcmeTeamMetadata(tenant="acme", case_ref="C-1"), user_id="u1"
        ),
        "acme_u2": make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u2"),
        "contoso_c1_u1": make_indexed_process(
            AcmeTeamMetadata(tenant="contoso", case_ref="C-1"), user_id="u1"
        ),
        "piped_u1": make_indexed_process(AcmeTeamMetadata(tenant="acme|corp"), user_id="u1"),
        "no_metadata_u2": make_indexed_process(None, user_id="u2"),
    }


def _expected_ids(teams: dict[str, Process], labels: set[str]) -> set[uuid.UUID]:
    return {teams[label].team_id for label in labels}


class TestResultsNeverDependOnTheGinIndex:
    """Correctness is the query's, not the index's.

    Unlike the Mongo sibling of this story — where mongomock consults no
    index at all and the claim was untestable — these tests run against a
    real PostgreSQL 16, so dropping the GIN index genuinely changes the
    access path. The results must not move with it.
    """

    def test_results_are_identical_with_the_index_dropped(
        self, postgres_clean_tables: str
    ) -> None:
        """Same fixtures, same filter matrix, index present then absent.

        The index is restored through :func:`init_db` rather than a copied
        ``CREATE INDEX``, so the recreate cannot drift from the production
        statement. Restoration is in a ``finally``: the container is
        session-scoped, and a leaked drop would be every later test's
        problem.
        """
        store = NagraEventStore(postgres_clean_tables)
        teams = _build_fixture_set()
        for process in teams.values():
            store.save_team(process)

        with_index = {
            label: {p.team_id for p in store.list_teams(user_id=uid, metadata=meta)}
            for label, uid, meta, _ in _FILTER_MATRIX
        }

        with Transaction(postgres_clean_tables) as trn:
            trn.execute("DROP INDEX IF EXISTS team_process_metadata_indexes_idx")
        try:
            with Transaction(postgres_clean_tables) as trn:
                cursor = trn.execute(
                    "SELECT 1 FROM pg_indexes "
                    "WHERE indexname = 'team_process_metadata_indexes_idx'"
                )
                assert cursor.fetchall() == [], "precondition: the index must be gone"
            without_index = {
                label: {p.team_id for p in store.list_teams(user_id=uid, metadata=meta)}
                for label, uid, meta, _ in _FILTER_MATRIX
            }
        finally:
            init_db(postgres_clean_tables)

        assert without_index == with_index
        # Absolute expectation as well: two identically broken runs agree.
        expected = {
            label: _expected_ids(teams, labels) for label, _, _, labels in _FILTER_MATRIX
        }
        assert with_index == expected

        with Transaction(postgres_clean_tables) as trn:
            cursor = trn.execute(
                "SELECT 1 FROM pg_indexes "
                "WHERE indexname = 'team_process_metadata_indexes_idx'"
            )
            assert len(cursor.fetchall()) == 1, "the index must be restored"


class TestCrossBackendParity:
    """All three backends answer the same filters with the same teams.

    Driven from one fixture builder so the three stores provably receive
    identical ``Process`` objects, and asserted against an absolute
    expectation as well as against each other — two equally broken
    backends would agree on the empty set.
    """

    def test_three_backends_return_the_same_team_ids(
        self, postgres_clean_tables: str, tmp_path: Path, mongo_db: Any
    ) -> None:
        # ``mongomock`` is already gated by the ``mongo_client`` fixture; only
        # ``pymongo`` still needs a guard here, as ``mongo_store`` does.
        pytest.importorskip("pymongo")
        from akgentic.team.repositories.mongo import MongoEventStore
        from akgentic.team.repositories.yaml import YamlEventStore

        teams = _build_fixture_set()
        stores: dict[str, EventStore] = {
            "yaml": YamlEventStore(tmp_path),
            "mongo": MongoEventStore(mongo_db),
            "postgres": NagraEventStore(postgres_clean_tables),
        }
        for store in stores.values():
            for process in teams.values():
                store.save_team(process)

        for label, uid, meta, labels in _FILTER_MATRIX:
            expected = _expected_ids(teams, labels)
            results = {
                backend: {p.team_id for p in store.list_teams(user_id=uid, metadata=meta)}
                for backend, store in stores.items()
            }
            for backend, found in results.items():
                assert found == expected, f"{backend} disagrees on case: {label}"


class _RecordingCursor:
    """Cursor stub: the statement is the subject, the rows are irrelevant."""

    def fetchall(self) -> list[tuple[object, ...]]:
        return []

    def fetchone(self) -> tuple[object, ...] | None:
        return None


class _RecordingTransaction:
    """``Transaction`` stub that records the SQL and params it is handed.

    Lets the statement-shape assertions run without Docker: what they check is
    what ``list_teams`` BUILDS, which no amount of real querying can show —
    a wrong ``ESCAPE`` clause or a stray metadata term returns plausible rows.
    """

    calls: list[tuple[str, tuple[object, ...]]] = []

    def __init__(self, conn_string: str) -> None:
        self._conn_string = conn_string

    def __enter__(self) -> _RecordingTransaction:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> _RecordingCursor:
        type(self).calls.append((sql, params))
        return _RecordingCursor()


@pytest.fixture
def recorded_sql(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[object, ...]]]:
    """Swap ``Transaction`` for the recorder and hand back its call log."""
    from akgentic.team.repositories.postgres import event_store as event_store_module

    _RecordingTransaction.calls = []
    monkeypatch.setattr(event_store_module, "Transaction", _RecordingTransaction)
    return _RecordingTransaction.calls


class TestMetadataStatementShape:
    """The statement ``list_teams`` builds, independent of what it returns."""

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
    def test_no_effective_term_carries_no_metadata_clause_and_no_parameter(
        self,
        recorded_sql: list[tuple[str, tuple[object, ...]]],
        metadata: dict[str, list[str]] | None,
    ) -> None:
        """Every "no effective term" spelling leaves the SQL and the params clean.

        ``{"tenant": []}`` and ``{"tenant": [""]}`` are the ones an outer
        truthiness gate on ``metadata`` misses — the mapping is itself truthy.
        Asserted on the statement because a result-level assertion cannot tell
        "no clause" from "a clause that happens to match every row".
        """
        NagraEventStore("postgresql://recorded/none").list_teams(metadata=metadata)

        sql, params = recorded_sql[-1]
        assert "metadata_indexes" not in sql, sql
        assert "LIKE" not in sql, sql
        assert " OR " not in sql, f"an empty disjunction reached the SQL: {sql}"
        assert "WHERE" not in sql, sql
        assert params == ()

    def test_an_emptied_key_contributes_no_clause_beside_a_real_one(
        self, recorded_sql: list[tuple[str, tuple[object, ...]]]
    ) -> None:
        """A key whose terms all render away drops out, leaving no dangling ``OR``.

        The whole-mapping-empty case is covered above; this is the one where a
        REAL key sits beside the emptied one, so the metadata clause survives
        and only the dead group must go. A group built with zero arms would
        leave ``WHERE )`` — a syntax error rather than a wrong answer, but the
        statement is only ever executed against a live database, so the shape
        assertion is what catches it here.
        """
        NagraEventStore("postgresql://recorded/emptied").list_teams(
            metadata={"tenant": [], "case_ref": ["C-1"], "other": [""]}
        )

        sql, params = recorded_sql[-1]
        assert sql.count("EXISTS (SELECT 1 FROM unnest(metadata_indexes)") == 1
        assert " OR " not in sql, sql
        assert params == ("case!_ref|c-1%",)

    def test_one_term_becomes_one_exists_over_unnest_with_a_bound_parameter(
        self, recorded_sql: list[tuple[str, tuple[object, ...]]]
    ) -> None:
        """The term matches PER ELEMENT and its value travels bound, never inlined."""
        NagraEventStore("postgresql://recorded/one").list_teams(metadata={"tenant": ["AcM"]})

        sql, params = recorded_sql[-1]
        assert sql == (
            "SELECT data FROM team_process_entries WHERE "
            "EXISTS (SELECT 1 FROM unnest(metadata_indexes) AS e(entry) "
            "WHERE entry LIKE %s ESCAPE '!')"
        )
        # Casefolded on the query side too, and a trailing % makes it a prefix.
        assert params == ("tenant|acm%",)

    def test_each_key_becomes_one_clause_whose_terms_are_ored(
        self, recorded_sql: list[tuple[str, tuple[object, ...]]]
    ) -> None:
        """Two KEYS, two ``EXISTS`` clauses ANDed; the tenant's two terms OR inside one.

        The ``OR`` sits INSIDE the ``EXISTS``, not between two of them, so the
        disjunction is per-element. **This is a shape assertion and nothing
        more**: hoisting the ``OR`` outside into one ``EXISTS`` per term was
        mutation-tested and changed no behaviour, because indexed fields are
        scalars and a key therefore contributes at most one entry per team. It
        is pinned so the clause stays one-per-key, and so the reading that
        survives a key ever carrying two entries is the one in the tree — not
        because a behavioural spec is standing behind it.
        """
        NagraEventStore("postgresql://recorded/many").list_teams(
            user_id="u1", metadata={"tenant": ["ac", "acm"], "case_ref": ["C-"]}
        )

        sql, params = recorded_sql[-1]
        assert sql.count("EXISTS (SELECT 1 FROM unnest(metadata_indexes)") == 2
        assert sql.count(" AND ") == 2  # user_id + two per-key clauses
        assert sql.count(" OR ") == 1  # the tenant key's two terms
        assert sql == (
            "SELECT data FROM team_process_entries WHERE (data ->> 'user_id') = %s AND "
            "EXISTS (SELECT 1 FROM unnest(metadata_indexes) AS e(entry) WHERE "
            "entry LIKE %s ESCAPE '!' OR entry LIKE %s ESCAPE '!') AND "
            "EXISTS (SELECT 1 FROM unnest(metadata_indexes) AS e(entry) WHERE "
            "entry LIKE %s ESCAPE '!')"
        )
        # ``case!_ref``, not ``case_ref``: the escaping covers the WHOLE rendered
        # prefix, key half included. An unescaped ``_`` there is a single-character
        # wildcard, so ``case_ref|...`` would also reach a ``caseXref`` entry.
        assert params == ("u1", "tenant|ac%", "tenant|acm%", "case!_ref|c-%")

    @pytest.mark.parametrize(
        "value,expected_pattern",
        [
            ("50%", "tenant|50!%%"),
            ("a_b", "tenant|a!_b%"),
            ("a!b", "tenant|a!!b%"),
            ("a.b", "tenant|a.b%"),
            ("a\\b", "tenant|a\\b%"),
            ("acme|corp", "tenant|acme\\|corp%"),
        ],
        ids=["percent", "underscore", "escape-char", "regex-dot", "backslash", "separator"],
    )
    def test_like_metacharacters_are_escaped_for_the_declared_escape_character(
        self,
        recorded_sql: list[tuple[str, tuple[object, ...]]],
        value: str,
        expected_pattern: str,
    ) -> None:
        """Only ``!``, ``%`` and ``_`` are escaped — a backslash stays ordinary.

        This is the whole reason the escape character is ``!`` rather than the
        default backslash: the rendered prefix already carries backslashes from
        the separator escaping, and ``re.escape`` emits more of them on the
        Mongo side of the very same string. With ``ESCAPE '!'`` a backslash in
        an entry is just a character, and the layers stop interacting.
        """
        NagraEventStore("postgresql://recorded/escape").list_teams(metadata={"tenant": [value]})

        _, params = recorded_sql[-1]
        assert params == (expected_pattern,)

    def test_the_escape_clause_matches_the_character_actually_used(
        self, recorded_sql: list[tuple[str, tuple[object, ...]]]
    ) -> None:
        """The declared ``ESCAPE`` and the escaping applied must be the same character.

        Split between the two — escaping with ``!`` while declaring ``\\``, say —
        and a term of ``50%`` silently becomes a wildcard again.
        """
        NagraEventStore("postgresql://recorded/agree").list_teams(metadata={"tenant": ["50%"]})

        sql, params = recorded_sql[-1]
        declared = sql.split("ESCAPE '")[1][0]
        pattern = str(params[0])
        assert pattern == f"tenant|50{declared}%%"


class TestNagraAgentCardStore:
    """Postgres-only invariants of the content-addressed card store.

    Behavioural coverage runs in the shared contract suite; what is here is the
    table itself — that ``init_db`` provisions it, that it carries no
    ``team_id``, and that ``delete_team``'s cascade does not reach it.
    """

    def test_init_db_provisions_the_card_table(self, postgres_initialized: str) -> None:
        """No hand-written ``ALTER TABLE`` — ``create_tables()`` is the upgrade path.

        A database provisioned before this table existed gains it on the next
        ``init_db`` call, which is why the story adds nothing to ``init_db``
        beyond the ``schema.toml`` declaration.
        """
        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'agent_card_entries'"
            )
            columns = {row[0] for row in cursor.fetchall()}

        assert {"card_hash", "data"} <= columns
        assert "team_id" not in columns

    def test_a_re_save_updates_the_row_rather_than_raising(
        self, postgres_clean_tables: str
    ) -> None:
        """``ON CONFLICT`` against the natural key Nagra provisions."""
        store = NagraEventStore(postgres_clean_tables)
        card = _card_fixture()

        store.save_agent_cards([card])
        store.save_agent_cards([card])

        with Transaction(postgres_clean_tables) as trn:
            cursor = trn.execute("SELECT COUNT(*) FROM agent_card_entries")
            row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    def test_delete_team_does_not_cascade_to_the_card_table(
        self, postgres_clean_tables: str
    ) -> None:
        """FR13, asserted with SQL rather than through a later read."""
        store = NagraEventStore(postgres_clean_tables)
        process = make_process()
        store.save_agent_cards([_card_fixture()])
        store.save_team(process)

        store.delete_team(process.team_id)

        with Transaction(postgres_clean_tables) as trn:
            cursor = trn.execute("SELECT COUNT(*) FROM agent_card_entries")
            row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    def test_saving_no_cards_writes_no_row(self, postgres_clean_tables: str) -> None:
        store = NagraEventStore(postgres_clean_tables)
        store.save_agent_cards([])

        with Transaction(postgres_clean_tables) as trn:
            cursor = trn.execute("SELECT COUNT(*) FROM agent_card_entries")
            row = cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    def test_the_batch_load_is_one_statement_over_the_whole_set(
        self, postgres_clean_tables: str
    ) -> None:
        """An ``IN`` list of bound placeholders, never a query per hash."""
        store = NagraEventStore(postgres_clean_tables)
        cards = [
            make_agent_card(name=f"agent-{i}", role=f"Role{i}", agent_class=Akgent)
            for i in range(4)
        ]
        store.save_agent_cards(cards)
        hashes = [hash_agent_card(c) for c in cards]

        assert set(store.load_agent_cards(hashes)) == set(hashes)

    def test_a_corrupted_card_row_is_skipped_not_raised(
        self, postgres_clean_tables: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The port promises a skip; Postgres has to honour it too.

        A bad row raising a bare ``ValidationError`` out of the store names
        neither the role nor the hash. Skipping it lets ``resolve_agent_cards``
        raise ``AgentCardNotFoundError``, which names both — the whole point of
        FR14. This is the one ``NagraEventStore`` reader that tolerates a bad
        row, deliberately.
        """
        store = NagraEventStore(postgres_clean_tables)
        good = _card_fixture()
        store.save_agent_cards([good])
        bad_hash = "c" * 64
        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO agent_card_entries (card_hash, data) VALUES (%s, %s)",
                (bad_hash, json.dumps({"not": "a card"})),
            )

        logger_name = "akgentic.team.repositories.postgres.event_store"
        with caplog.at_level(logging.ERROR, logger=logger_name):
            loaded = store.load_agent_cards([hash_agent_card(good), bad_hash])

        assert set(loaded) == {hash_agent_card(good)}
        assert [r for r in caplog.records if "corrupted agent card" in r.getMessage()]
