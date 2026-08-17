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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
import pytest
from nagra import Transaction  # type: ignore[import-untyped]

from akgentic.team.models import Process, TeamStatus
from akgentic.team.repositories.postgres import NagraEventStore, init_db

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

from tests.models.conftest import AcmeTeamMetadata, make_indexed_process, make_persisted_event


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
        assert set(stored) == {"tenant|acme", "case_ref|C-1"}

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
        assert set(stored) == {"tenant|contoso", "case_ref|C-9"}
        # And the filter agrees with the column: the old value is gone.
        assert store.list_teams(metadata={"tenant": "acme"}) == []
        assert [p.team_id for p in store.list_teams(metadata={"tenant": "contoso"})] == [
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

        assert store.list_teams(metadata={"tenant": "acme"}) == []
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

        found = store.list_teams(metadata={"tenant": "acme"})
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
        assert store.list_teams(metadata={"tenant": "acme"}) == []

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

_FilterCase = tuple[str, str | None, dict[str, str] | None, set[str]]

_FILTER_MATRIX: list[_FilterCase] = [
    ("single key", None, {"tenant": "acme"}, {"acme_c1_u1", "acme_u2"}),
    ("two keys AND-combined", None, {"tenant": "acme", "case_ref": "C-1"}, {"acme_c1_u1"}),
    ("pair no team carries", None, {"tenant": "acme", "case_ref": "C-9"}, set()),
    ("value containing a literal separator", None, {"tenant": "acme|corp"}, {"piped_u1"}),
    ("value spanning two entries", None, {"tenant": "acme|case_ref|C-1"}, set()),
    (
        "no metadata filter",
        None,
        None,
        {"acme_c1_u1", "acme_u2", "contoso_c1_u1", "piped_u1", "no_metadata_u2"},
    ),
    ("metadata AND user_id", "u1", {"tenant": "acme"}, {"acme_c1_u1"}),
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
        self, postgres_clean_tables: str, tmp_path: Path, mongo_db: object
    ) -> None:
        pytest.importorskip("pymongo")
        pytest.importorskip("mongomock")
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
