"""Tests for ``_ensure_schema_loaded`` and ``init_db`` (AC #4, #5, #6, #7, #18).

Two groups:

* **Unit tests** that don't need a Postgres container — spy on
  ``Schema.default.load_toml`` to assert ``_ensure_schema_loaded`` runs the
  load exactly once, and read the event-store stub to confirm ``init_db``
  is not called from its constructor.
* **Integration tests** that use the session-scoped ``postgres_initialized``
  fixture to exercise ``init_db`` against a real container, including an
  idempotency pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nagra")


class TestEnsureSchemaLoadedIdempotent:
    """AC #4: ``_ensure_schema_loaded`` performs its work exactly once."""

    def test_load_toml_called_once_across_repeated_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spy ``load_toml`` and assert it fires exactly once across N calls.

        Stubbing ``load_toml`` keeps ``Schema.default`` clean so the spy
        does not poison later tests / fixtures. We also reset the
        ``_SCHEMA_LOADED`` guard back to its pre-test value so the real
        loader still runs exactly once when the session fixture wakes up.
        """
        from nagra import Schema

        import akgentic.team.repositories.postgres as pg_pkg

        original_flag = pg_pkg._SCHEMA_LOADED
        monkeypatch.setattr(pg_pkg, "_SCHEMA_LOADED", False, raising=False)

        call_count = {"n": 0}

        def stub_load(path: Path) -> object:
            call_count["n"] += 1
            return None

        monkeypatch.setattr(Schema.default, "load_toml", stub_load)

        pg_pkg._ensure_schema_loaded()
        pg_pkg._ensure_schema_loaded()
        pg_pkg._ensure_schema_loaded()

        assert call_count["n"] == 1

        monkeypatch.setattr(pg_pkg, "_SCHEMA_LOADED", original_flag, raising=False)


class TestEventStoreStubDoesNotCallInitDb:
    """AC #7: ``NagraEventStore.__init__`` must NOT call ``init_db``."""

    def test_event_store_source_does_not_call_init_db(self) -> None:
        stub_path = (
            Path(__file__).parents[3]
            / "src"
            / "akgentic"
            / "team"
            / "repositories"
            / "postgres"
            / "event_store.py"
        )
        text = stub_path.read_text()
        assert "init_db(" not in text, (
            "event_store.py must not call init_db() — operators run it"
        )
        assert "_ensure_schema_loaded" in text, (
            "event_store.py must call _ensure_schema_loaded() in __init__"
        )


class TestInitDbIntegration:
    """AC #5, #6, #18: ``init_db`` creates tables and is idempotent."""

    def test_init_db_creates_three_tables(self, postgres_initialized: str) -> None:
        from nagra import Transaction

        expected = {"team_process_entries", "event_entries", "agent_state_entries"}
        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            found = {row[0] for row in cursor.fetchall()}
        assert expected.issubset(found)

    def test_init_db_is_idempotent(self, postgres_initialized: str) -> None:
        from nagra import Transaction

        from akgentic.team.repositories.postgres import init_db

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            before = {row[0] for row in cursor.fetchall()}

        # Second call must not raise and must not change the table set.
        init_db(postgres_initialized)

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            after = {row[0] for row in cursor.fetchall()}

        assert before == after

    def test_init_db_creates_user_id_functional_index(
        self, postgres_initialized: str
    ) -> None:
        """After ``init_db``, ``team_process_user_id_idx`` exists in ``pg_indexes``.

        Story 19.4 — the functional expression index
        ``((data ->> 'user_id'))`` on ``team_process_entries`` backs the
        ``WHERE (data ->> 'user_id') = %s`` push-down in
        :meth:`NagraEventStore.list_teams`. See ADR-16 §4.
        """
        from nagra import Transaction

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT 1 FROM pg_indexes "
                "WHERE indexname = 'team_process_user_id_idx'"
            )
            rows = cursor.fetchall()
        assert len(rows) == 1

    def test_init_db_metadata_indexes_column_is_a_text_array(
        self, postgres_initialized: str
    ) -> None:
        """``metadata_indexes`` exists on ``team_process_entries`` as ``TEXT[]``.

        Asserted against the live database rather than by reading
        ``schema.toml`` back: the point at issue is what Nagra's postgresql
        flavor emitted for the declared ``str[]`` dtype, which the TOML
        cannot answer. ``data_type = 'ARRAY'`` with ``udt_name = '_text'``
        is how PostgreSQL spells ``TEXT[]`` in ``information_schema``.
        """
        from nagra import Transaction

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT data_type, udt_name, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_name = 'team_process_entries' "
                "AND column_name = 'metadata_indexes'"
            )
            rows = cursor.fetchall()

        assert len(rows) == 1, "expected exactly one metadata_indexes column"
        data_type, udt_name, is_nullable = rows[0]
        assert data_type == "ARRAY"
        assert udt_name == "_text"
        # Nullable by construction: the column is not in natural_key nor
        # not_null, which is what lets it be added to a populated table.
        assert is_nullable == "YES"

    def test_init_db_adds_the_column_to_a_populated_existing_table(
        self, postgres_clean_tables: str
    ) -> None:
        """The upgrade path: a POPULATED live table missing the column gets it back.

        Every other schema assertion here is satisfied by a table Nagra
        created from scratch, which proves nothing about a deployment that
        predates the column. Dropping it and re-running ``init_db``
        simulates exactly that upgrade — ``Schema.create_tables()`` adds
        declared-but-absent columns to existing tables, so no hand-written
        ``ALTER`` is needed.

        The table carries a row throughout, which is the half of the claim an
        empty table cannot make: an ``ADD COLUMN`` on an empty table succeeds
        whatever the column's nullability, so only a populated one shows that
        a real deployment upgrades without losing rows and without needing a
        backfill. The surviving row lands on ``NULL`` and from then on behaves
        exactly like any other pre-migration row.

        Dropping the column also drops the GIN index that depends on it, so
        this covers the index's half of the upgrade too. Restored in a
        ``finally``: the container is session-scoped.
        """
        import json

        from nagra import Transaction

        from akgentic.team.repositories.postgres import NagraEventStore, init_db
        from tests.models.conftest import AcmeTeamMetadata, make_indexed_process

        existing = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u1")
        with Transaction(postgres_clean_tables) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data, metadata_indexes) "
                "VALUES (%s, %s, %s)",
                (
                    str(existing.team_id),
                    json.dumps(existing.model_dump()),
                    list(existing.metadata_indexes),
                ),
            )

        try:
            with Transaction(postgres_clean_tables) as trn:
                trn.execute(
                    "ALTER TABLE team_process_entries DROP COLUMN metadata_indexes"
                )

            with Transaction(postgres_clean_tables) as trn:
                cursor = trn.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'team_process_entries' "
                    "AND column_name = 'metadata_indexes'"
                )
                assert cursor.fetchall() == [], "precondition: column must be gone"
        finally:
            init_db(postgres_clean_tables)

        with Transaction(postgres_clean_tables) as trn:
            cursor = trn.execute(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'team_process_entries' "
                "AND column_name = 'metadata_indexes'"
            )
            column_rows = cursor.fetchall()
            cursor = trn.execute(
                "SELECT 1 FROM pg_indexes "
                "WHERE indexname = 'team_process_metadata_indexes_idx'"
            )
            index_rows = cursor.fetchall()
            cursor = trn.execute(
                "SELECT metadata_indexes FROM team_process_entries WHERE id = %s",
                (str(existing.team_id),),
            )
            surviving = cursor.fetchall()

        assert len(column_rows) == 1
        assert column_rows[0][0] == "_text"
        assert len(index_rows) == 1, "the GIN index must come back with the column"
        # The row survived the round trip and is now an ordinary legacy row:
        # NULL index, lists normally, matched by no metadata filter.
        assert surviving == [(None,)], "the pre-existing row must survive on NULL"
        store = NagraEventStore(postgres_clean_tables)
        assert [p.team_id for p in store.list_teams()] == [existing.team_id]
        assert store.list_teams(metadata={"tenant": "acme"}) == []

    def test_init_db_creates_metadata_indexes_gin_index(
        self, postgres_initialized: str
    ) -> None:
        """``team_process_metadata_indexes_idx`` exists and is a GIN index.

        The name is part of the contract — an operator inspecting a
        database, and the drop/restore test for the index-absent invariant,
        both address it by name. ``USING gin`` is asserted because a btree
        index over the same column would exist under the same name while
        serving no ``@>`` query.
        """
        from nagra import Transaction

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'team_process_metadata_indexes_idx'"
            )
            rows = cursor.fetchall()

        assert len(rows) == 1
        indexdef = rows[0][0]
        assert "USING gin" in indexdef
        assert "metadata_indexes" in indexdef
        assert "team_process_entries" in indexdef

    def test_repeated_init_db_leaves_one_column_and_one_index(
        self, postgres_initialized: str
    ) -> None:
        """AC: idempotence across the column and the GIN index together.

        The session fixture already ran ``init_db`` once; two more calls
        must not raise, must not duplicate the column, must not duplicate
        the index, and must leave the table set unchanged.
        """
        from nagra import Transaction

        from akgentic.team.repositories.postgres import init_db

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            before = {row[0] for row in cursor.fetchall()}

        init_db(postgres_initialized)  # must not raise
        init_db(postgres_initialized)  # must not raise

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            after = {row[0] for row in cursor.fetchall()}
            cursor = trn.execute(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'team_process_entries' "
                "AND column_name = 'metadata_indexes'"
            )
            column_count = cursor.fetchone()[0]
            cursor = trn.execute(
                "SELECT count(*) FROM pg_indexes "
                "WHERE indexname = 'team_process_metadata_indexes_idx'"
            )
            index_count = cursor.fetchone()[0]

        assert before == after
        assert column_count == 1
        assert index_count == 1

    def test_init_db_user_id_index_creation_is_idempotent(
        self, postgres_initialized: str
    ) -> None:
        """A second ``init_db`` call exercises the ``IF NOT EXISTS`` branch.

        The session-scoped ``postgres_initialized`` fixture already ran
        ``init_db`` once during setup; calling again here must not raise
        and must not duplicate the index. Idempotency is delivered by the
        ``CREATE INDEX IF NOT EXISTS`` clause. See ADR-16 §4.
        """
        from nagra import Transaction

        from akgentic.team.repositories.postgres import init_db

        init_db(postgres_initialized)  # must not raise

        with Transaction(postgres_initialized) as trn:
            cursor = trn.execute(
                "SELECT 1 FROM pg_indexes "
                "WHERE indexname = 'team_process_user_id_idx'"
            )
            rows = cursor.fetchall()
        assert len(rows) == 1
