"""Tests for ``akgentic.team.scripts.migrate_postgres`` — story 31-5, AC 15-18, 25.

Two classes, deliberately gated differently:

* :class:`TestMigratePostgresScriptConfiguration` needs no service at all —
  ``main()`` returns 2 before it imports the backend, and ``--help`` imports
  nothing. It runs in the default gate.
* :class:`TestMigratePostgresScriptAgainstADatabase` drives the script against a
  real PostgreSQL container and therefore needs **Docker**. It carries
  ``pytest.mark.integration`` (AC 25) so ``pytest -m "not integration"`` — which
  is this package's default ``addopts`` — deselects it. That is deliberately
  stronger than the surrounding convention: the ``importorskip`` gates the rest
  of the Postgres suite uses skip on a failed *import*, not on an unavailable
  service, so with ``testcontainers`` installed and Docker down they error
  rather than skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from typing import Any

import pytest

import akgentic.team.scripts.migrate_postgres as script
from tests.models.conftest import make_process
from tests.scripts.test_migration import _legacy_document, _team_card

_ENV_VAR = "DB_CONN_STRING_PERSISTENCE"


def _seed(conn_string: str, document: dict[str, Any]) -> uuid.UUID:
    """Insert one raw stored document into ``team_process_entries``."""
    from nagra import Transaction  # type: ignore[import-untyped]

    team_id = uuid.UUID(str(document["team_id"]))
    with Transaction(conn_string) as trn:
        trn.execute(
            "INSERT INTO team_process_entries (id, data, metadata_indexes) "
            "VALUES (%s, %s, %s)",
            (str(team_id), json.dumps(document), []),
        )
    return team_id


class TestMigratePostgresScriptConfiguration:
    """AC 16, 17: reachable without Docker — no backend import, no connection."""

    def test_help_carries_the_rollout_ordering_constraint(self) -> None:
        """AC 16: the operator learns WHEN to run this from ``--help`` alone."""
        text = script.build_parser().format_help()

        assert "BETWEEN stopping the old version and starting the new one" in text
        assert "--conn-string" in text

    def test_missing_conn_string_returns_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC 17: the same variable name ``init_db`` uses, and the same code."""
        monkeypatch.delenv(_ENV_VAR, raising=False)

        assert script.main([]) == 2

    def test_empty_conn_string_returns_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty variable is treated as unset — same exit code 2."""
        monkeypatch.setenv(_ENV_VAR, "")

        assert script.main([]) == 2

    def test_unavailable_backend_returns_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ``[postgres]`` extra exits 1 rather than raising a traceback.

        Relies on the ``NagraEventStore`` import living inside ``main()`` — a
        module-scope import would make this path unreachable.
        """
        monkeypatch.setenv(_ENV_VAR, "postgresql://fake/db")

        monkeypatch.setitem(sys.modules, "akgentic.team.repositories.postgres", None)
        assert script.main([]) == 1

    def test_an_unreachable_database_returns_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused connection is exit 1, never a false success."""
        pytest.importorskip("nagra")
        monkeypatch.setenv(
            _ENV_VAR, "postgresql://nobody:nobody@127.0.0.1:1/does-not-exist"
        )

        assert script.main([]) == 1


class TestMigratePostgresScriptModulePath:
    """The module is runnable as ``python -m akgentic.team.scripts.migrate_postgres``."""

    def test_python_dash_m_without_env_exits_2(self) -> None:
        """Proves the module path resolves and ``main()`` reaches ``sys.exit``."""
        env = dict(os.environ)
        env.pop(_ENV_VAR, None)
        result = subprocess.run(
            [sys.executable, "-m", "akgentic.team.scripts.migrate_postgres"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"


@pytest.mark.integration
class TestMigratePostgresScriptAgainstADatabase:
    """AC 15, 18, 25: the real conversion — NEEDS DOCKER, hence the marker."""

    def test_an_unmigrated_store_migrates_and_exits_0(
        self, postgres_clean_tables: str
    ) -> None:
        """AC 15, 18: read raw, write through the store, exit 0."""
        from akgentic.team.repositories.postgres import NagraEventStore

        team_id = _seed(postgres_clean_tables, _legacy_document(_team_card()))
        store = NagraEventStore(postgres_clean_tables)
        # The legacy shape is refused by validation and skipped, not raised.
        assert store.load_team(team_id) is None

        assert script.main(["--conn-string", postgres_clean_tables]) == 0

        migrated = store.load_team(team_id)
        assert migrated is not None
        assert migrated.entry_point.name == "lead"
        assert store.load_agent_cards([ref.card_hash for ref in migrated.agent_cards])

    def test_a_second_run_is_a_no_op_and_still_exits_0(
        self, postgres_clean_tables: str
    ) -> None:
        """Idempotent at the script level, not only in the core."""
        from nagra import Transaction  # type: ignore[import-untyped]

        _seed(postgres_clean_tables, _legacy_document())

        assert script.main(["--conn-string", postgres_clean_tables]) == 0
        with Transaction(postgres_clean_tables) as trn:
            after_first = trn.execute(
                "SELECT count(*) FROM agent_card_entries"
            ).fetchone()[0]
        assert script.main(["--conn-string", postgres_clean_tables]) == 0

        with Transaction(postgres_clean_tables) as trn:
            after_second = trn.execute(
                "SELECT count(*) FROM agent_card_entries"
            ).fetchone()[0]
        assert after_second == after_first

    def test_the_env_var_drives_the_run(
        self, postgres_clean_tables: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 17: ``DB_CONN_STRING_PERSISTENCE`` is the documented configuration."""
        monkeypatch.setenv(_ENV_VAR, postgres_clean_tables)
        team_id = _seed(postgres_clean_tables, _legacy_document())

        assert script.main([]) == 0

        from akgentic.team.repositories.postgres import NagraEventStore

        assert NagraEventStore(postgres_clean_tables).load_team(team_id) is not None

    def test_a_failed_document_exits_1_without_stopping_the_run(
        self, postgres_clean_tables: str
    ) -> None:
        """AC 8, 9: the good document still converts and the exit code is 1."""
        from akgentic.team.repositories.postgres import NagraEventStore

        broken = _legacy_document()
        broken["team_card"] = {"not": "a team card"}
        _seed(postgres_clean_tables, broken)
        good_id = _seed(postgres_clean_tables, _legacy_document())

        assert script.main(["--conn-string", postgres_clean_tables]) == 1

        assert NagraEventStore(postgres_clean_tables).load_team(good_id) is not None

    def test_an_already_migrated_store_needs_no_conversion(
        self, postgres_clean_tables: str
    ) -> None:
        """The realistic re-run: every document already carries the projection."""
        from akgentic.team.repositories.postgres import NagraEventStore

        store = NagraEventStore(postgres_clean_tables)
        process = make_process(team_card=_team_card())
        store.save_team(process)

        assert script.main(["--conn-string", postgres_clean_tables]) == 0

        reloaded = store.load_team(process.team_id)
        assert reloaded is not None
        assert reloaded.entry_point == process.entry_point

    def test_an_empty_store_exits_0(self, postgres_clean_tables: str) -> None:
        """Nothing to do is success, not a failure."""
        assert script.main(["--conn-string", postgres_clean_tables]) == 0
