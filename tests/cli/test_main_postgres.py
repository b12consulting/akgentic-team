"""Tests for ``ak-team --backend postgres``.

Option validation runs without a database. The live tests read a store written
through ``NagraEventStore`` and need ``nagra`` and ``testcontainers[postgres]``
(they skip otherwise), the same as the rest of the Postgres suite.
"""

from __future__ import annotations

import json
import sys

import pytest
from typer.testing import CliRunner

from akgentic.team.cli.main import app
from akgentic.team.models import TeamStatus
from tests.models.conftest import make_persisted_event, make_process

_DSN_ENV = "DB_CONN_STRING_PERSISTENCE"


@pytest.fixture(autouse=True)
def _no_dsn_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator's exported DSN must not satisfy the 'missing DSN' tests."""
    monkeypatch.delenv(_DSN_ENV, raising=False)


class TestPostgresOptions:
    """``--backend postgres`` validates its DSN before touching a store."""

    def test_missing_dsn_names_the_flag_and_the_env_var(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(app, ["--backend", "postgres", "list"])
        assert result.exit_code == 1
        assert "--postgres-conn-string" in result.output
        assert _DSN_ENV in result.output

    @pytest.mark.parametrize(
        "dsn",
        ["postgres://u:s3cret-pw@localhost/db", "mysql://u:s3cret-pw@localhost/db"],
    )
    def test_non_postgresql_scheme_is_rejected_without_echoing_it(
        self, cli_runner: CliRunner, dsn: str
    ) -> None:
        result = cli_runner.invoke(
            app, ["--backend", "postgres", "--postgres-conn-string", dsn, "list"]
        )
        assert result.exit_code == 1
        assert "postgresql://" in result.output
        assert "s3cret-pw" not in result.output

    def test_missing_extra_names_the_install_command(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``None`` in sys.modules makes the import raise ImportError, whatever
        # the environment has installed.
        monkeypatch.setitem(sys.modules, "akgentic.team.repositories.postgres", None)
        result = cli_runner.invoke(
            app,
            [
                "--backend",
                "postgres",
                "--postgres-conn-string",
                "postgresql://u:pw@localhost/db",
                "list",
            ],
        )
        assert result.exit_code == 1
        assert "akgentic-team[cli,postgres]" in result.output


class TestPostgresCommands:
    """``list`` and ``inspect`` read a real Postgres store."""

    @pytest.fixture(autouse=True)
    def _needs_postgres(self) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("testcontainers.postgres")

    def test_list_and_inspect_read_the_postgres_store(
        self,
        cli_runner: CliRunner,
        postgres_clean_tables: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from akgentic.team.repositories.postgres import NagraEventStore

        store = NagraEventStore(postgres_clean_tables)
        process = make_process(status=TeamStatus.STOPPED)
        store.save_team(process)
        store.save_event(make_persisted_event(team_id=process.team_id, sequence=1))

        # The DSN comes from the env var operators already export for init_db.
        monkeypatch.setenv(_DSN_ENV, postgres_clean_tables)
        listed = cli_runner.invoke(app, ["--backend", "postgres", "--format", "json", "list"])
        assert listed.exit_code == 0, listed.output
        assert str(process.team_id) in listed.output

        inspected = cli_runner.invoke(
            app,
            ["--backend", "postgres", "--format", "json", "inspect", str(process.team_id)],
        )
        assert inspected.exit_code == 0, inspected.output
        assert json.loads(inspected.output)["event_count"] == 1
