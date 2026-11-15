"""Tests for ``akgentic.team.scripts.copy_yaml_to_postgres``.

Gated like ``test_migrate_postgres_script``:

* :class:`TestCopyScriptConfiguration` needs no service — ``main()`` returns
  before it touches a database.
* :class:`TestCopyScriptAgainstADatabase` drives the script against a real
  PostgreSQL container, so it needs Docker and carries
  ``pytest.mark.integration`` (deselected by the default ``addopts``).
  The copy logic itself is covered against every backend in
  ``tests/repositories/test_store_copy.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from akgentic.core.agent import Akgent

import akgentic.team.scripts.copy_yaml_to_postgres as script
from akgentic.team.models import TeamStatus
from akgentic.team.repositories.yaml import YamlEventStore
from tests.conftest import seed_agent_cards
from tests.models.conftest import make_persisted_event, make_process, make_team_card

_ENV_VAR = "DB_CONN_STRING_PERSISTENCE"


@pytest.fixture(autouse=True)
def _no_dsn_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)


def _yaml_store_with_one_team(root: Path) -> YamlEventStore:
    store = YamlEventStore(root)
    team_card = make_team_card(agent_class=Akgent)
    seed_agent_cards(store, team_card)
    process = make_process(team_card=team_card, status=TeamStatus.STOPPED)
    store.save_team(process)
    for sequence in (1, 2):
        store.save_event(make_persisted_event(team_id=process.team_id, sequence=sequence))
    return store


class TestCopyScriptConfiguration:
    """Exit 2 for anything the operator must fix before a copy can start."""

    def test_missing_dsn_exits_2(self, tmp_path: Path) -> None:
        assert script.main(["--data-dir", str(tmp_path)]) == 2

    def test_non_postgresql_scheme_exits_2_without_logging_the_dsn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            code = script.main(
                ["--data-dir", str(tmp_path), "--conn-string", "postgres://u:s3cret-pw@h/db"]
            )
        assert code == 2
        assert "s3cret-pw" not in caplog.text

    def test_missing_source_directory_exits_2(self, tmp_path: Path) -> None:
        code = script.main(
            ["--data-dir", str(tmp_path / "absent"), "--conn-string", "postgresql://u:p@h/db"]
        )
        assert code == 2

    def test_dsn_is_read_from_the_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the env var set, validation moves on to the source directory."""
        monkeypatch.setenv(_ENV_VAR, "postgresql://u:p@h/db")
        assert script.main(["--data-dir", str(tmp_path / "absent")]) == 2


@pytest.mark.integration
class TestCopyScriptAgainstADatabase:
    """The script end to end: YAML directory in, Postgres rows out."""

    @pytest.fixture(autouse=True)
    def _needs_postgres(self) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("testcontainers.postgres")

    def test_copies_and_a_rerun_is_a_no_op(
        self, tmp_path: Path, postgres_clean_tables: str
    ) -> None:
        from akgentic.team.repositories.postgres import NagraEventStore

        source = _yaml_store_with_one_team(tmp_path / "es")
        (team,) = source.list_teams()

        args = ["--data-dir", str(tmp_path / "es"), "--conn-string", postgres_clean_tables]
        assert script.main(args) == 0
        assert script.main(args) == 0

        target = NagraEventStore(postgres_clean_tables)
        assert target.load_team(team.team_id) is not None
        assert [e.sequence for e in target.load_events(team.team_id)] == [1, 2]

    def test_dry_run_writes_nothing(self, tmp_path: Path, postgres_clean_tables: str) -> None:
        from akgentic.team.repositories.postgres import NagraEventStore

        source = _yaml_store_with_one_team(tmp_path / "es")
        (team,) = source.list_teams()

        code = script.main(
            [
                "--data-dir",
                str(tmp_path / "es"),
                "--conn-string",
                postgres_clean_tables,
                "--dry-run",
            ]
        )

        assert code == 0
        assert NagraEventStore(postgres_clean_tables).load_team(team.team_id) is None
