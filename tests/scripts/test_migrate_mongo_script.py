"""Tests for ``akgentic.team.scripts.migrate_mongo`` — story 31-5, AC 15-18.

Runs against ``mongomock`` (in the ``dev`` extra), so no MongoDB server and no
Docker. Runs in the default gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import akgentic.team.scripts.migrate_mongo as script
from tests.models.conftest import make_process
from tests.scripts.test_migration import _legacy_document, _team_card


@pytest.fixture(autouse=True)
def _clear_mongo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient ``MONGO_*`` value may decide what these specs measure."""
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("MONGO_DB", raising=False)


def _mongo_store(mongo_db: Any) -> Any:
    """A ``MongoEventStore`` over the mongomock database, for reading back."""
    from akgentic.team.repositories.mongo import MongoEventStore

    return MongoEventStore(mongo_db)


def _seed(mongo_db: Any, document: dict[str, Any]) -> uuid.UUID:
    """Insert one raw stored document into the ``teams`` collection."""
    from akgentic.team.repositories.mongo import TEAMS_COLLECTION

    mongo_db[TEAMS_COLLECTION].insert_one(dict(document))
    return uuid.UUID(str(document["team_id"]))


class TestMigrateMongoScriptConfiguration:
    """AC 16, 17: the ``--help`` text and the missing-configuration exit code."""

    def test_help_carries_the_rollout_ordering_constraint(self) -> None:
        """AC 16: the operator learns WHEN to run this from ``--help`` alone."""
        text = script.build_parser().format_help()

        assert "BETWEEN stopping the old version and starting the new one" in text
        assert "--mongo-uri" in text
        assert "--mongo-db" in text

    def test_missing_uri_returns_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC 17: the same variable names ``init_mongo`` uses, and the same code."""
        monkeypatch.setenv("MONGO_DB", "akgentic")

        assert script.main([]) == 2

    def test_missing_db_returns_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each variable is checked independently."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")

        assert script.main([]) == 2

    @pytest.mark.parametrize("blank_var", ["MONGO_URI", "MONGO_DB"])
    def test_empty_value_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, blank_var: str
    ) -> None:
        """An empty variable is treated as unset — same exit code 2."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", "akgentic")
        monkeypatch.setenv(blank_var, "")

        assert script.main([]) == 2

    def test_the_flags_beat_the_environment(self, mongo_client: Any) -> None:
        """A flag is what an operator reaches for when the ``.env`` is wrong."""
        with patch("pymongo.MongoClient", return_value=mongo_client):
            assert (
                script.main(["--mongo-uri", "mongodb://fake", "--mongo-db", "flagged"]) == 0
            )

    def test_unavailable_backend_returns_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ``[mongo]`` extra exits 1 rather than raising a traceback."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", "akgentic")

        with patch.dict(sys.modules, {"pymongo": None}):
            assert script.main([]) == 1

    def test_unreachable_server_returns_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``MongoClient`` connects lazily, so a ping is what keeps exit 0 honest."""
        pytest.importorskip("pymongo")
        from pymongo.errors import ServerSelectionTimeoutError

        monkeypatch.setenv("MONGO_URI", "mongodb://unreachable")
        monkeypatch.setenv("MONGO_DB", "akgentic")
        client = MagicMock()
        client.__enter__.return_value = client
        client.admin.command.side_effect = ServerSelectionTimeoutError("no servers")

        with patch("pymongo.MongoClient", return_value=client):
            assert script.main([]) == 1


class TestMigrateMongoScriptRuns:
    """AC 15, 18: the script converts a real (mongomock) store."""

    def test_an_unmigrated_store_migrates_and_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: Any, mongo_db: Any
    ) -> None:
        """AC 15, 18: read raw, write through the store, exit 0."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", mongo_db.name)
        team_id = _seed(mongo_db, _legacy_document(_team_card()))
        assert _mongo_store(mongo_db).load_team(team_id) is None

        with patch("pymongo.MongoClient", return_value=mongo_client):
            assert script.main([]) == 0

        migrated = _mongo_store(mongo_db).load_team(team_id)
        assert migrated is not None
        assert migrated.entry_point.name == "lead"
        assert migrated.agent_cards

    def test_a_second_run_is_a_no_op_and_still_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: Any, mongo_db: Any
    ) -> None:
        """Idempotent at the script level, not only in the core."""
        from akgentic.team.repositories.mongo import AGENT_CARDS_COLLECTION

        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", mongo_db.name)
        _seed(mongo_db, _legacy_document())

        with patch("pymongo.MongoClient", return_value=mongo_client):
            assert script.main([]) == 0
            after_first = mongo_db[AGENT_CARDS_COLLECTION].count_documents({})
            assert script.main([]) == 0

        assert mongo_db[AGENT_CARDS_COLLECTION].count_documents({}) == after_first

    def test_an_empty_store_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: Any, mongo_db: Any
    ) -> None:
        """Nothing to do is success, not a failure."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", mongo_db.name)

        with patch("pymongo.MongoClient", return_value=mongo_client):
            assert script.main([]) == 0

    def test_a_failed_document_exits_1_without_stopping_the_run(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: Any, mongo_db: Any
    ) -> None:
        """AC 8, 9: the good document still converts and the exit code is 1."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", mongo_db.name)
        broken = _legacy_document()
        broken["team_card"] = {"not": "a team card"}
        _seed(mongo_db, broken)
        good_id = _seed(mongo_db, _legacy_document())

        with patch("pymongo.MongoClient", return_value=mongo_client):
            assert script.main([]) == 1

        assert _mongo_store(mongo_db).load_team(good_id) is not None

    def test_an_already_migrated_store_needs_no_conversion(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: Any, mongo_db: Any
    ) -> None:
        """The realistic re-run: every document already carries the projection."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setenv("MONGO_DB", mongo_db.name)
        store = _mongo_store(mongo_db)
        process = make_process(team_card=_team_card())
        store.save_team(process)

        with patch("pymongo.MongoClient", return_value=mongo_client):
            assert script.main([]) == 0

        reloaded = store.load_team(process.team_id)
        assert reloaded is not None
        assert reloaded.entry_point == process.entry_point


class TestMigrateMongoScriptModulePath:
    """The module is runnable as ``python -m akgentic.team.scripts.migrate_mongo``."""

    def test_python_dash_m_without_env_exits_2(self) -> None:
        """Proves the module path resolves and ``main()`` reaches ``sys.exit``."""
        env = dict(os.environ)
        env.pop("MONGO_URI", None)
        env.pop("MONGO_DB", None)
        result = subprocess.run(
            [sys.executable, "-m", "akgentic.team.scripts.migrate_mongo"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"
