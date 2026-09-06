"""Tests for ``akgentic.team.scripts.migrate_yaml`` — story 31-5, AC 15-18.

Needs no service of any kind: the YAML store is a directory. Runs in the
default gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, NoReturn

import pytest
import yaml

import akgentic.team.scripts.migrate_yaml as script
from akgentic.team.repositories.yaml import CARDS_DIRNAME, YamlEventStore
from tests.models.conftest import make_process, to_legacy_document
from tests.scripts.test_migration import _legacy_document, _team_card


def _write_team(data_dir: Path, document: dict[str, Any]) -> uuid.UUID:
    """Plant one raw stored document where ``YamlEventStore`` reads it."""
    team_id = uuid.UUID(str(document["team_id"]))
    team_dir = data_dir / str(team_id)
    team_dir.mkdir(parents=True, exist_ok=True)
    with open(team_dir / "team.yaml", "w") as handle:
        yaml.dump(document, handle, default_flow_style=False)
    return team_id


class TestMigrateYamlScriptConfiguration:
    """AC 16, 17: the ``--help`` text and the unusable-configuration exit code."""

    def test_help_carries_the_rollout_ordering_constraint(self) -> None:
        """AC 16: the operator learns WHEN to run this from ``--help`` alone."""
        text = script.build_parser().format_help()

        assert "BETWEEN stopping the old version and starting the new one" in text
        assert "--data-dir" in text

    def test_a_missing_data_dir_exits_2(self) -> None:
        """AC 17: argparse rejects the missing required argument with code 2."""
        with pytest.raises(SystemExit) as excinfo:
            script.main([])

        assert excinfo.value.code == 2

    def test_a_data_dir_that_is_not_a_directory_exits_2(self, tmp_path: Path) -> None:
        """AC 17: an unusable directory is configuration, not a failed document."""
        missing = tmp_path / "no-such-store"

        assert script.main(["--data-dir", str(missing)]) == 2

    def test_a_file_where_the_data_dir_should_be_exits_2(self, tmp_path: Path) -> None:
        """The other unusable shape: a path that exists but is not a directory."""
        not_a_dir = tmp_path / "store"
        not_a_dir.write_text("")

        assert script.main(["--data-dir", str(not_a_dir)]) == 2

    def test_an_unreadable_data_dir_exits_1_rather_than_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Story 31-6 AC 12: ``main`` fails like its two sibling scripts.

        The directory passes ``is_dir()`` — so this is past the exit-2 gate —
        and then raises on iteration, which is what a permissions change or a
        read-only mount looks like mid-run. Per-file read errors are already
        handled by ``_read_team_document``; this is the store-level failure it
        cannot see. The operator must get one of the documented 0/1/2 codes,
        not a traceback.
        """
        data_dir = tmp_path / "store"
        data_dir.mkdir()

        def _refuse(self: Path) -> NoReturn:
            raise PermissionError(f"Permission denied: {self}")

        monkeypatch.setattr(Path, "iterdir", _refuse)

        assert script.main(["--data-dir", str(data_dir)]) == 1


class TestMigrateYamlScriptRuns:
    """AC 15, 18: the script converts a real store and reports its exit code."""

    def test_an_unmigrated_store_migrates_and_exits_0(self, tmp_path: Path) -> None:
        """AC 15, 18: read raw, write through the store, exit 0."""
        card = _team_card()
        team_id = _write_team(tmp_path, _legacy_document(card))
        store = YamlEventStore(tmp_path)
        assert store.load_team(team_id) is None  # the legacy shape is unreadable

        assert script.main(["--data-dir", str(tmp_path)]) == 0

        migrated = store.load_team(team_id)
        assert migrated is not None
        assert migrated.entry_point.name == "lead"
        assert (tmp_path / CARDS_DIRNAME).is_dir()

    def test_a_second_run_is_a_no_op_and_still_exits_0(self, tmp_path: Path) -> None:
        """Idempotent at the script level, not only in the core."""
        _write_team(tmp_path, _legacy_document())

        assert script.main(["--data-dir", str(tmp_path)]) == 0
        first = sorted(path.name for path in (tmp_path / CARDS_DIRNAME).iterdir())
        assert script.main(["--data-dir", str(tmp_path)]) == 0

        assert sorted(path.name for path in (tmp_path / CARDS_DIRNAME).iterdir()) == first

    def test_an_empty_store_exits_0(self, tmp_path: Path) -> None:
        """Nothing to do is success, not a failure."""
        assert script.main(["--data-dir", str(tmp_path)]) == 0

    def test_a_failed_document_exits_1(self, tmp_path: Path) -> None:
        """AC 9: not aborting the run is not the same as reporting success."""
        broken = _legacy_document()
        broken["team_card"] = {"not": "a team card"}
        _write_team(tmp_path, broken)
        good_id = _write_team(tmp_path, _legacy_document())

        assert script.main(["--data-dir", str(tmp_path)]) == 1

        # The good one still converted — the failure did not abort the run.
        assert YamlEventStore(tmp_path).load_team(good_id) is not None

    def test_an_unparseable_team_file_is_a_failure_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        """A corrupted file is counted, and the rest of the store still migrates."""
        team_dir = tmp_path / str(uuid.uuid4())
        team_dir.mkdir(parents=True)
        (team_dir / "team.yaml").write_text("{ this is not: valid: yaml: [")
        good_id = _write_team(tmp_path, _legacy_document())

        assert script.main(["--data-dir", str(tmp_path)]) == 1
        assert YamlEventStore(tmp_path).load_team(good_id) is not None

    def test_the_card_directory_and_non_uuid_directories_are_skipped(
        self, tmp_path: Path
    ) -> None:
        """``agent_cards/`` is a deliberate sibling of the team directories.

        Walking into it would read card blobs as team documents and count every
        one of them as a failure.
        """
        (tmp_path / CARDS_DIRNAME).mkdir()
        (tmp_path / CARDS_DIRNAME / "deadbeef.yaml").write_text("role: Lead\n")
        (tmp_path / "not-a-team").mkdir()
        _write_team(tmp_path, _legacy_document())

        assert script.main(["--data-dir", str(tmp_path)]) == 0

    def test_an_already_migrated_store_needs_no_conversion(self, tmp_path: Path) -> None:
        """The realistic re-run: every document already carries the projection."""
        store = YamlEventStore(tmp_path)
        card = _team_card()
        process = make_process(team_card=card)
        store.save_team(process)

        assert script.main(["--data-dir", str(tmp_path)]) == 0

        reloaded = store.load_team(process.team_id)
        assert reloaded is not None
        assert reloaded.entry_point == process.entry_point

    def test_a_document_with_no_team_card_is_skipped(self, tmp_path: Path) -> None:
        """AC 7 at the script level: nothing to derive from is not a failure."""
        card = _team_card()
        document = to_legacy_document(make_process(team_card=card), card)
        document.pop("team_card")
        _write_team(tmp_path, document)

        assert script.main(["--data-dir", str(tmp_path)]) == 0


class TestMigrateYamlScriptModulePath:
    """The module is runnable as ``python -m akgentic.team.scripts.migrate_yaml``."""

    def test_python_dash_m_help_exits_0_and_shows_the_ordering(self) -> None:
        """Proves the module path resolves and ``main()`` reaches ``sys.exit``."""
        result = subprocess.run(
            [sys.executable, "-m", "akgentic.team.scripts.migrate_yaml", "--help"],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "BETWEEN stopping the old version" in result.stdout

    def test_python_dash_m_without_a_data_dir_exits_2(self) -> None:
        """The documented command with no configuration exits 2, not 1."""
        result = subprocess.run(
            [sys.executable, "-m", "akgentic.team.scripts.migrate_yaml"],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"
