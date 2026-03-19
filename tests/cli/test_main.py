"""Tests for the ak-team CLI commands.

Validates list, inspect, and global options using Typer CliRunner
with YamlEventStore backed by temporary directories.

Acceptance Criteria: AC1-AC11 from Story 6.1.
"""

from __future__ import annotations

import json
import uuid

import yaml
from typer.testing import CliRunner

from akgentic.team.cli.main import app
from akgentic.team.models import TeamStatus
from akgentic.team.repositories.yaml import YamlEventStore

from tests.cli.conftest import populate_teams
from tests.models.conftest import (
    make_agent_state_snapshot,
    make_persisted_event,
    make_process,
)


class TestListCommand:
    """Tests for `ak-team list` command (AC2, AC3, AC4, AC5)."""

    def test_list_with_no_teams_returns_empty(
        self, cli_runner: CliRunner, data_dir: object
    ) -> None:
        """AC4: list with no teams shows empty output, exit code 0."""
        result = cli_runner.invoke(app, ["--data-dir", str(data_dir), "list"])
        assert result.exit_code == 0
        assert "No teams found" in result.output

    def test_list_with_teams_shows_all(
        self, cli_runner: CliRunner, data_dir: object, yaml_store: YamlEventStore
    ) -> None:
        """AC2: list with teams shows all team names and IDs."""
        teams = populate_teams(yaml_store, count=2)
        result = cli_runner.invoke(app, ["--data-dir", str(data_dir), "list"])
        assert result.exit_code == 0
        for t in teams:
            # Table truncates to 8 chars
            assert str(t.team_id)[:8] in result.output
            assert t.team_card.name in result.output

    def test_list_status_filter_running(
        self, cli_runner: CliRunner, data_dir: object, yaml_store: YamlEventStore
    ) -> None:
        """AC3: list --status running filters correctly."""
        running = populate_teams(yaml_store, count=1, status=TeamStatus.RUNNING)
        stopped = populate_teams(yaml_store, count=1, status=TeamStatus.STOPPED)
        result = cli_runner.invoke(
            app, ["--data-dir", str(data_dir), "list", "--status", "running"]
        )
        assert result.exit_code == 0
        assert str(running[0].team_id)[:8] in result.output
        assert str(stopped[0].team_id)[:8] not in result.output

    def test_list_format_json(
        self, cli_runner: CliRunner, data_dir: object, yaml_store: YamlEventStore
    ) -> None:
        """AC5: list --format json produces valid JSON."""
        populate_teams(yaml_store, count=1)
        result = cli_runner.invoke(
            app, ["--data-dir", str(data_dir), "--format", "json", "list"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_list_format_yaml(
        self, cli_runner: CliRunner, data_dir: object, yaml_store: YamlEventStore
    ) -> None:
        """AC5: list --format yaml produces valid YAML."""
        populate_teams(yaml_store, count=1)
        result = cli_runner.invoke(
            app, ["--data-dir", str(data_dir), "--format", "yaml", "list"]
        )
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1


class TestInspectCommand:
    """Tests for `ak-team inspect` command (AC6, AC7)."""

    def test_inspect_existing_team(
        self, cli_runner: CliRunner, data_dir: object, yaml_store: YamlEventStore
    ) -> None:
        """AC6: inspect shows full metadata for existing team."""
        process = make_process()
        yaml_store.save_team(process)
        yaml_store.save_event(
            make_persisted_event(team_id=process.team_id, sequence=1)
        )
        yaml_store.save_agent_state(
            make_agent_state_snapshot(team_id=process.team_id, agent_id="agent-a")
        )

        result = cli_runner.invoke(
            app, ["--data-dir", str(data_dir), "inspect", str(process.team_id)]
        )
        assert result.exit_code == 0
        assert str(process.team_id) in result.output
        assert process.team_card.name in result.output

    def test_inspect_nonexistent_team(
        self, cli_runner: CliRunner, data_dir: object
    ) -> None:
        """AC7: inspect non-existent team shows error, exit code 1."""
        fake_id = str(uuid.uuid4())
        result = cli_runner.invoke(
            app, ["--data-dir", str(data_dir), "inspect", fake_id]
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_inspect_invalid_uuid(
        self, cli_runner: CliRunner, data_dir: object
    ) -> None:
        """AC7: inspect with invalid UUID shows error, exit code 1."""
        result = cli_runner.invoke(
            app, ["--data-dir", str(data_dir), "inspect", "not-a-uuid"]
        )
        assert result.exit_code == 1
        assert "Invalid UUID" in result.output

    def test_inspect_json_format(
        self, cli_runner: CliRunner, data_dir: object, yaml_store: YamlEventStore
    ) -> None:
        """AC5, AC6: inspect --format json includes event and agent state counts."""
        process = make_process()
        yaml_store.save_team(process)
        yaml_store.save_event(
            make_persisted_event(team_id=process.team_id, sequence=1)
        )

        result = cli_runner.invoke(
            app,
            [
                "--data-dir", str(data_dir),
                "--format", "json",
                "inspect", str(process.team_id),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["event_count"] == 1
        assert parsed["agent_state_count"] == 0


class TestGlobalOptions:
    """Tests for global CLI options (AC1, AC8)."""

    def test_invalid_backend(self, cli_runner: CliRunner) -> None:
        """AC1: Invalid backend shows error, exit code 1."""
        result = cli_runner.invoke(
            app, ["--backend", "sqlite", "list"]
        )
        assert result.exit_code == 1
        assert "Invalid backend" in result.output

    def test_mongodb_backend_without_mongo_uri(self, cli_runner: CliRunner) -> None:
        """AC1: --backend mongodb without --mongo-uri shows error."""
        result = cli_runner.invoke(
            app, ["--backend", "mongodb", "list"]
        )
        assert result.exit_code == 1
        assert "--mongo-uri" in result.output
