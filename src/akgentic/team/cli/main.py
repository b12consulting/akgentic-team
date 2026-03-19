"""ak-team Typer CLI application via [cli] optional extra.

Provides the ``ak-team`` entry point with global options and subcommands
for managing team lifecycle instances.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console

from akgentic.team.cli._output import OutputFormat, render
from akgentic.team.ports import EventStore

__all__ = ["app"]

logger = logging.getLogger(__name__)
err_console = Console(stderr=True)


@dataclass
class GlobalState:
    """Shared state passed through Typer context.

    Attributes:
        data_dir: Root directory for team data files.
        format: Output format for command results.
        backend: Storage backend (yaml or mongodb).
        mongo_uri: MongoDB connection URI.
        mongo_db: MongoDB database name.
    """

    data_dir: Path = field(default_factory=lambda: Path("./data/"))
    format: OutputFormat = OutputFormat.table
    backend: str = "yaml"
    mongo_uri: str | None = None
    mongo_db: str | None = None


app = typer.Typer(name="ak-team", help="Manage team instances.")


@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Path = typer.Option(
        Path("./data/"),
        "--data-dir",
        help="Root directory for team data files.",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        help="Output format: table, json, or yaml.",
    ),
    backend: str = typer.Option(
        "yaml",
        "--backend",
        help="Storage backend: yaml or mongodb.",
    ),
    mongo_uri: str | None = typer.Option(
        None,
        "--mongo-uri",
        envvar="MONGO_URI",
        help="MongoDB connection URI (required when --backend=mongodb).",
    ),
    mongo_db: str | None = typer.Option(
        None,
        "--mongo-db",
        envvar="MONGO_DB",
        help="MongoDB database name (required when --backend=mongodb).",
    ),
) -> None:
    """Akgentic team CLI -- manage team lifecycle instances."""
    valid_backends = ("yaml", "mongodb")
    if backend not in valid_backends:
        err_console.print(
            f"[red]Error:[/red] Invalid backend '{backend}'. "
            f"Must be one of: {', '.join(valid_backends)}"
        )
        raise typer.Exit(code=1)

    if backend == "mongodb":
        errors = _validate_mongodb_options(mongo_uri, mongo_db)
        if errors:
            for err in errors:
                err_console.print(f"[red]Error:[/red] {err}")
            raise typer.Exit(code=1)

    ctx.ensure_object(dict)
    ctx.obj = GlobalState(
        data_dir=data_dir,
        format=fmt,
        backend=backend,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
    )


def _validate_mongodb_options(
    mongo_uri: str | None,
    mongo_db: str | None,
) -> list[str]:
    """Validate MongoDB connection options and return error messages.

    Args:
        mongo_uri: MongoDB connection URI, or None if not provided.
        mongo_db: MongoDB database name, or None if not provided.

    Returns:
        List of error message strings (empty if valid).
    """
    errors: list[str] = []
    if not mongo_uri:
        errors.append(
            "--mongo-uri (or MONGO_URI env var) is required when --backend=mongodb"
        )
    if not mongo_db:
        errors.append(
            "--mongo-db (or MONGO_DB env var) is required when --backend=mongodb"
        )
    return errors


def _build_event_store(state: GlobalState) -> EventStore:
    """Construct an EventStore from the global CLI state.

    Args:
        state: The global state containing backend configuration.

    Returns:
        An EventStore implementation matching the configured backend.
    """
    if state.backend == "mongodb":
        import pymongo

        client: pymongo.MongoClient = pymongo.MongoClient(state.mongo_uri)  # type: ignore[type-arg]
        db = client[state.mongo_db]  # type: ignore[index]
        from akgentic.team.repositories.mongo import MongoEventStore

        return MongoEventStore(db)

    from akgentic.team.repositories.yaml import YamlEventStore

    return YamlEventStore(state.data_dir)


def _get_state(ctx: typer.Context) -> GlobalState:
    """Retrieve the global state from the Typer context.

    Args:
        ctx: The current Typer command context.

    Returns:
        The GlobalState instance stored in ctx.obj, or a default.
    """
    state: GlobalState = ctx.obj
    if state is None:
        state = GlobalState()
    return state


@app.command(name="list")
def list_teams_cmd(
    ctx: typer.Context,
    status: str | None = typer.Option(
        None,
        "--status",
        help="Filter by team status: running, stopped, or deleted.",
    ),
) -> None:
    """List all team instances."""
    state = _get_state(ctx)
    event_store = _build_event_store(state)
    teams = event_store.list_teams()

    if status is not None:
        valid_statuses = ("running", "stopped", "deleted")
        if status not in valid_statuses:
            err_console.print(
                f"[red]Error:[/red] Invalid status '{status}'. "
                f"Must be one of: {', '.join(valid_statuses)}"
            )
            raise typer.Exit(code=1)
        teams = [t for t in teams if t.status.value == status]

    render(teams, state.format)


@app.command(name="inspect")
def inspect_cmd(
    ctx: typer.Context,
    team_id: str = typer.Argument(help="Team UUID to inspect."),
) -> None:
    """Inspect a team instance by ID."""
    try:
        parsed_id = uuid.UUID(team_id)
    except ValueError:
        err_console.print(
            f"[red]Error:[/red] Invalid UUID format: '{team_id}'"
        )
        raise typer.Exit(code=1)  # noqa: B904

    state = _get_state(ctx)
    event_store = _build_event_store(state)
    process = event_store.load_team(parsed_id)

    if process is None:
        err_console.print(
            f"[red]Error:[/red] Team '{team_id}' not found."
        )
        raise typer.Exit(code=1)

    event_count = len(event_store.load_events(parsed_id))
    agent_state_count = len(event_store.load_agent_states(parsed_id))

    render(
        process,
        state.format,
        event_count=event_count,
        agent_state_count=agent_state_count,
    )
