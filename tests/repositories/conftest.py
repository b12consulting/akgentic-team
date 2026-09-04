"""Fixtures for the parametrized ``EventStore`` contract suite.

Hosts the ``event_store`` fixture the shared ``TestEventStoreContract`` runs
against, and ``seed_raw_team``, which writes a document into whichever backend
that fixture yielded **without** going through ``save_team``.

The per-backend fixtures it composes (``mongo_store``,
``postgres_clean_tables``) live one level up in ``tests/conftest.py``: the
migration-script specs under ``tests/scripts/`` need the same backends, and
pytest's fixture lookup only walks leaf-to-root, so a fixture defined here
would be invisible to them.

Skip semantics mirror the per-backend behaviour:

* ``yaml`` always runs (pure stdlib + Pydantic).
* ``mongo`` requires ``pymongo`` and ``mongomock`` (both in the ``dev`` extra).
* ``postgres`` requires ``nagra``, ``psycopg``, ``testcontainers[postgres]``
  **and a running Docker daemon** — ``importorskip`` covers the imports only,
  so a machine with the packages and no Docker errors here rather than
  skipping. Pre-existing, and true of every test in this suite.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

RawTeamSeeder = Callable[[uuid.UUID, dict[str, Any]], None]
"""Writes one raw team document straight into the backend's storage."""


@pytest.fixture(params=["yaml", "mongo", "postgres"])
def event_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[EventStore]:
    """Yield a fresh ``EventStore`` for each backend parameter.

    The fixture id matches the parameter string (``yaml`` / ``mongo`` /
    ``postgres``) so pytest output identifies the backend at a glance.
    Each branch yields a clean store per test.
    """
    backend = request.param
    if backend == "yaml":
        from akgentic.team.repositories.yaml import YamlEventStore

        yield YamlEventStore(tmp_path)
        return

    if backend == "mongo":
        pytest.importorskip("pymongo")
        pytest.importorskip("mongomock")
        store = request.getfixturevalue("mongo_store")
        yield store
        return

    if backend == "postgres":
        pytest.importorskip("nagra")
        pytest.importorskip("psycopg")
        pytest.importorskip("testcontainers.postgres")
        from akgentic.team.repositories.postgres import NagraEventStore

        conn = request.getfixturevalue("postgres_clean_tables")
        yield NagraEventStore(conn)
        return

    msg = f"Unknown event_store backend: {backend}"
    raise ValueError(msg)


@pytest.fixture
def seed_raw_team(
    request: pytest.FixtureRequest,
    event_store: EventStore,
    tmp_path: Path,
) -> RawTeamSeeder:
    """Return a function that plants a raw document in the yielded backend.

    ``save_team`` cannot seed an *unloadable* document — it takes a validated
    ``Process`` — so the corrupted-document contract can only be exercised by
    writing past it, per backend, exactly where that backend reads from.

    The backend is read off the ``event_store`` parametrization rather than
    passed in, so one contract test covers all three.
    """
    del event_store  # requested so the backend's storage exists and is clean
    backend: str = request.node.callspec.params["event_store"]

    def _seed(team_id: uuid.UUID, document: dict[str, Any]) -> None:
        if backend == "yaml":
            team_dir = tmp_path / str(team_id)
            team_dir.mkdir(parents=True, exist_ok=True)
            with open(team_dir / "team.yaml", "w") as handle:
                yaml.dump(document, handle, default_flow_style=False)
            return
        if backend == "mongo":
            mongo_db = request.getfixturevalue("mongo_db")
            from akgentic.team.repositories.mongo import TEAMS_COLLECTION

            # A copy: insert_one stamps ``_id`` onto the mapping it is handed.
            mongo_db[TEAMS_COLLECTION].insert_one(dict(document))
            return
        conn = request.getfixturevalue("postgres_clean_tables")
        from nagra import Transaction  # type: ignore[import-untyped]

        with Transaction(conn) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data, metadata_indexes) "
                "VALUES (%s, %s, %s)",
                (str(team_id), json.dumps(document), []),
            )

    return _seed
