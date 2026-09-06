"""Shape tests for ``schema.toml`` (AC #3).

Parse the file with ``tomllib`` directly — no Nagra or Postgres container
needed — so they run in every environment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

SCHEMA_PATH = (
    Path(__file__).parents[3]
    / "src"
    / "akgentic"
    / "team"
    / "repositories"
    / "postgres"
    / "schema.toml"
)

EXPECTED_TABLES = {
    "team_process_entries",
    "event_entries",
    "agent_state_entries",
    "agent_card_entries",
}


def _load_schema() -> dict[str, object]:
    with SCHEMA_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"schema.toml missing at {SCHEMA_PATH}"


def test_schema_has_exactly_four_top_level_tables() -> None:
    schema = _load_schema()
    assert set(schema.keys()) == EXPECTED_TABLES


def test_team_process_entries_shape() -> None:
    schema = _load_schema()
    table = schema["team_process_entries"]
    assert isinstance(table, dict)
    assert table["natural_key"] == ["id"]
    columns = table["columns"]
    assert isinstance(columns, dict)
    # ``metadata_indexes`` is the table's first promoted non-key column: the
    # derived metadata index, declared here as ``str[]`` so the postgresql
    # flavor emits ``TEXT[]`` and the GIN index in ``init_db`` has something
    # to index. The value itself still lives in ``data``.
    assert columns == {"id": "str", "data": "json", "metadata_indexes": "str[]"}


def test_event_entries_shape() -> None:
    schema = _load_schema()
    table = schema["event_entries"]
    assert isinstance(table, dict)
    assert table["natural_key"] == ["team_id", "sequence"]
    columns = table["columns"]
    assert isinstance(columns, dict)
    assert columns == {"team_id": "str", "sequence": "int", "data": "json"}


def test_agent_state_entries_shape() -> None:
    schema = _load_schema()
    table = schema["agent_state_entries"]
    assert isinstance(table, dict)
    assert table["natural_key"] == ["team_id", "agent_id"]
    columns = table["columns"]
    assert isinstance(columns, dict)
    assert columns == {"team_id": "str", "agent_id": "str", "data": "json"}


def test_agent_card_entries_shape() -> None:
    """The content-addressed card store: keyed by hash, with NO team_id.

    A ``team_id`` column here would be the whole design gone — the store is
    shared across every team that references a card, which is what makes
    "one blob per card" and "never deleted with a team" possible at all.
    """
    schema = _load_schema()
    table = schema["agent_card_entries"]
    assert isinstance(table, dict)
    assert table["natural_key"] == ["card_hash"]
    columns = table["columns"]
    assert isinstance(columns, dict)
    assert columns == {"card_hash": "str", "data": "json"}
