"""Shared test fixtures for akgentic-team tests.

Holds the projection helpers every hand-built ``Process`` routes through, and
the per-backend store fixtures (``mongo_store``, ``postgres_clean_tables`` and
what they compose). The backend fixtures live at THIS level, rather than under
``tests/repositories/``, because pytest's fixture lookup only walks
leaf-to-root: both the parametrized ``event_store`` fixture in
``tests/repositories/conftest.py`` and the migration-script specs in
``tests/scripts/`` need them, and neither can see fixtures defined in the
other's directory.

Skip semantics:

* ``mongo`` requires ``pymongo`` and ``mongomock`` (both in the ``dev``
  extra); ``mongo_store`` builds a fresh mongomock database per test, so no
  truncation is needed.
* ``postgres`` requires ``nagra``, ``psycopg`` and ``testcontainers[postgres]``
  **and a running Docker daemon**. The session-scoped container is started once
  and the four tables are truncated between tests by ``postgres_clean_tables``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from akgentic.team.models import TeamCard
from akgentic.team.projection import derive_team_projection

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore
    from akgentic.team.repositories.mongo import MongoEventStore


def projection_kwargs(team_card: TeamCard) -> dict[str, Any]:
    """Return the seven ``Process`` projection fields derived from *team_card*.

    Every test that constructs a ``Process`` by hand routes through here, so no
    fixture can hand-write a projection that the real write path would never
    produce — and none of them has to be revisited when the projection grows a
    field.
    """
    projection = derive_team_projection(team_card)
    return {
        "team_name": projection.team_name,
        "team_description": projection.team_description,
        "entry_point": projection.entry_point,
        "supervisors": projection.supervisors,
        "agent_cards": projection.agent_cards,
        "message_types": projection.message_types,
        "metadata_type": projection.metadata_type,
    }


def seed_agent_cards(store: EventStore, team_card: TeamCard) -> None:
    """Seed *store* with the cards a real ``create_team`` would have written.

    The counterpart of ``projection_kwargs``: that helper produces the
    ``AgentCardRef``s a hand-built ``Process`` carries, and this one puts the
    blobs those refs point at into the store — through the same derivation, so
    the hashes agree by construction rather than by a hand-written fixture that
    the real write path would never produce.

    Both halves are needed together. A ``Process`` seeded without its cards is a
    document whose hashes resolve against nothing, which restore now fails on
    loudly (``AgentCardNotFoundError``) instead of silently restoring a team the
    orchestrator cannot describe.
    """
    store.save_agent_cards(derive_team_projection(team_card).cards)


# --- Mongo fixtures -----------------------------------------------------


@pytest.fixture
def mongo_client() -> Any:
    """Create a mongomock client for testing."""
    pytest.importorskip("mongomock")
    import mongomock

    return mongomock.MongoClient()


@pytest.fixture
def mongo_db(mongo_client: Any) -> Any:
    """Create a test database from the mongomock client."""
    return mongo_client["test_akgentic_team"]


@pytest.fixture
def mongo_store(mongo_db: Any) -> MongoEventStore:
    """Create a MongoEventStore backed by a mongomock database."""
    pytest.importorskip("pymongo")
    from akgentic.team.repositories.mongo import MongoEventStore

    return MongoEventStore(mongo_db)


# --- Postgres fixtures --------------------------------------------------


def _to_nagra_conn_string(sqlalchemy_url: str) -> str:
    """Strip the SQLAlchemy driver suffix from a testcontainers URL."""
    if "+" in sqlalchemy_url.split("://", 1)[0]:
        scheme, rest = sqlalchemy_url.split("://", 1)
        scheme = scheme.split("+", 1)[0]
        return f"{scheme}://{rest}"
    return sqlalchemy_url


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[Any]:
    """Start a single ``postgres:16-alpine`` container for the test session."""
    pytest.importorskip("testcontainers.postgres")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_conn_string(postgres_container: Any) -> str:
    """Nagra-compatible connection string derived from the session container."""
    raw_url = postgres_container.get_connection_url()
    return _to_nagra_conn_string(raw_url)


@pytest.fixture(scope="session")
def postgres_initialized(postgres_conn_string: str) -> str:
    """Run :func:`init_db` exactly once against the session container."""
    pytest.importorskip("nagra")
    from akgentic.team.repositories.postgres import init_db

    init_db(postgres_conn_string)
    return postgres_conn_string


@pytest.fixture
def postgres_clean_tables(postgres_initialized: str) -> Iterator[str]:
    """Truncate the four team event-store tables between tests.

    ``agent_card_entries`` is included even though ``delete_team`` never touches
    it: the store is deliberately shared across teams, so without a truncate a
    card seeded by one test would still resolve in the next and a
    "hash absent from the store" assertion would pass for the wrong reason.
    """
    from nagra import Transaction  # type: ignore[import-untyped]

    yield postgres_initialized
    with Transaction(postgres_initialized) as trn:
        trn.execute(
            "TRUNCATE team_process_entries, event_entries, agent_state_entries, "
            "agent_card_entries"
        )
