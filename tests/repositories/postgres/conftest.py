"""Fixtures for the Nagra-backed Postgres event-store tests.

Module-level ``pytest.importorskip`` calls cleanly skip the entire
directory when ``nagra`` or ``testcontainers[postgres]`` is missing —
mirrors the Mongo behaviour in ``tests/repositories/mongo/conftest.py``
and the catalog Postgres conftest.

Session-scoped fixtures (``postgres_container``, ``postgres_conn_string``,
``postgres_initialized``) live in this module rather than the package-root
``tests/conftest.py`` because no other ``tests/`` subdirectory currently
needs them. If a ``tests/api/`` or ``tests/cli/`` Postgres dependency
appears later, lift them up the same way the catalog package did
(see catalog ``tests/conftest.py`` for the pattern).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("nagra")
pytest.importorskip("testcontainers.postgres")

from testcontainers.postgres import PostgresContainer  # noqa: E402


def _to_nagra_conn_string(sqlalchemy_url: str) -> str:
    """Strip the SQLAlchemy driver suffix from a testcontainers URL.

    ``testcontainers`` emits URLs like
    ``postgresql+psycopg2://user:pw@host:port/db``. Nagra's ``Transaction``
    wraps a psycopg / libpq connection, which accepts the standard
    ``postgresql://`` scheme without the driver suffix. Strip the driver
    so the URL is portable regardless of Nagra's current psycopg binding.
    """
    if "+" in sqlalchemy_url.split("://", 1)[0]:
        scheme, rest = sqlalchemy_url.split("://", 1)
        scheme = scheme.split("+", 1)[0]
        return f"{scheme}://{rest}"
    return sqlalchemy_url


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Start a single ``postgres:16-alpine`` container for the test session."""
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_conn_string(postgres_container: PostgresContainer) -> str:
    """Nagra-compatible connection string derived from the session container."""
    raw_url = postgres_container.get_connection_url()
    return _to_nagra_conn_string(raw_url)


@pytest.fixture(scope="session")
def postgres_initialized(postgres_conn_string: str) -> str:
    """Run :func:`init_db` exactly once against the session container."""
    from akgentic.team.repositories.postgres import init_db

    init_db(postgres_conn_string)
    return postgres_conn_string


@pytest.fixture
def postgres_clean_tables(postgres_initialized: str) -> Iterator[str]:
    """Truncate the three team event-store tables between tests."""
    from nagra import Transaction

    yield postgres_initialized
    with Transaction(postgres_initialized) as trn:
        trn.execute(
            "TRUNCATE team_process_entries, event_entries, agent_state_entries"
        )
