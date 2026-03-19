"""Test fixtures for MongoDB repository tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mongomock
import pytest

if TYPE_CHECKING:
    from akgentic.team.repositories.mongo import MongoEventStore


@pytest.fixture
def mongo_client() -> mongomock.MongoClient:
    """Create a mongomock client for testing."""
    return mongomock.MongoClient()


@pytest.fixture
def mongo_db(mongo_client: mongomock.MongoClient) -> mongomock.Database:
    """Create a test database from the mongomock client."""
    return mongo_client["test_akgentic_team"]


@pytest.fixture
def mongo_store(mongo_db: mongomock.Database) -> MongoEventStore:
    """Create a MongoEventStore backed by a mongomock database."""
    from akgentic.team.repositories.mongo import MongoEventStore

    return MongoEventStore(mongo_db)
